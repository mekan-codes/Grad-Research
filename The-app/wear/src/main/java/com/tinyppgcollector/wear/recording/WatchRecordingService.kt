package com.tinyppgcollector.wear.recording

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.tinyppgcollector.wear.WearMainActivity
import com.tinyppgcollector.wear.comm.PhoneMessageClient
import com.tinyppgcollector.wear.model.WatchSessionCommand
import com.tinyppgcollector.wear.permissions.WatchPermissionManager
import com.tinyppgcollector.wear.protocol.WearProtocol
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject

class WatchRecordingService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var phoneMessageClient: PhoneMessageClient
    private lateinit var ppgRecorder: SamsungPpgRecorder
    private lateinit var imuRecorder: SamsungImuRecorder
    private lateinit var ppgBuffer: WatchSampleBuffer
    private lateinit var imuBuffer: WatchSampleBuffer

    private var recording = false
    private var activeSessionId: String? = null
    private var startedSent = false
    private var foregroundStarted = false
    private var ppgActive = false
    private var imuActive = false
    private var statusJob: Job? = null
    private var flushRetryJob: Job? = null
    private val ppgSamples = AtomicLong(0L)
    private val imuSamples = AtomicLong(0L)

    override fun onCreate() {
        super.onCreate()
        phoneMessageClient = PhoneMessageClient(this)
        ppgRecorder = SamsungPpgRecorder(this)
        imuRecorder = SamsungImuRecorder(this)
        ppgBuffer = WatchSampleBuffer(
            scope = scope,
            phoneMessageClient = phoneMessageClient,
            path = WearProtocol.WATCH_PPG_BATCH,
            onBatchSent = ::onBatchSent,
        )
        imuBuffer = WatchSampleBuffer(
            scope = scope,
            phoneMessageClient = phoneMessageClient,
            path = WearProtocol.WATCH_IMU_BATCH,
            onBatchSent = ::onBatchSent,
        )
        updateUi("Service ready")
        Log.i(TAG, "WatchRecordingService created")
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_SESSION -> {
                val payload = intent.getStringExtra(EXTRA_PAYLOAD).orEmpty()
                val sessionId = payloadSessionId(payload) ?: "starting"
                WatchRecordingStateStore.command("START_SESSION")
                if (!foregroundStarted && !beginForeground(sessionId)) {
                    scope.launch {
                        sendWatchError("foreground service failed before START_SESSION")
                    }
                    stopSelf(startId)
                    return START_NOT_STICKY
                }
                scope.launch {
                    sendStatus("START_SESSION received")
                    runCatching { parseStartCommand(payload) }
                        .onSuccess { command -> startSession(command) }
                        .onFailure { throwable ->
                            sendWatchError(
                                "START_SESSION parse failed: ${throwable.message ?: throwable.javaClass.simpleName}",
                            )
                            finishStoppedService()
                        }
                }
                return START_STICKY
            }

            ACTION_STOP_SESSION -> {
                WatchRecordingStateStore.command("STOP_SESSION")
                scope.launch {
                    sendStatus("STOP_SESSION received")
                    stopSession()
                }
                return if (recording) START_STICKY else START_NOT_STICKY
            }

            ACTION_PING -> {
                WatchRecordingStateStore.command("PING")
                scope.launch {
                    sendStatus("PING received")
                    if (!recording) stopSelf(startId)
                }
            }

            ACTION_WATCH_STATUS -> {
                WatchRecordingStateStore.command("WATCH_STATUS")
                scope.launch {
                    sendStatus("WATCH_STATUS received")
                    if (!recording) stopSelf(startId)
                }
            }

            else -> Log.w(TAG, "Unknown service action=${intent?.action}")
        }
        return if (recording) START_STICKY else START_NOT_STICKY
    }

    private suspend fun startSession(command: WatchSessionCommand) {
        if (recording) stopSession()

        activeSessionId = command.sessionId
        startedSent = false
        ppgActive = false
        imuActive = false
        ppgSamples.set(0L)
        imuSamples.set(0L)
        flushRetryJob?.cancel()
        updateUi("START_SESSION received")

        val missingPermissions = WatchPermissionManager.missingPermissions(this)
        if (missingPermissions.isNotEmpty()) {
            val message = "missing permissions: ${missingPermissions.joinToString()}"
            sendWatchError("Sensor permission denied on watch: $message")
            sendStatus("missing permissions: $message")
            finishStoppedService()
            return
        }
        sendStatus("permissions OK")

        recording = true
        updateUi("recording true")
        if (!foregroundStarted && !startForegroundForRecording(command.sessionId)) {
            recording = false
            activeSessionId = null
            updateUi("foreground service failed")
            stopSelf()
            return
        }

        ppgBuffer.start()
        imuBuffer.start()
        startStatusTicker()

        startImu(command.sessionId)
        startSamsungPpg(command.sessionId)
        sendStatus("Watch recording start requested")
    }

    private suspend fun startImu(sessionId: String) {
        val imuResult = imuRecorder.start(sessionId) { sample ->
            val count = imuSamples.incrementAndGet()
            WatchRecordingStateStore.update {
                it.copy(
                    recording = recording,
                    imuAvailable = imuActive,
                    imuSamples = count,
                )
            }
            imuBuffer.add(sample.toCsvRow())
        }
        imuActive = imuResult.started
        Log.i(TAG, "IMU start result started=${imuResult.started} message=${imuResult.message}")
        if (imuResult.started) {
            sendStatus("IMU started")
            sendStartedIfReady(sessionId)
        } else {
            sendWatchError("IMU error: ${imuResult.message}")
            sendStatus("IMU error: ${imuResult.message}")
        }
        updateUi(if (imuResult.started) "IMU started" else "IMU error")
    }

    private suspend fun startSamsungPpg(sessionId: String) {
        val ppgResult = ppgRecorder.start(
            sessionId = sessionId,
            onSample = { sample ->
                val count = ppgSamples.incrementAndGet()
                WatchRecordingStateStore.update {
                    it.copy(
                        recording = recording,
                        ppgAvailable = ppgActive,
                        ppgSamples = count,
                    )
                }
                ppgBuffer.add(sample.toCsvRow())
            },
            onError = { message ->
                val error = "Samsung PPG error: $message"
                ppgActive = false
                Log.e(TAG, error)
                WatchRecordingStateStore.error(error)
                scope.launch {
                    phoneMessageClient.sendError(error)
                    sendStatus(error)
                }
            },
            onStatus = { message ->
                Log.i(TAG, "Samsung PPG status: $message")
                scope.launch {
                    if (message.contains("recording started", ignoreCase = true)) {
                        ppgActive = true
                        updateUi("Samsung PPG started")
                        sendStartedIfReady(sessionId)
                        sendStatus("Samsung PPG started")
                    } else {
                        sendStatus(message)
                    }
                }
            },
        )

        Log.i(TAG, "PPG start result started=${ppgResult.started} message=${ppgResult.message}")
        if (ppgResult.started) {
            sendStatus("Samsung PPG connection requested")
        } else {
            sendWatchError("Samsung PPG error: ${ppgResult.message}")
            sendStatus("Samsung PPG error: ${ppgResult.message}")
        }
    }

    private suspend fun stopSession() {
        if (!recording) {
            sendStatus("Watch idle")
            if (!phoneMessageClient.sendDoneAfterFlush()) {
                startFlushRetry()
            } else {
                WatchRecordingStateStore.status("WATCH_DONE sent")
                finishStoppedService()
            }
            return
        }

        val sessionId = activeSessionId
        recording = false
        updateUi("STOP_SESSION received")
        statusJob?.cancel()
        statusJob = null
        ppgRecorder.stop()
        imuRecorder.stop()
        ppgBuffer.stopAndFlush()
        imuBuffer.stopAndFlush()
        sendStatus("Watch recording stopped; flushed session=$sessionId")
        activeSessionId = sessionId
        if (phoneMessageClient.sendDoneAfterFlush()) {
            WatchRecordingStateStore.status("WATCH_DONE sent")
            finishStoppedService()
        } else {
            startFlushRetry()
        }
    }

    private fun startStatusTicker() {
        statusJob?.cancel()
        statusJob = scope.launch {
            while (isActive && recording) {
                delay(STATUS_INTERVAL_MS)
                sendStatus("Watch recording")
            }
        }
    }

    private suspend fun sendStartedIfReady(sessionId: String) {
        if (startedSent || !recording || !imuActive) return
        startedSent = true
        Log.i(TAG, "WATCH_STARTED sent session=$sessionId ppgActive=$ppgActive imuActive=$imuActive")
        phoneMessageClient.sendStarted(
            sessionId = sessionId,
            ppgAvailable = ppgActive,
            imuAvailable = imuActive,
            samsungSdkAvailable = ppgRecorder.samsungSensorSdkAvailable,
            imuSensorAvailable = imuRecorder.isAvailable,
            ppgSamples = ppgSamples.get(),
            imuSamples = imuSamples.get(),
            message = if (ppgActive) "Watch sensors active" else "Watch IMU active; Samsung PPG pending",
        )
        sendStatus(if (ppgActive) "Watch sensors active" else "Watch IMU active; Samsung PPG pending")
    }

    private suspend fun onBatchSent(path: String, rows: Int, delivered: Boolean) {
        val label = when (path) {
            WearProtocol.WATCH_PPG_BATCH -> "PPG"
            WearProtocol.WATCH_IMU_BATCH -> "IMU"
            else -> path
        }
        val message = "$label batch sent rows=$rows delivered=$delivered"
        Log.i(TAG, message)
        sendStatus(message)
    }

    private suspend fun sendWatchError(message: String) {
        Log.e(TAG, message)
        WatchRecordingStateStore.error(message)
        phoneMessageClient.sendError(message)
    }

    private fun startFlushRetry() {
        flushRetryJob?.cancel()
        flushRetryJob = scope.launch {
            while (isActive) {
                if (phoneMessageClient.sendDoneAfterFlush()) {
                    WatchRecordingStateStore.status("WATCH_DONE sent")
                    finishStoppedService()
                    return@launch
                }
                delay(FLUSH_RETRY_INTERVAL_MS)
            }
        }
    }

    private fun finishStoppedService() {
        flushRetryJob?.cancel()
        flushRetryJob = null
        recording = false
        activeSessionId = null
        startedSent = false
        ppgActive = false
        imuActive = false
        updateUi("Watch stopped")
        stopForegroundIfStarted()
        stopSelf()
    }

    private suspend fun sendStatus(message: String) {
        Log.i(
            TAG,
            "status=$message recording=$recording ppgActive=$ppgActive imuActive=$imuActive " +
                "ppgSamples=${ppgSamples.get()} imuSamples=${imuSamples.get()}",
        )
        updateUi(message)
        phoneMessageClient.sendStatus(
            recording = recording,
            ppgAvailable = ppgActive,
            imuAvailable = imuActive,
            samsungSdkAvailable = ppgRecorder.samsungSensorSdkAvailable,
            imuSensorAvailable = imuRecorder.isAvailable,
            ppgSamples = ppgSamples.get(),
            imuSamples = imuSamples.get(),
            message = message,
        )
    }

    private suspend fun startForegroundForRecording(sessionId: String): Boolean {
        return if (beginForeground(sessionId)) {
            sendStatus("foreground service started")
            true
        } else {
            sendWatchError("foreground service failed")
            false
        }
    }

    private fun beginForeground(sessionId: String): Boolean {
        return runCatching {
            val notification = buildNotification(sessionId)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification, foregroundServiceTypes())
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            foregroundStarted = true
        }.onFailure { throwable ->
            foregroundStarted = false
            val message = "foreground service failed: ${throwable.message ?: throwable.javaClass.simpleName}"
            Log.e(TAG, message, throwable)
            WatchRecordingStateStore.error(message)
        }.isSuccess
    }

    private fun foregroundServiceTypes(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC or ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH
        } else {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
    }

    private fun stopForegroundIfStarted() {
        if (!foregroundStarted) return
        runCatching { stopForeground(STOP_FOREGROUND_REMOVE) }
        foregroundStarted = false
    }

    private fun updateUi(message: String) {
        WatchRecordingStateStore.update {
            it.copy(
                readyStatus = message,
                recording = recording,
                ppgAvailable = ppgActive,
                imuAvailable = imuActive,
                ppgSamples = ppgSamples.get(),
                imuSamples = imuSamples.get(),
            )
        }
    }

    private fun parseStartCommand(payload: String): WatchSessionCommand {
        val json = JSONObject(payload)
        return WatchSessionCommand(
            sessionId = json.getString("session_id"),
            subjectId = json.optString("subject_id"),
            activityLabel = json.optString("activity_label"),
            phoneStartElapsedRealtimeNs = json.optLong("phone_start_elapsed_realtime_ns"),
        )
    }

    private fun payloadSessionId(payload: String): String? {
        return runCatching {
            JSONObject(payload).optString("session_id").takeIf { it.isNotBlank() }
        }.getOrNull()
    }

    private fun buildNotification(sessionId: String): Notification {
        ensureChannel()
        val intent = Intent(this, WearMainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("TinyPPG recording")
            .setContentText(sessionId)
            .setSmallIcon(android.R.drawable.ic_menu_upload)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Watch recording",
            NotificationManager.IMPORTANCE_LOW,
        )
        manager.createNotificationChannel(channel)
    }

    override fun onDestroy() {
        Log.i(TAG, "WatchRecordingService destroyed")
        statusJob?.cancel()
        flushRetryJob?.cancel()
        ppgRecorder.stop()
        imuRecorder.stop()
        recording = false
        ppgActive = false
        imuActive = false
        stopForegroundIfStarted()
        updateUi("Service destroyed")
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "TinyPPGWatch"
        private const val CHANNEL_ID = "tiny_ppg_watch_recording"
        private const val NOTIFICATION_ID = 2001
        private const val STATUS_INTERVAL_MS = 5_000L
        private const val FLUSH_RETRY_INTERVAL_MS = 1_000L
        private const val ACTION_START_SESSION = "com.tinyppgcollector.wear.action.START_SESSION"
        private const val ACTION_STOP_SESSION = "com.tinyppgcollector.wear.action.STOP_SESSION"
        private const val ACTION_PING = "com.tinyppgcollector.wear.action.PING"
        private const val ACTION_WATCH_STATUS = "com.tinyppgcollector.wear.action.WATCH_STATUS"
        private const val EXTRA_PAYLOAD = "payload"

        fun startSessionIntent(context: Context, payload: String): Intent {
            return Intent(context, WatchRecordingService::class.java)
                .setAction(ACTION_START_SESSION)
                .putExtra(EXTRA_PAYLOAD, payload)
        }

        fun stopSessionIntent(context: Context, payload: String): Intent {
            return Intent(context, WatchRecordingService::class.java)
                .setAction(ACTION_STOP_SESSION)
                .putExtra(EXTRA_PAYLOAD, payload)
        }

        fun pingIntent(context: Context): Intent {
            return Intent(context, WatchRecordingService::class.java)
                .setAction(ACTION_PING)
        }

        fun statusIntent(context: Context): Intent {
            return Intent(context, WatchRecordingService::class.java)
                .setAction(ACTION_WATCH_STATUS)
        }
    }
}
