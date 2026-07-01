package com.tinyppgcollector.wear.recording

import android.content.Context
import android.content.pm.PackageManager
import android.os.SystemClock
import android.util.Log
import androidx.core.content.ContextCompat
import com.samsung.android.service.health.tracking.ConnectionListener
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.HealthTrackerException
import com.samsung.android.service.health.tracking.HealthTrackingService
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import com.samsung.android.service.health.tracking.data.PpgType
import com.samsung.android.service.health.tracking.data.ValueKey
import com.samsung.android.service.health.tracking.data.ValueKey.PpgSet
import com.tinyppgcollector.wear.model.WatchPpgSample
import com.tinyppgcollector.wear.permissions.WatchPermissionManager
import java.util.concurrent.atomic.AtomicLong

data class RecorderStartResult(
    val started: Boolean,
    val message: String,
)

class SamsungPpgRecorder(private val context: Context) {
    private var healthTrackingService: HealthTrackingService? = null
    private var healthTracker: HealthTracker? = null
    private var activeSessionId: String? = null
    private var activeOnSample: ((WatchPpgSample) -> Unit)? = null
    private var activeOnError: ((String) -> Unit)? = null
    private var activeOnStatus: ((String) -> Unit)? = null
    @Volatile
    private var stopRequested = false
    private val sampleIndex = AtomicLong(0L)

    val samsungSensorSdkAvailable: Boolean
        get() = runCatching {
            HealthTrackingService::class.java
            HealthTrackerType.PPG_CONTINUOUS
            PpgType.GREEN
        }.isSuccess

    fun start(
        sessionId: String,
        onSample: (WatchPpgSample) -> Unit,
        onError: (String) -> Unit,
        onStatus: (String) -> Unit = {},
    ): RecorderStartResult {
        stop()
        stopRequested = false
        Log.i(TAG, "Samsung PPG start requested session=$sessionId")

        if (!hasSensorPermission()) {
            Log.e(TAG, "Samsung PPG permission denied")
            return RecorderStartResult(false, "Sensor permission denied on watch")
        }
        if (!samsungSensorSdkAvailable) {
            Log.e(TAG, "Samsung Health Sensor SDK unavailable")
            return RecorderStartResult(
                false,
                "Samsung Health Sensor SDK unavailable: add samsung-health-sensor-api.aar to wear/libs",
            )
        }

        activeSessionId = sessionId
        activeOnSample = onSample
        activeOnError = onError
        activeOnStatus = onStatus
        sampleIndex.set(0L)

        return runCatching {
            healthTrackingService = HealthTrackingService(
                createConnectionListener(),
                context.applicationContext,
            ).also { service ->
                service.connectService()
            }
            Log.i(TAG, "Samsung PPG HealthTrackingService connectService called")
            RecorderStartResult(true, "Samsung PPG tracker connection requested")
        }.getOrElse { throwable ->
            Log.e(TAG, "Samsung PPG start failed: ${throwable.message}", throwable)
            clearState()
            RecorderStartResult(false, "Samsung Health Sensor SDK unavailable: ${throwable.cleanMessage()}")
        }
    }

    fun stop() {
        Log.i(TAG, "Samsung PPG stop requested")
        stopRequested = true
        runCatching { healthTracker?.flush() }
        runCatching { healthTracker?.unsetEventListener() }
        runCatching { healthTrackingService?.disconnectService() }
        clearState()
        stopRequested = false
    }

    private fun createConnectionListener(): ConnectionListener {
        return object : ConnectionListener {
            override fun onConnectionSuccess() {
                Log.i(TAG, "Samsung Health Sensor SDK connection success")
                startPpgTracker()
            }

            override fun onConnectionEnded() {
                Log.i(TAG, "Samsung Health Sensor SDK connection ended stopRequested=$stopRequested")
                if (!stopRequested) {
                    activeOnError?.invoke("Samsung Health Sensor SDK connection ended")
                }
            }

            override fun onConnectionFailed(exception: HealthTrackerException) {
                Log.e(TAG, "Samsung Health Sensor SDK connection failed: ${exception.cleanMessage()}", exception)
                activeOnError?.invoke(
                    "Samsung Health Sensor SDK unavailable: ${exception.cleanMessage()}",
                )
            }
        }
    }

    private fun startPpgTracker() {
        val sessionId = activeSessionId ?: return
        val service = healthTrackingService ?: return
        runCatching {
            if (!isTrackerSupported(service)) {
                Log.e(TAG, "PPG_CONTINUOUS is not supported")
                activeOnError?.invoke("PPG tracker unavailable: PPG_CONTINUOUS is not supported on this watch")
                return
            }

            val ppgTypes = linkedSetOf(PpgType.GREEN, PpgType.RED, PpgType.IR)
            healthTracker = service.getHealthTracker(HealthTrackerType.PPG_CONTINUOUS, ppgTypes)
            healthTracker?.setEventListener(createTrackerListener(sessionId))
            Log.i(TAG, "Samsung PPG_CONTINUOUS event listener set ppgTypes=$ppgTypes")
            activeOnStatus?.invoke("Samsung PPG_CONTINUOUS recording started")
        }.onFailure { throwable ->
            Log.e(TAG, "PPG tracker unavailable: ${throwable.message}", throwable)
            activeOnError?.invoke("PPG tracker unavailable: ${throwable.cleanMessage()}")
        }
    }

    private fun isTrackerSupported(service: HealthTrackingService): Boolean {
        return service
            .getTrackingCapability()
            .getSupportHealthTrackerTypes()
            .contains(HealthTrackerType.PPG_CONTINUOUS)
    }

    private fun createTrackerListener(sessionId: String): HealthTracker.TrackerEventListener {
        return object : HealthTracker.TrackerEventListener {
            override fun onDataReceived(dataPoints: MutableList<DataPoint>) {
                if (dataPoints.isNotEmpty()) {
                    Log.d(TAG, "Samsung PPG data received rows=${dataPoints.size}")
                }
                dataPoints.forEach { dataPoint ->
                    activeOnSample?.invoke(readSample(sessionId, dataPoint))
                }
            }

            override fun onFlushCompleted() {
                Log.i(TAG, "Samsung PPG flush completed")
                activeOnStatus?.invoke("Samsung PPG tracker flushed")
            }

            override fun onError(error: HealthTracker.TrackerError) {
                Log.e(TAG, "Samsung PPG tracker error=$error")
                val message = when (error) {
                    HealthTracker.TrackerError.PERMISSION_ERROR ->
                        "Sensor permission denied on watch: Samsung PPG permission error"
                    HealthTracker.TrackerError.SDK_POLICY_ERROR ->
                        "PPG tracker unavailable: Samsung SDK policy error"
                }
                activeOnError?.invoke(message)
            }
        }
    }

    private fun readSample(sessionId: String, dataPoint: DataPoint): WatchPpgSample {
        val timestampUnixMs = readTimestampUnixMs(dataPoint)
        return WatchPpgSample(
            sessionId = sessionId,
            timestampUnixMs = timestampUnixMs,
            elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
            sensorTimestampNs = timestampUnixMs * NS_PER_MS,
            sampleIndex = sampleIndex.getAndIncrement(),
            ppgGreen = dataPoint.intValue(PpgSet.PPG_GREEN),
            ppgRed = dataPoint.intValue(PpgSet.PPG_RED),
            ppgIr = dataPoint.intValue(PpgSet.PPG_IR),
            status = buildStatus(dataPoint),
        )
    }

    private fun buildStatus(dataPoint: DataPoint): String {
        val parts = listOfNotNull(
            dataPoint.intValue(PpgSet.GREEN_STATUS)?.let { "green=$it" },
            dataPoint.intValue(PpgSet.RED_STATUS)?.let { "red=$it" },
            dataPoint.intValue(PpgSet.IR_STATUS)?.let { "ir=$it" },
        )
        return parts.joinToString(";")
    }

    private fun DataPoint.intValue(key: ValueKey<Int>): Int? {
        return runCatching { getValue(key) }.getOrNull()
    }

    private fun readTimestampUnixMs(dataPoint: DataPoint): Long {
        return runCatching { dataPoint.timestamp }
            .getOrNull()
            ?.takeIf { it > 0L }
            ?: System.currentTimeMillis()
    }

    private fun hasSensorPermission(): Boolean {
        return WatchPermissionManager.requiredPermissions().all { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun HealthTrackerException.cleanMessage(): String {
        val text = message ?: javaClass.simpleName
        return "$text (code=$errorCode)"
    }

    private fun Throwable.cleanMessage(): String = message ?: javaClass.simpleName

    private fun clearState() {
        healthTracker = null
        healthTrackingService = null
        activeSessionId = null
        activeOnSample = null
        activeOnError = null
        activeOnStatus = null
    }

    companion object {
        private const val TAG = "TinyPPGWatch"
        private const val NS_PER_MS = 1_000_000L
    }
}
