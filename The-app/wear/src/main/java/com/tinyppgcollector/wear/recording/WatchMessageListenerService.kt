package com.tinyppgcollector.wear.recording

import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.WearableListenerService
import com.tinyppgcollector.wear.comm.PhoneMessageClient
import com.tinyppgcollector.wear.protocol.WearProtocol
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class WatchMessageListenerService : WearableListenerService() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var phoneMessageClient: PhoneMessageClient

    override fun onCreate() {
        super.onCreate()
        phoneMessageClient = PhoneMessageClient(this)
        Log.i(TAG, "WatchMessageListenerService created")
    }

    override fun onMessageReceived(event: MessageEvent) {
        Log.i(TAG, "Incoming message path=${event.path} bytes=${event.data.size}")
        when (event.path) {
            WearProtocol.START_SESSION -> {
                WatchRecordingStateStore.command("START_SESSION")
                val payload = event.data.toString(Charsets.UTF_8)
                runCatching {
                    ContextCompat.startForegroundService(
                        this,
                        WatchRecordingService.startSessionIntent(this, payload),
                    )
                }.onFailure { throwable ->
                    sendForwardingError("START_SESSION", throwable)
                }
            }

            WearProtocol.STOP_SESSION -> {
                WatchRecordingStateStore.command("STOP_SESSION")
                val payload = event.data.toString(Charsets.UTF_8)
                runCatching {
                    startService(WatchRecordingService.stopSessionIntent(this, payload))
                }.onFailure { throwable ->
                    sendForwardingError("STOP_SESSION", throwable)
                }
            }

            WearProtocol.PING -> {
                WatchRecordingStateStore.command("PING")
                sendStatusSnapshot("PING received")
            }

            WearProtocol.WATCH_STATUS_COMMAND -> {
                WatchRecordingStateStore.command("WATCH_STATUS")
                sendStatusSnapshot("WATCH_STATUS received")
            }
        }
    }

    private fun sendStatusSnapshot(message: String) {
        scope.launch {
            val state = WatchRecordingStateStore.state.value
            phoneMessageClient.sendStatus(
                recording = state.recording,
                ppgAvailable = state.ppgAvailable,
                imuAvailable = state.imuAvailable,
                samsungSdkAvailable = SamsungPpgRecorder(this@WatchMessageListenerService).samsungSensorSdkAvailable,
                imuSensorAvailable = SamsungImuRecorder(this@WatchMessageListenerService).isAvailable,
                ppgSamples = state.ppgSamples,
                imuSamples = state.imuSamples,
                message = message,
            )
        }
    }

    private fun sendForwardingError(command: String, throwable: Throwable) {
        val message = "$command forwarding failed: ${throwable.message ?: throwable.javaClass.simpleName}"
        Log.e(TAG, message, throwable)
        WatchRecordingStateStore.error(message)
        scope.launch {
            phoneMessageClient.sendError(message)
        }
    }

    override fun onDestroy() {
        Log.i(TAG, "WatchMessageListenerService destroyed")
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "TinyPPGWatch"
    }
}
