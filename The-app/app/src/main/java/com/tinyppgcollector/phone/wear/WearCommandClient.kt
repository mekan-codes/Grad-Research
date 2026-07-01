package com.tinyppgcollector.phone.wear

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import java.io.Closeable
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject

class WearCommandClient(context: Context) : MessageClient.OnMessageReceivedListener, Closeable {
    private val appContext = context.applicationContext
    private val messageClient = Wearable.getMessageClient(appContext)
    private val nodeClient = Wearable.getNodeClient(appContext)

    private val _status = MutableStateFlow(WatchConnectionStatus(false, "Watch not connected"))
    val status: StateFlow<WatchConnectionStatus> = _status

    private val _incoming = MutableSharedFlow<WatchIncoming>(extraBufferCapacity = 64)
    val incoming: SharedFlow<WatchIncoming> = _incoming

    private val watchDoneCounter = MutableStateFlow(0L)
    private val watchStatusCounter = MutableStateFlow(0L)

    init {
        messageClient.addListener(this)
    }

    suspend fun pingWatch(): Boolean {
        val statusSnapshot = watchStatusCounter.value
        val sent = sendToAllNodes(
            WearProtocol.PING,
            JSONObject()
            .put("timestamp_unix_ms", System.currentTimeMillis())
            .put("elapsed_realtime_ns", SystemClock.elapsedRealtimeNanos())
            .toString()
            .toByteArray(),
        )
        if (!sent) return false

        val responded = withTimeoutOrNull(PING_TIMEOUT_MS) {
            watchStatusCounter.filter { it > statusSnapshot }.first()
            true
        } ?: false

        if (!responded) {
            val message = "Watch reachable, but TinyPPG watch app did not respond"
            Log.w(TAG, message)
            _status.value = WatchConnectionStatus(false, message)
        }
        return responded
    }

    suspend fun sendStartSession(
        sessionId: String,
        subjectId: String,
        activityLabel: String,
        phoneStartElapsedRealtimeNs: Long,
    ): Boolean {
        val payload = JSONObject()
            .put("session_id", sessionId)
            .put("subject_id", subjectId)
            .put("activity_label", activityLabel)
            .put("phone_start_elapsed_realtime_ns", phoneStartElapsedRealtimeNs)
            .toString()
            .toByteArray()
        return sendToAllNodes(WearProtocol.START_SESSION, payload)
    }

    suspend fun sendStopSession(sessionId: String): Boolean {
        val payload = JSONObject()
            .put("session_id", sessionId)
            .put("timestamp_unix_ms", System.currentTimeMillis())
            .put("elapsed_realtime_ns", SystemClock.elapsedRealtimeNanos())
            .toString()
            .toByteArray()
        return sendToAllNodes(WearProtocol.STOP_SESSION, payload)
    }

    fun watchDoneSnapshot(): Long = watchDoneCounter.value

    suspend fun waitForWatchDone(afterSnapshot: Long, timeoutMs: Long): Boolean {
        return withTimeoutOrNull(timeoutMs) {
            watchDoneCounter.filter { it > afterSnapshot }.first()
            true
        } ?: false
    }

    override fun onMessageReceived(event: MessageEvent) {
        val payload = event.data.toString(Charsets.UTF_8)
        Log.i(TAG, "Incoming watch message path=${event.path} bytes=${event.data.size}")
        when (event.path) {
            WearProtocol.WATCH_STATUS -> {
                val status = WatchStatusPayload.parse(payload)
                watchStatusCounter.value = watchStatusCounter.value + 1
                _status.value = WatchConnectionStatus(true, status.message.ifBlank { "Watch connected" })
                _incoming.tryEmit(WatchIncoming.Status(status))
            }

            WearProtocol.WATCH_STARTED -> {
                Log.i(TAG, "WATCH_STARTED payload=$payload")
                _incoming.tryEmit(WatchIncoming.Started(WatchStatusPayload.parse(payload)))
            }

            WearProtocol.WATCH_PPG_BATCH -> {
                Log.i(TAG, "WATCH_PPG_BATCH received rows=${payload.rowCount()}")
                _incoming.tryEmit(WatchIncoming.PpgBatch(payload))
            }

            WearProtocol.WATCH_IMU_BATCH -> {
                Log.i(TAG, "WATCH_IMU_BATCH received rows=${payload.rowCount()}")
                _incoming.tryEmit(WatchIncoming.ImuBatch(payload))
            }

            WearProtocol.WATCH_ERROR -> {
                Log.e(TAG, "WATCH_ERROR: ${payload.ifBlank { "Watch error" }}")
                _incoming.tryEmit(WatchIncoming.Error(payload.ifBlank { "Watch error" }))
            }

            WearProtocol.WATCH_DONE -> {
                Log.i(TAG, "WATCH_DONE received")
                watchDoneCounter.value = watchDoneCounter.value + 1
                _incoming.tryEmit(WatchIncoming.Done)
            }
        }
    }

    private suspend fun sendToAllNodes(path: String, payload: ByteArray): Boolean {
        return runCatching {
            val nodes = nodeClient.connectedNodes.awaitTask()
            if (nodes.isEmpty()) {
                _status.value = WatchConnectionStatus(false, "Watch not connected")
                Log.w(TAG, "No connected Wear nodes for path=$path")
                return false
            }
            nodes.forEach { node ->
                Log.i(TAG, "Sending path=$path to node=${node.displayName}/${node.id}")
                messageClient.sendMessage(node.id, path, payload).awaitTask()
            }
            true
        }.getOrElse { throwable ->
            Log.e(TAG, "Watch communication failed path=$path: ${throwable.message}", throwable)
            _status.value = WatchConnectionStatus(false, throwable.message ?: "Watch communication failed")
            false
        }
    }

    override fun close() {
        messageClient.removeListener(this)
    }

    private fun String.rowCount(): Int = lineSequence().count { it.isNotBlank() }

    companion object {
        private const val TAG = "TinyPPGPhone"
        private const val PING_TIMEOUT_MS = 3_000L
    }
}

private suspend fun <T> Task<T>.awaitTask(): T = suspendCancellableCoroutine { continuation ->
    addOnSuccessListener { result -> continuation.resume(result) }
    addOnFailureListener { throwable -> continuation.resumeWithException(throwable) }
    addOnCanceledListener { continuation.cancel() }
}
