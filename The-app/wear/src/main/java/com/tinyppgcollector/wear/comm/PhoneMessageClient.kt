package com.tinyppgcollector.wear.comm

import android.content.Context
import android.util.Log
import com.google.android.gms.wearable.Wearable
import com.tinyppgcollector.wear.protocol.WearProtocol
import com.tinyppgcollector.wear.protocol.awaitTask
import java.util.ArrayDeque
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject

class PhoneMessageClient(context: Context) {
    private val appContext = context.applicationContext
    private val messageClient = Wearable.getMessageClient(appContext)
    private val nodeClient = Wearable.getNodeClient(appContext)
    private val pending = ArrayDeque<PendingMessage>()
    private val mutex = Mutex()
    private val flushMutex = Mutex()

    suspend fun sendStatus(
        recording: Boolean,
        ppgAvailable: Boolean,
        imuAvailable: Boolean,
        samsungSdkAvailable: Boolean,
        imuSensorAvailable: Boolean,
        ppgSamples: Long,
        imuSamples: Long,
        message: String,
    ) {
        val payload = JSONObject()
            .put("recording", recording)
            .put("ppg_available", ppgAvailable)
            .put("imu_available", imuAvailable)
            .put("samsung_sdk_available", samsungSdkAvailable)
            .put("imu_sensor_available", imuSensorAvailable)
            .put("ppg_samples", ppgSamples)
            .put("imu_samples", imuSamples)
            .put("watch_model", android.os.Build.MODEL)
            .put("message", message)
            .toString()
            .toByteArray()
        send(WearProtocol.WATCH_STATUS, payload)
    }

    suspend fun sendStarted(
        sessionId: String,
        ppgAvailable: Boolean,
        imuAvailable: Boolean,
        samsungSdkAvailable: Boolean,
        imuSensorAvailable: Boolean,
        ppgSamples: Long,
        imuSamples: Long,
        message: String,
    ) {
        val payload = JSONObject()
            .put("session_id", sessionId)
            .put("recording", true)
            .put("ppg_available", ppgAvailable)
            .put("imu_available", imuAvailable)
            .put("samsung_sdk_available", samsungSdkAvailable)
            .put("imu_sensor_available", imuSensorAvailable)
            .put("ppg_samples", ppgSamples)
            .put("imu_samples", imuSamples)
            .put("watch_model", android.os.Build.MODEL)
            .put("message", message)
            .toString()
            .toByteArray()
        send(WearProtocol.WATCH_STARTED, payload)
    }

    suspend fun sendError(message: String) {
        Log.e(TAG, "WATCH_ERROR: $message")
        send(WearProtocol.WATCH_ERROR, message.toByteArray())
    }

    suspend fun sendDoneAfterFlush(): Boolean {
        flushPending()
        if (hasPending()) return false
        send(WearProtocol.WATCH_DONE, ByteArray(0))
        flushPending()
        val sent = !hasPending()
        Log.i(TAG, "WATCH_DONE sent=$sent")
        return sent
    }

    suspend fun send(path: String, payload: ByteArray): Boolean {
        mutex.withLock {
            pending.add(PendingMessage(path, payload))
        }
        val delivered = flushPending()
        Log.i(TAG, "send path=$path bytes=${payload.size} delivered=$delivered")
        return delivered
    }

    suspend fun flushPending(): Boolean {
        flushMutex.withLock {
            val nodes = runCatching { nodeClient.connectedNodes.awaitTask() }.getOrElse { emptyList() }
            if (nodes.isEmpty()) {
                Log.w(TAG, "No connected phone nodes; pending=${pending.size}")
                return false
            }

            while (true) {
                val next = mutex.withLock { pending.peekFirst() } ?: return true
                val sent = runCatching {
                    nodes.forEach { node ->
                        messageClient.sendMessage(node.id, next.path, next.payload).awaitTask()
                    }
                }.onFailure { throwable ->
                    Log.e(TAG, "Failed sending ${next.path}: ${throwable.message}", throwable)
                }.isSuccess

                if (!sent) return false
                mutex.withLock { pending.removeFirst() }
            }
        }
    }

    suspend fun hasPending(): Boolean = mutex.withLock { pending.isNotEmpty() }

    private data class PendingMessage(
        val path: String,
        val payload: ByteArray,
    )

    companion object {
        private const val TAG = "TinyPPGWatch"
    }
}
