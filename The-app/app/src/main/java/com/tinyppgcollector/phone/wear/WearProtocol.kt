package com.tinyppgcollector.phone.wear

import org.json.JSONObject

object WearProtocol {
    private const val PREFIX = "/tinyppg"

    const val START_SESSION = "$PREFIX/start_session"
    const val STOP_SESSION = "$PREFIX/stop_session"
    const val PING = "$PREFIX/ping"
    const val WATCH_STATUS_COMMAND = "$PREFIX/watch_status_command"

    const val WATCH_STATUS = "$PREFIX/watch_status"
    const val WATCH_STARTED = "$PREFIX/watch_started"
    const val WATCH_PPG_BATCH = "$PREFIX/watch_ppg_batch"
    const val WATCH_IMU_BATCH = "$PREFIX/watch_imu_batch"
    const val WATCH_ERROR = "$PREFIX/watch_error"
    const val WATCH_DONE = "$PREFIX/watch_done"
}

data class WatchConnectionStatus(
    val connected: Boolean,
    val message: String,
)

data class WatchStatusPayload(
    val message: String,
    val recording: Boolean? = null,
    val ppgAvailable: Boolean? = null,
    val imuAvailable: Boolean? = null,
    val ppgSamples: Long? = null,
    val imuSamples: Long? = null,
    val watchModel: String? = null,
    val samsungSdkAvailable: Boolean? = null,
    val imuSensorAvailable: Boolean? = null,
) {
    companion object {
        fun parse(payload: String): WatchStatusPayload {
            val json = runCatching { JSONObject(payload) }.getOrNull()
            if (json == null) {
                return WatchStatusPayload(message = payload.ifBlank { "Watch connected" })
            }
            return WatchStatusPayload(
                message = json.optString("message", "Watch connected"),
                recording = json.optionalBoolean("recording"),
                ppgAvailable = json.optionalBoolean("ppg_available"),
                imuAvailable = json.optionalBoolean("imu_available"),
                ppgSamples = json.optionalLong("ppg_samples"),
                imuSamples = json.optionalLong("imu_samples"),
                watchModel = json.optString("watch_model").ifBlank { null },
                samsungSdkAvailable = json.optionalBoolean("samsung_sdk_available"),
                imuSensorAvailable = json.optionalBoolean("imu_sensor_available"),
            )
        }

        private fun JSONObject.optionalBoolean(name: String): Boolean? {
            return if (has(name) && !isNull(name)) optBoolean(name) else null
        }

        private fun JSONObject.optionalLong(name: String): Long? {
            return if (has(name) && !isNull(name)) optLong(name) else null
        }
    }
}

sealed interface WatchIncoming {
    data class PpgBatch(val rows: String) : WatchIncoming
    data class ImuBatch(val rows: String) : WatchIncoming
    data class Status(val status: WatchStatusPayload) : WatchIncoming
    data class Started(val status: WatchStatusPayload) : WatchIncoming
    data class Error(val message: String) : WatchIncoming
    data object Done : WatchIncoming
}
