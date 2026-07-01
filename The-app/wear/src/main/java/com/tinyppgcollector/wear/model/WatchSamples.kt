package com.tinyppgcollector.wear.model

data class WatchSessionCommand(
    val sessionId: String,
    val subjectId: String,
    val activityLabel: String,
    val phoneStartElapsedRealtimeNs: Long,
)

data class WatchPpgSample(
    val sessionId: String,
    val timestampUnixMs: Long,
    val elapsedRealtimeNs: Long,
    val sensorTimestampNs: Long?,
    val sampleIndex: Long,
    val ppgGreen: Int?,
    val ppgRed: Int?,
    val ppgIr: Int?,
    val status: String?,
) {
    fun toCsvRow(): String = listOf(
        sessionId,
        timestampUnixMs.toString(),
        elapsedRealtimeNs.toString(),
        sensorTimestampNs?.toString().orEmpty(),
        sampleIndex.toString(),
        ppgGreen?.toString().orEmpty(),
        ppgRed?.toString().orEmpty(),
        ppgIr?.toString().orEmpty(),
        status.orEmpty(),
    ).joinToString(",")
}

data class WatchImuSample(
    val sessionId: String,
    val timestampUnixMs: Long,
    val elapsedRealtimeNs: Long,
    val sensorTimestampNs: Long?,
    val sampleIndex: Long,
    val accX: Float,
    val accY: Float,
    val accZ: Float,
) {
    fun toCsvRow(): String = listOf(
        sessionId,
        timestampUnixMs.toString(),
        elapsedRealtimeNs.toString(),
        sensorTimestampNs?.toString().orEmpty(),
        sampleIndex.toString(),
        accX.toString(),
        accY.toString(),
        accZ.toString(),
    ).joinToString(",")
}
