package com.tinyppgcollector.phone.model

enum class ActivityLabel(val rawValue: String, val displayName: String) {
    Rest("rest", "Rest"),
    Walking("walking", "Walking"),
    Running("running", "Running"),
    MotionArtifact("motion_artifact", "Motion artifact"),
    Recovery("recovery", "Recovery"),
    Custom("custom", "Custom"),
}

enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Unavailable,
    Error,
}

data class StreamCounts(
    val watchPpg: Long = 0,
    val watchImu: Long = 0,
    val polarEcg: Long = 0,
    val polarHr: Long = 0,
    val polarAcc: Long = 0,
)

data class RecordingUiState(
    val subjectId: String = "",
    val notes: String = "",
    val activityLabel: ActivityLabel = ActivityLabel.Rest,
    val customActivityLabel: String = "",
    val isRecording: Boolean = false,
    val sessionId: String? = null,
    val savedFolderPath: String? = null,
    val elapsedSeconds: Long = 0,
    val watchConnectionState: ConnectionState = ConnectionState.Disconnected,
    val watchStatusText: String = "Watch not connected",
    val polarConnectionState: ConnectionState = ConnectionState.Disconnected,
    val polarStatusText: String = "Polar H10 not connected",
    val counts: StreamCounts = StreamCounts(),
    val fileValidationSummary: String? = null,
    val lastError: String? = null,
) {
    val resolvedActivityLabel: String
        get() = if (activityLabel == ActivityLabel.Custom) {
            customActivityLabel.ifBlank { ActivityLabel.Custom.rawValue }
        } else {
            activityLabel.rawValue
        }
}

data class SessionConfig(
    val subjectId: String,
    val activityLabel: String,
    val notes: String,
)
