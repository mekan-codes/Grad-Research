package com.tinyppgcollector.wear.recording

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class WatchRecordingUiState(
    val readyStatus: String = "Ready for phone commands",
    val lastCommand: String = "none",
    val recording: Boolean = false,
    val ppgAvailable: Boolean = false,
    val imuAvailable: Boolean = false,
    val ppgSamples: Long = 0L,
    val imuSamples: Long = 0L,
    val lastError: String = "",
)

object WatchRecordingStateStore {
    private val _state = MutableStateFlow(WatchRecordingUiState())
    val state: StateFlow<WatchRecordingUiState> = _state.asStateFlow()

    fun update(transform: (WatchRecordingUiState) -> WatchRecordingUiState) {
        _state.value = transform(_state.value)
    }

    fun status(message: String) {
        update { it.copy(readyStatus = message) }
    }

    fun command(command: String) {
        update { it.copy(lastCommand = command, readyStatus = "$command received") }
    }

    fun error(message: String) {
        update { it.copy(lastError = message, readyStatus = message) }
    }
}
