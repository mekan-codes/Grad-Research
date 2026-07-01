package com.tinyppgcollector.phone

import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tinyppgcollector.phone.model.ActivityLabel
import com.tinyppgcollector.phone.model.ConnectionState
import com.tinyppgcollector.phone.model.RecordingUiState
import com.tinyppgcollector.phone.model.SessionConfig
import com.tinyppgcollector.phone.model.StreamCounts
import com.tinyppgcollector.phone.permissions.PermissionManager
import com.tinyppgcollector.phone.polar.PolarH10Recorder
import com.tinyppgcollector.phone.polar.PolarSampleEvent
import com.tinyppgcollector.phone.recording.RecordingForegroundService
import com.tinyppgcollector.phone.storage.SessionSampleCounts
import com.tinyppgcollector.phone.storage.SessionManager
import com.tinyppgcollector.phone.wear.WatchConnectionStatus
import com.tinyppgcollector.phone.wear.WatchIncoming
import com.tinyppgcollector.phone.wear.WearCommandClient
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

class RecordingViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager(application.applicationContext)
    private val wearCommandClient = WearCommandClient(application.applicationContext)
    private val polarRecorder = PolarH10Recorder(application.applicationContext)

    private val _uiState = MutableStateFlow(RecordingUiState())
    val uiState: StateFlow<RecordingUiState> = _uiState.asStateFlow()

    private var elapsedJob: Job? = null
    private val processedWatchDoneCounter = MutableStateFlow(0L)

    init {
        wearCommandClient.status
            .onEach(::handleWatchStatus)
            .launchIn(viewModelScope)

        wearCommandClient.incoming
            .onEach(::handleWatchIncoming)
            .launchIn(viewModelScope)

        polarRecorder.status
            .onEach { status ->
                _uiState.value = _uiState.value.copy(
                    polarConnectionState = status.state,
                    polarStatusText = status.message,
                )
            }
            .launchIn(viewModelScope)

        polarRecorder.errors
            .onEach { message ->
                if (!message.startsWith("Polar ACC stream unavailable")) {
                    _uiState.value = _uiState.value.copy(
                        polarConnectionState = ConnectionState.Error,
                        polarStatusText = message,
                    )
                }
                showError(message)
            }
            .launchIn(viewModelScope)

        polarRecorder.samples
            .onEach(::handlePolarSample)
            .launchIn(viewModelScope)
    }

    fun updateSubjectId(value: String) {
        _uiState.value = _uiState.value.copy(subjectId = value)
    }

    fun updateNotes(value: String) {
        _uiState.value = _uiState.value.copy(notes = value)
    }

    fun updateActivityLabel(value: ActivityLabel) {
        _uiState.value = _uiState.value.copy(activityLabel = value)
    }

    fun updateCustomActivityLabel(value: String) {
        _uiState.value = _uiState.value.copy(customActivityLabel = value)
    }

    fun connectWatch() {
        viewModelScope.launch {
            val connected = wearCommandClient.pingWatch()
            if (!connected) showError(wearCommandClient.status.value.message)
        }
    }

    fun connectPolar() {
        viewModelScope.launch {
            polarRecorder.connect()
        }
    }

    fun onPermissionsDenied() {
        showError("Required sensor, Bluetooth, or notification permission denied")
    }

    fun startRecording() {
        val state = _uiState.value
        if (state.isRecording) return
        if (state.subjectId.isBlank()) {
            showError("Subject ID is required")
            return
        }
        if (!PermissionManager.hasRequiredPermissions(getApplication())) {
            showError("Sensor and Bluetooth permissions are required before recording")
            return
        }

        viewModelScope.launch {
            runCatching {
                val config = SessionConfig(
                    subjectId = state.subjectId.trim(),
                    activityLabel = state.resolvedActivityLabel,
                    notes = state.notes.trim(),
                )
                val phoneStartElapsedNs = SystemClock.elapsedRealtimeNanos()
                val activeSession = sessionManager.startSession(config)
                RecordingForegroundService.start(getApplication(), activeSession.sessionId)

                _uiState.value = _uiState.value.copy(
                    isRecording = true,
                    sessionId = activeSession.sessionId,
                    savedFolderPath = activeSession.displayPath,
                    elapsedSeconds = 0,
                    counts = StreamCounts(),
                    fileValidationSummary = null,
                    lastError = null,
                )

                sessionManager.recordEvent("START", config.activityLabel, config.notes)
                sessionManager.updatePolarMetadata(
                    deviceId = polarRecorder.connectedDeviceId,
                    sdkAvailable = polarRecorder.polarSdkAvailable,
                    ecgHz = polarRecorder.polarEcgHz,
                )
                val polarStarted = polarRecorder.startRecording(activeSession.sessionId)
                sessionManager.updatePolarMetadata(
                    deviceId = polarRecorder.connectedDeviceId,
                    sdkAvailable = polarRecorder.polarSdkAvailable,
                    ecgHz = polarRecorder.polarEcgHz,
                    started = polarStarted,
                )
                val watchStarted = wearCommandClient.sendStartSession(
                    sessionId = activeSession.sessionId,
                    subjectId = config.subjectId,
                    activityLabel = config.activityLabel,
                    phoneStartElapsedRealtimeNs = phoneStartElapsedNs,
                )
                if (!watchStarted) {
                    val message = "Watch not connected; session folder is still open on phone"
                    sessionManager.recordWarning(message)
                    showError(message)
                }
                startElapsedTicker(phoneStartElapsedNs)
            }.onFailure { throwable ->
                sessionManager.closeAfterFailure()
                RecordingForegroundService.stop(getApplication())
                showError(throwable.message ?: "Could not start recording")
            }
        }
    }

    fun stopRecording() {
        val sessionId = _uiState.value.sessionId ?: return
        if (!_uiState.value.isRecording) return

        viewModelScope.launch {
            runCatching {
                sessionManager.recordEvent(
                    "STOP_REQUESTED",
                    _uiState.value.resolvedActivityLabel,
                    _uiState.value.notes,
                )
                val watchDoneSnapshot = processedWatchDoneCounter.value
                val stopSentToWatch = wearCommandClient.sendStopSession(sessionId)
                if (!stopSentToWatch) {
                    sessionManager.recordWarning("Watch disconnected during STOP_SESSION")
                }

                polarRecorder.stopRecording()
                sessionManager.updatePolarMetadata(
                    deviceId = polarRecorder.connectedDeviceId,
                    sdkAvailable = polarRecorder.polarSdkAvailable,
                    ecgHz = polarRecorder.polarEcgHz,
                    stopped = true,
                )

                val watchDone = if (stopSentToWatch) {
                    waitForProcessedWatchDone(watchDoneSnapshot)
                } else {
                    false
                }
                if (watchDone) {
                    sessionManager.updateWatchMetadata(stopped = true)
                } else {
                    sessionManager.recordWarning("Timed out waiting for WATCH_DONE")
                }

                sessionManager.recordEvent(
                    "STOP_COMPLETE",
                    _uiState.value.resolvedActivityLabel,
                    "watch_done=$watchDone",
                )
                val stoppedSession = sessionManager.stopSession()
                elapsedJob?.cancel()
                RecordingForegroundService.stop(getApplication())

                _uiState.value = _uiState.value.copy(
                    isRecording = false,
                    savedFolderPath = stoppedSession?.displayPath ?: _uiState.value.savedFolderPath,
                    counts = stoppedSession?.counts?.toStreamCounts() ?: _uiState.value.counts,
                    fileValidationSummary = stoppedSession?.validation?.summary(),
                    watchStatusText = "Watch stopped or disconnected",
                    elapsedSeconds = _uiState.value.elapsedSeconds,
                )
            }.onFailure { throwable ->
                showError(throwable.message ?: "Could not stop recording cleanly")
            }
        }
    }

    private suspend fun handlePolarSample(event: PolarSampleEvent) {
        when (event) {
            is PolarSampleEvent.Ecg -> {
                val added = sessionManager.appendPolarEcgRow(
                    timestampUnixMs = event.timestampUnixMs,
                    elapsedRealtimeNs = event.elapsedRealtimeNs,
                    sensorTimestampNs = event.sensorTimestampNs,
                    sampleIndex = event.sampleIndex,
                    ecgUv = event.ecgUv,
                )
                if (added > 0) incrementCounts { it.copy(polarEcg = it.polarEcg + added) }
            }

            is PolarSampleEvent.Hr -> {
                val added = sessionManager.appendPolarHrRow(
                    timestampUnixMs = event.timestampUnixMs,
                    elapsedRealtimeNs = event.elapsedRealtimeNs,
                    sampleIndex = event.sampleIndex,
                    hrBpm = event.hrBpm,
                    rrMs = event.rrMs,
                )
                if (added > 0) incrementCounts { it.copy(polarHr = it.polarHr + added) }
            }

            is PolarSampleEvent.Acc -> {
                val added = sessionManager.appendPolarAccRow(
                    timestampUnixMs = event.timestampUnixMs,
                    elapsedRealtimeNs = event.elapsedRealtimeNs,
                    sensorTimestampNs = event.sensorTimestampNs,
                    sampleIndex = event.sampleIndex,
                    accX = event.accX,
                    accY = event.accY,
                    accZ = event.accZ,
                )
                if (added > 0) incrementCounts { it.copy(polarAcc = it.polarAcc + added) }
            }
        }
    }

    private suspend fun handleWatchIncoming(incoming: WatchIncoming) {
        when (incoming) {
            is WatchIncoming.PpgBatch -> {
                val added = sessionManager.appendWatchPpgBatch(incoming.rows)
                Log.i(TAG, "WATCH_PPG_BATCH appended rows=$added")
                incrementCounts { it.copy(watchPpg = it.watchPpg + added) }
                if (added > 0) {
                    _uiState.value = _uiState.value.copy(
                        watchStatusText = "WATCH_PPG_BATCH received rows=$added",
                    )
                }
            }

            is WatchIncoming.ImuBatch -> {
                val added = sessionManager.appendWatchImuBatch(incoming.rows)
                Log.i(TAG, "WATCH_IMU_BATCH appended rows=$added")
                incrementCounts { it.copy(watchImu = it.watchImu + added) }
                if (added > 0) {
                    _uiState.value = _uiState.value.copy(
                        watchStatusText = "WATCH_IMU_BATCH received rows=$added",
                    )
                }
            }

            is WatchIncoming.Error -> {
                Log.e(TAG, "WATCH_ERROR received: ${incoming.message}")
                _uiState.value = _uiState.value.copy(
                    watchConnectionState = ConnectionState.Error,
                    watchStatusText = incoming.message,
                )
                showError(incoming.message)
            }

            is WatchIncoming.Status -> {
                val status = incoming.status
                Log.i(
                    TAG,
                    "WATCH_STATUS message=${status.message} recording=${status.recording} " +
                        "ppgAvailable=${status.ppgAvailable} imuAvailable=${status.imuAvailable} " +
                        "ppgSamples=${status.ppgSamples} imuSamples=${status.imuSamples}",
                )
                status.watchModel?.let { model ->
                    sessionManager.updateWatchMetadata(
                        watchModel = model,
                        samsungSdkAvailable = status.samsungSdkAvailable ?: status.ppgAvailable,
                    )
                }
            }

            is WatchIncoming.Started -> {
                Log.i(TAG, "WATCH_STARTED received: ${incoming.status.message}")
                _uiState.value = _uiState.value.copy(
                    watchConnectionState = ConnectionState.Connected,
                    watchStatusText = incoming.status.message.ifBlank { "Watch started" },
                )
                sessionManager.updateWatchMetadata(
                    watchModel = incoming.status.watchModel,
                    samsungSdkAvailable = incoming.status.samsungSdkAvailable ?: incoming.status.ppgAvailable,
                    started = true,
                )
                sessionManager.recordEvent("WATCH_STARTED", "watch", incoming.status.message)
            }

            WatchIncoming.Done -> {
                Log.i(TAG, "WATCH_DONE received")
                sessionManager.updateWatchMetadata(stopped = true)
                sessionManager.recordEvent("WATCH_DONE", "watch", "Watch flushed remaining samples")
                processedWatchDoneCounter.value = processedWatchDoneCounter.value + 1
            }
        }
    }

    private fun handleWatchStatus(status: WatchConnectionStatus) {
        _uiState.value = _uiState.value.copy(
            watchConnectionState = if (status.connected) ConnectionState.Connected else ConnectionState.Disconnected,
            watchStatusText = status.message,
        )
    }

    private fun incrementCounts(update: (StreamCounts) -> StreamCounts) {
        _uiState.value = _uiState.value.copy(counts = update(_uiState.value.counts))
    }

    private fun startElapsedTicker(startElapsedNs: Long) {
        elapsedJob?.cancel()
        elapsedJob = viewModelScope.launch {
            while (isActive) {
                val elapsedNs = SystemClock.elapsedRealtimeNanos() - startElapsedNs
                _uiState.value = _uiState.value.copy(elapsedSeconds = elapsedNs / 1_000_000_000L)
                delay(500)
            }
        }
    }

    private fun showError(message: String) {
        _uiState.value = _uiState.value.copy(lastError = message)
        if (_uiState.value.sessionId != null) {
            viewModelScope.launch {
                sessionManager.recordWarning(message)
            }
        }
    }

    private suspend fun waitForProcessedWatchDone(afterSnapshot: Long): Boolean {
        return withTimeoutOrNull(WATCH_DONE_TIMEOUT_MS) {
            processedWatchDoneCounter.filter { it > afterSnapshot }.first()
            true
        } ?: false
    }

    private fun SessionSampleCounts.toStreamCounts(): StreamCounts {
        return StreamCounts(
            watchPpg = watchPpg,
            watchImu = watchImu,
            polarEcg = polarEcg,
            polarHr = polarHr,
            polarAcc = polarAcc,
        )
    }

    override fun onCleared() {
        wearCommandClient.close()
        polarRecorder.close()
        super.onCleared()
    }

    companion object {
        private const val TAG = "TinyPPGPhone"
        private const val WATCH_DONE_TIMEOUT_MS = 10_000L
    }
}
