package com.tinyppgcollector.phone.storage

import android.content.Context
import android.os.Build
import android.os.SystemClock
import com.tinyppgcollector.phone.model.SessionConfig
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class ActiveSession(
    val sessionId: String,
    val displayPath: String,
)

data class StoppedSession(
    val sessionId: String,
    val displayPath: String,
    val validation: SessionValidationResult,
    val counts: SessionSampleCounts,
)

class SessionManager(private val context: Context) {
    private var writer: CsvSessionWriter? = null
    private var metadata: SessionMetadata? = null
    private var sampleCounts = SessionSampleCounts()
    private var warnings = emptyList<String>()
    private val watchPpgCount = AtomicLong(0)
    private val watchImuCount = AtomicLong(0)
    private val polarEcgCount = AtomicLong(0)
    private val polarHrCount = AtomicLong(0)
    private val polarAccCount = AtomicLong(0)

    suspend fun startSession(config: SessionConfig): ActiveSession = withContext(Dispatchers.IO) {
        check(writer == null) { "A session is already active" }

        val startUnixMs = System.currentTimeMillis()
        val sessionId = buildSessionId(startUnixMs, config.subjectId)
        resetCounts()
        warnings = emptyList()
        val initialMetadata = SessionMetadata(
            sessionId = sessionId,
            subjectId = config.subjectId,
            activityLabel = config.activityLabel,
            notes = config.notes,
            startTimeUnixMs = startUnixMs,
            stopTimeUnixMs = null,
            phoneModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            watchModel = null,
            polarDeviceId = null,
            appVersion = appVersion(),
            polarEcgHz = null,
            samsungSensorSdkAvailable = false,
            polarSdkAvailable = false,
            sampleCounts = sampleCounts,
            errorsWarnings = warnings,
        )

        val sessionWriter = CsvSessionWriter(context, sessionId)
        val result = sessionWriter.open(initialMetadata)
        sessionWriter.appendEvent(
            eventType = "SESSION_CREATED",
            eventLabel = config.activityLabel,
            notes = config.notes,
            timestampUnixMs = startUnixMs,
            elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
        )

        writer = sessionWriter
        metadata = initialMetadata
        ActiveSession(result.sessionId, result.displayPath)
    }

    suspend fun recordEvent(eventType: String, eventLabel: String, notes: String = "") {
        withContext(Dispatchers.IO) {
            writer?.appendEvent(
                eventType = eventType,
                eventLabel = eventLabel,
                notes = notes,
                timestampUnixMs = System.currentTimeMillis(),
                elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
            )
        }
    }

    suspend fun appendWatchPpgBatch(batchRows: String): Int = withContext(Dispatchers.IO) {
        val added = writer?.appendWatchPpgBatch(batchRows) ?: 0
        if (added > 0) {
            watchPpgCount.addAndGet(added.toLong())
            sampleCounts = currentCounts()
        }
        added
    }

    suspend fun appendWatchImuBatch(batchRows: String): Int = withContext(Dispatchers.IO) {
        val added = writer?.appendWatchImuBatch(batchRows) ?: 0
        if (added > 0) {
            watchImuCount.addAndGet(added.toLong())
            sampleCounts = currentCounts()
        }
        added
    }

    suspend fun updatePolarMetadata(
        deviceId: String?,
        sdkAvailable: Boolean,
        ecgHz: Int? = null,
        started: Boolean? = null,
        stopped: Boolean? = null,
    ) = withContext(Dispatchers.IO) {
        val activeWriter = writer ?: return@withContext
        val activeMetadata = metadata ?: return@withContext
        val updatedMetadata = activeMetadata.copy(
            polarDeviceId = deviceId ?: activeMetadata.polarDeviceId,
            polarSdkAvailable = sdkAvailable,
            polarEcgHz = ecgHz ?: activeMetadata.polarEcgHz,
            polarStarted = started ?: activeMetadata.polarStarted,
            polarStopped = stopped ?: activeMetadata.polarStopped,
            sampleCounts = currentCounts(),
            errorsWarnings = warnings,
        )
        metadata = updatedMetadata
        activeWriter.writeMetadata(updatedMetadata)
    }

    suspend fun updateWatchMetadata(
        watchModel: String? = null,
        samsungSdkAvailable: Boolean? = null,
        started: Boolean? = null,
        stopped: Boolean? = null,
    ) = withContext(Dispatchers.IO) {
        val activeWriter = writer ?: return@withContext
        val activeMetadata = metadata ?: return@withContext
        val updatedMetadata = activeMetadata.copy(
            watchModel = watchModel ?: activeMetadata.watchModel,
            samsungSensorSdkAvailable = samsungSdkAvailable ?: activeMetadata.samsungSensorSdkAvailable,
            watchStarted = started ?: activeMetadata.watchStarted,
            watchStopped = stopped ?: activeMetadata.watchStopped,
            sampleCounts = currentCounts(),
            errorsWarnings = warnings,
        )
        metadata = updatedMetadata
        activeWriter.writeMetadata(updatedMetadata)
    }

    suspend fun recordWarning(message: String) = withContext(Dispatchers.IO) {
        if (message.isBlank()) return@withContext
        if (!warnings.contains(message)) {
            warnings = warnings + message
        }
        val activeWriter = writer ?: return@withContext
        val activeMetadata = metadata ?: return@withContext
        val updatedMetadata = activeMetadata.copy(
            sampleCounts = currentCounts(),
            errorsWarnings = warnings,
        )
        metadata = updatedMetadata
        activeWriter.writeMetadata(updatedMetadata)
    }

    suspend fun appendPolarEcgRow(
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
        sensorTimestampNs: Long?,
        sampleIndex: Long,
        ecgUv: Int,
    ): Int = withContext(Dispatchers.IO) {
        val activeWriter = writer ?: return@withContext 0
        activeWriter.appendPolarEcgRow(
            timestampUnixMs = timestampUnixMs,
            elapsedRealtimeNs = elapsedRealtimeNs,
            sensorTimestampNs = sensorTimestampNs,
            sampleIndex = sampleIndex,
            ecgUv = ecgUv,
        )
        polarEcgCount.incrementAndGet()
        sampleCounts = currentCounts()
        1
    }

    suspend fun appendPolarHrRow(
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
        sampleIndex: Long,
        hrBpm: Int,
        rrMs: Int?,
    ): Int = withContext(Dispatchers.IO) {
        val activeWriter = writer ?: return@withContext 0
        activeWriter.appendPolarHrRow(
            timestampUnixMs = timestampUnixMs,
            elapsedRealtimeNs = elapsedRealtimeNs,
            sampleIndex = sampleIndex,
            hrBpm = hrBpm,
            rrMs = rrMs,
        )
        polarHrCount.incrementAndGet()
        sampleCounts = currentCounts()
        1
    }

    suspend fun appendPolarAccRow(
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
        sensorTimestampNs: Long?,
        sampleIndex: Long,
        accX: Float,
        accY: Float,
        accZ: Float,
    ): Int = withContext(Dispatchers.IO) {
        val activeWriter = writer ?: return@withContext 0
        activeWriter.appendPolarAccRow(
            timestampUnixMs = timestampUnixMs,
            elapsedRealtimeNs = elapsedRealtimeNs,
            sensorTimestampNs = sensorTimestampNs,
            sampleIndex = sampleIndex,
            accX = accX,
            accY = accY,
            accZ = accZ,
        )
        polarAccCount.incrementAndGet()
        sampleCounts = currentCounts()
        1
    }

    suspend fun stopSession(): StoppedSession? = withContext(Dispatchers.IO) {
        val activeWriter = writer ?: return@withContext null
        val activeMetadata = metadata ?: return@withContext null
        val stoppedMetadata = activeMetadata.copy(
            stopTimeUnixMs = System.currentTimeMillis(),
            sampleCounts = currentCounts(),
            errorsWarnings = warnings,
        )
        activeWriter.writeMetadata(stoppedMetadata)
        activeWriter.flush()
        activeWriter.close()
        val preparedExport = activeWriter.writePreparedSession()
        val finalWarnings = if (preparedExport.rowCount == 0L) {
            (warnings + "prepared_session.csv has zero rows because watch_ppg.csv has no green PPG rows").distinct()
        } else {
            warnings
        }
        val finalMetadata = stoppedMetadata.copy(
            preparedSessionGenerated = true,
            preparedSessionRowCount = preparedExport.rowCount,
            preparedSessionNote = preparedExport.note,
            errorsWarnings = finalWarnings,
        )
        activeWriter.writeMetadata(finalMetadata)
        val validation = activeWriter.validateFiles()
        writer = null
        metadata = null
        val finalCounts = currentCounts()
        resetCounts()
        warnings = emptyList()
        StoppedSession(
            sessionId = stoppedMetadata.sessionId,
            displayPath = activeWriter.displayPath,
            validation = validation,
            counts = finalCounts,
        )
    }

    suspend fun closeAfterFailure() = withContext(Dispatchers.IO) {
        writer?.close()
        writer = null
        metadata = null
        resetCounts()
        warnings = emptyList()
    }

    private fun resetCounts() {
        watchPpgCount.set(0)
        watchImuCount.set(0)
        polarEcgCount.set(0)
        polarHrCount.set(0)
        polarAccCount.set(0)
        sampleCounts = SessionSampleCounts()
    }

    private fun currentCounts(): SessionSampleCounts {
        return SessionSampleCounts(
            watchPpg = watchPpgCount.get(),
            watchImu = watchImuCount.get(),
            polarEcg = polarEcgCount.get(),
            polarHr = polarHrCount.get(),
            polarAcc = polarAccCount.get(),
        )
    }

    private fun buildSessionId(startUnixMs: Long, subjectId: String): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(startUnixMs))
        val cleanSubject = subjectId.ifBlank { "subject" }
            .replace(Regex("[^A-Za-z0-9_-]"), "_")
            .take(48)
        return "session_${timestamp}_$cleanSubject"
    }

    private fun appVersion(): String {
        return runCatching {
            val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
            packageInfo.versionName ?: "unknown"
        }.getOrDefault("unknown")
    }
}
