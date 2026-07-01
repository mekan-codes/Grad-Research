package com.tinyppgcollector.phone.storage

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

data class SessionSampleCounts(
    val watchPpg: Long = 0,
    val watchImu: Long = 0,
    val polarEcg: Long = 0,
    val polarHr: Long = 0,
    val polarAcc: Long = 0,
)

data class PreparedSessionExport(
    val rowCount: Long,
    val note: String,
)

data class SessionMetadata(
    val sessionId: String,
    val subjectId: String,
    val activityLabel: String,
    val notes: String,
    val startTimeUnixMs: Long,
    val stopTimeUnixMs: Long?,
    val phoneModel: String,
    val watchModel: String?,
    val polarDeviceId: String?,
    val appVersion: String,
    val watchPpgTargetHz: Int = 25,
    val watchImuTargetHz: Int = 25,
    val watchPpgHz: Int = 25,
    val watchImuHz: Int = 25,
    val polarEcgHz: Int?,
    val polarHrEnabled: Boolean = true,
    val polarRrEnabled: Boolean = true,
    val samsungSensorSdkAvailable: Boolean,
    val polarSdkAvailable: Boolean,
    val watchStarted: Boolean = false,
    val watchStopped: Boolean = false,
    val polarStarted: Boolean = false,
    val polarStopped: Boolean = false,
    val preparedSessionGenerated: Boolean = false,
    val preparedSessionRowCount: Long = 0,
    val preparedSessionNote: String? = null,
    val sampleCounts: SessionSampleCounts = SessionSampleCounts(),
    val errorsWarnings: List<String> = emptyList(),
)

data class SessionFileValidation(
    val fileName: String,
    val exists: Boolean,
    val hasExpectedHeader: Boolean?,
    val rowCount: Long,
    val message: String,
)

data class SessionValidationResult(
    val files: List<SessionFileValidation>,
) {
    val success: Boolean
        get() = files.all { file ->
            file.exists && (file.hasExpectedHeader != false)
        }

    fun summary(): String {
        return files.joinToString(separator = "\n") { file ->
            val header = when (file.hasExpectedHeader) {
                true -> "header ok"
                false -> "bad header"
                null -> "exists"
            }
            "${file.fileName}: $header, rows=${file.rowCount}"
        }
    }
}

data class SessionWriterResult(
    val sessionId: String,
    val displayPath: String,
)

class CsvSessionWriter(
    private val context: Context,
    private val sessionId: String,
) {
    private val lock = Any()
    private val folderName = sessionId
    private val relativeFolder = "${Environment.DIRECTORY_DOCUMENTS}/TinyPPGCollector/$folderName"
    private val writers = mutableMapOf<String, BufferedWriter>()
    private val outputTargets = mutableMapOf<String, OutputTarget>()
    private var metadataTarget: OutputTarget? = null

    val displayPath: String = "Documents/TinyPPGCollector/$folderName"

    fun open(initialMetadata: SessionMetadata): SessionWriterResult {
        synchronized(lock) {
            CSV_HEADERS.forEach { (fileName, header) ->
                val writer = createWriter(fileName, "text/csv")
                writer.write(header)
                writer.newLine()
                writer.flush()
                writers[fileName] = writer
            }
            metadataTarget = createOutputTarget("metadata.json", "application/json")
            outputTargets["metadata.json"] = metadataTarget ?: error("metadata.json target was not created")
            writeMetadataLocked(initialMetadata)
        }
        return SessionWriterResult(sessionId, displayPath)
    }

    fun appendEvent(
        eventType: String,
        eventLabel: String,
        notes: String,
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
    ) {
        appendCsvRow(
            FILE_EVENTS,
            listOf(
                sessionId,
                timestampUnixMs.toString(),
                elapsedRealtimeNs.toString(),
                eventType,
                eventLabel,
                notes,
            ),
        )
    }

    fun appendWatchPpgBatch(batchRows: String): Int = appendRawRows(FILE_WATCH_PPG, batchRows)

    fun appendWatchImuBatch(batchRows: String): Int = appendRawRows(FILE_WATCH_IMU, batchRows)

    fun appendPolarEcgRow(
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
        sensorTimestampNs: Long?,
        sampleIndex: Long,
        ecgUv: Int,
    ) {
        appendCsvRow(
            FILE_POLAR_ECG,
            listOf(
                sessionId,
                timestampUnixMs.toString(),
                elapsedRealtimeNs.toString(),
                sensorTimestampNs?.toString().orEmpty(),
                sampleIndex.toString(),
                ecgUv.toString(),
            ),
        )
    }

    fun appendPolarHrRow(
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
        sampleIndex: Long,
        hrBpm: Int,
        rrMs: Int?,
    ) {
        appendCsvRow(
            FILE_POLAR_HR,
            listOf(
                sessionId,
                timestampUnixMs.toString(),
                elapsedRealtimeNs.toString(),
                sampleIndex.toString(),
                hrBpm.toString(),
                rrMs?.toString().orEmpty(),
            ),
        )
    }

    fun appendPolarAccRow(
        timestampUnixMs: Long,
        elapsedRealtimeNs: Long,
        sensorTimestampNs: Long?,
        sampleIndex: Long,
        accX: Float,
        accY: Float,
        accZ: Float,
    ) {
        appendCsvRow(
            FILE_POLAR_ACC,
            listOf(
                sessionId,
                timestampUnixMs.toString(),
                elapsedRealtimeNs.toString(),
                sensorTimestampNs?.toString().orEmpty(),
                sampleIndex.toString(),
                accX.toString(),
                accY.toString(),
                accZ.toString(),
            ),
        )
    }

    fun writeMetadata(metadata: SessionMetadata) {
        synchronized(lock) {
            writeMetadataLocked(metadata)
        }
    }

    fun flush() {
        synchronized(lock) {
            writers.values.forEach { it.flush() }
        }
    }

    fun close() {
        synchronized(lock) {
            writers.values.forEach { writer ->
                runCatching {
                    writer.flush()
                    writer.close()
                }
            }
            writers.clear()
        }
    }

    fun validateFiles(): SessionValidationResult {
        synchronized(lock) {
            val csvResults = CSV_HEADERS.map { (fileName, expectedHeader) ->
                validateCsvFile(fileName, expectedHeader)
            }
            val preparedResult = validateCsvFile(FILE_PREPARED_SESSION, PREPARED_SESSION_HEADER)
            val metadataResult = validateMetadataFile()
            return SessionValidationResult(listOf(metadataResult) + csvResults + preparedResult)
        }
    }

    fun writePreparedSession(): PreparedSessionExport {
        synchronized(lock) {
            val ppgRows = readPreparedPpgRows()
            val imuRows = readPreparedImuRows()
            val hrRows = readPreparedHrRows()
            val ecgRows = readPreparedEcgRows()
            val target = createOutputTarget(FILE_PREPARED_SESSION, "text/csv")
            outputTargets[FILE_PREPARED_SESSION] = target

            var rowCount = 0L
            target.openWriter(truncate = true).use { writer ->
                writer.write(PREPARED_SESSION_HEADER)
                writer.newLine()
                if (ppgRows.isNotEmpty()) {
                    val startTimestampMs = ppgRows.first().timestampUnixMs
                    ppgRows.forEach { ppg ->
                        val imu = imuRows.nearestByTimestamp(ppg.timestampUnixMs)
                        val hr = hrRows.nearestByTimestamp(ppg.timestampUnixMs)
                        val ecg = ecgRows.nearestByTimestamp(ppg.timestampUnixMs)
                        writer.write(
                            listOf(
                                formatSeconds(ppg.timestampUnixMs - startTimestampMs),
                                ppg.ppg.orEmpty(),
                                hr?.hr.orEmpty(),
                                imu?.accX.orEmpty(),
                                imu?.accY.orEmpty(),
                                imu?.accZ.orEmpty(),
                                ecg?.ecg.orEmpty(),
                            ).joinToString(",") { escapeCsv(it) },
                        )
                        writer.newLine()
                        rowCount++
                    }
                }
            }

            return PreparedSessionExport(
                rowCount = rowCount,
                note = PREPARED_SESSION_NOTE,
            )
        }
    }

    private fun appendCsvRow(fileName: String, cells: List<String>) {
        synchronized(lock) {
            val writer = writers[fileName] ?: error("$fileName is not open")
            writer.write(cells.joinToString(",") { escapeCsv(it) })
            writer.newLine()
            writer.flush()
        }
    }

    private fun appendRawRows(fileName: String, batchRows: String): Int {
        val cleanRows = batchRows
            .lineSequence()
            .map { it.trimEnd('\r') }
            .filter { it.isNotBlank() }
            .toList()

        if (cleanRows.isEmpty()) return 0

        synchronized(lock) {
            val writer = writers[fileName] ?: error("$fileName is not open")
            cleanRows.forEach { row ->
                writer.write(row)
                writer.newLine()
            }
            writer.flush()
        }
        return cleanRows.size
    }

    private fun writeMetadataLocked(metadata: SessionMetadata) {
        val target = metadataTarget ?: error("metadata.json is not open")
        target.openWriter(truncate = true).use { writer ->
            writer.write(metadata.toJson().toString(2))
            writer.newLine()
        }
    }

    private fun createWriter(fileName: String, mimeType: String): BufferedWriter {
        val target = createOutputTarget(fileName, mimeType)
        outputTargets[fileName] = target
        return target.openWriter(truncate = true)
    }

    private fun createOutputTarget(fileName: String, mimeType: String): OutputTarget {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, relativeFolder)
            }
            val uri = context.contentResolver.insert(
                MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
                values,
            ) ?: error("Could not create $fileName in $displayPath")
            OutputTarget.MediaStoreTarget(context, uri)
        } else {
            val folder = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                "TinyPPGCollector/$folderName",
            )
            if (!folder.exists() && !folder.mkdirs()) {
                error("Could not create ${folder.absolutePath}")
            }
            OutputTarget.FileTarget(File(folder, fileName))
        }
    }

    private fun SessionMetadata.toJson(): JSONObject {
        return JSONObject()
            .put("session_id", sessionId)
            .put("subject_id", subjectId)
            .put("activity_label", activityLabel)
            .put("notes", notes)
            .put("start_time_unix_ms", startTimeUnixMs)
            .put("stop_time_unix_ms", stopTimeUnixMs ?: JSONObject.NULL)
            .put("phone_model", phoneModel)
            .put("watch_model", watchModel ?: JSONObject.NULL)
            .put("polar_device_id", polarDeviceId ?: JSONObject.NULL)
            .put("app_version", appVersion)
            .put("watch_ppg_target_hz", watchPpgTargetHz)
            .put("watch_imu_target_hz", watchImuTargetHz)
            .put("watch_ppg_hz", watchPpgHz)
            .put("watch_imu_hz", watchImuHz)
            .put("polar_ecg_hz", polarEcgHz ?: JSONObject.NULL)
            .put("polar_hr_enabled", polarHrEnabled)
            .put("polar_rr_enabled", polarRrEnabled)
            .put("samsung_sensor_sdk_available", samsungSensorSdkAvailable)
            .put("polar_sdk_available", polarSdkAvailable)
            .put("watch_started", watchStarted)
            .put("watch_stopped", watchStopped)
            .put("polar_started", polarStarted)
            .put("polar_stopped", polarStopped)
            .put("prepared_session_file", FILE_PREPARED_SESSION)
            .put("prepared_session_generated", preparedSessionGenerated)
            .put("prepared_session_row_count", preparedSessionRowCount)
            .put("prepared_session_note", preparedSessionNote ?: JSONObject.NULL)
            .put("stream_started_flags", JSONObject()
                .put("watch", watchStarted)
                .put("polar", polarStarted))
            .put("stream_stopped_flags", JSONObject()
                .put("watch", watchStopped)
                .put("polar", polarStopped))
            .put(
                "sample_counts",
                JSONObject()
                    .put("watch_ppg", sampleCounts.watchPpg)
                    .put("watch_imu", sampleCounts.watchImu)
                    .put("polar_ecg", sampleCounts.polarEcg)
                    .put("polar_hr", sampleCounts.polarHr)
                    .put("polar_acc", sampleCounts.polarAcc),
            )
            .put("errors_warnings", JSONArray(errorsWarnings))
    }

    private fun validateMetadataFile(): SessionFileValidation {
        val fileName = "metadata.json"
        val target = outputTargets[fileName]
        val lines = target?.readLines().orEmpty()
        val exists = target != null && lines.isNotEmpty()
        return SessionFileValidation(
            fileName = fileName,
            exists = exists,
            hasExpectedHeader = null,
            rowCount = if (exists) 1 else 0,
            message = if (exists) "metadata.json exists" else "metadata.json missing",
        )
    }

    private fun validateCsvFile(fileName: String, expectedHeader: String): SessionFileValidation {
        val target = outputTargets[fileName]
        val lines = target?.readLines().orEmpty()
        val exists = target != null && lines.isNotEmpty()
        val header = lines.firstOrNull()
        val hasExpectedHeader = exists && header == expectedHeader
        val rowCount = if (exists) {
            lines.drop(1).count { it.isNotBlank() }.toLong()
        } else {
            0L
        }
        return SessionFileValidation(
            fileName = fileName,
            exists = exists,
            hasExpectedHeader = hasExpectedHeader,
            rowCount = rowCount,
            message = when {
                !exists -> "$fileName missing"
                !hasExpectedHeader -> "$fileName header mismatch"
                else -> "$fileName validated"
            },
        )
    }

    private fun readPreparedPpgRows(): List<PreparedPpgRow> {
        return readDataLines(FILE_WATCH_PPG).mapNotNull { cells ->
            val timestamp = cells.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null
            PreparedPpgRow(
                timestampUnixMs = timestamp,
                ppg = cells.getOrNull(5).orEmpty(),
            )
        }.filter { it.ppg.isNotBlank() }
            .sortedBy { it.timestampUnixMs }
    }

    private fun readPreparedImuRows(): List<PreparedImuRow> {
        return readDataLines(FILE_WATCH_IMU).mapNotNull { cells ->
            val timestamp = cells.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null
            PreparedImuRow(
                timestampUnixMs = timestamp,
                accX = cells.getOrNull(5).orEmpty(),
                accY = cells.getOrNull(6).orEmpty(),
                accZ = cells.getOrNull(7).orEmpty(),
            )
        }.sortedBy { it.timestampUnixMs }
    }

    private fun readPreparedHrRows(): List<PreparedHrRow> {
        return readDataLines(FILE_POLAR_HR).mapNotNull { cells ->
            val timestamp = cells.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null
            PreparedHrRow(
                timestampUnixMs = timestamp,
                hr = cells.getOrNull(4).orEmpty(),
            )
        }.filter { it.hr.isNotBlank() }
            .sortedBy { it.timestampUnixMs }
    }

    private fun readPreparedEcgRows(): List<PreparedEcgRow> {
        return readDataLines(FILE_POLAR_ECG).mapNotNull { cells ->
            val timestamp = cells.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null
            PreparedEcgRow(
                timestampUnixMs = timestamp,
                ecg = cells.getOrNull(5).orEmpty(),
            )
        }.filter { it.ecg.isNotBlank() }
            .sortedBy { it.timestampUnixMs }
    }

    private fun readDataLines(fileName: String): List<List<String>> {
        return outputTargets[fileName]
            ?.readLines()
            .orEmpty()
            .drop(1)
            .filter { it.isNotBlank() }
            .map { line -> line.split(",") }
    }

    private fun <T : TimestampedPreparedRow> List<T>.nearestByTimestamp(timestampUnixMs: Long): T? {
        if (isEmpty()) return null
        val insertionPoint = binarySearchBy(timestampUnixMs) { it.timestampUnixMs }
        if (insertionPoint >= 0) return this[insertionPoint]

        val nextIndex = -insertionPoint - 1
        val previous = getOrNull(nextIndex - 1)
        val next = getOrNull(nextIndex)
        return when {
            previous == null -> next
            next == null -> previous
            timestampUnixMs - previous.timestampUnixMs <= next.timestampUnixMs - timestampUnixMs -> previous
            else -> next
        }
    }

    private fun formatSeconds(deltaMs: Long): String {
        return String.format(Locale.US, "%.3f", deltaMs / 1000.0)
    }

    private fun escapeCsv(value: String): String {
        val needsQuotes = value.any { it == ',' || it == '"' || it == '\n' || it == '\r' }
        if (!needsQuotes) return value
        return "\"" + value.replace("\"", "\"\"") + "\""
    }

    private sealed interface OutputTarget {
        fun openWriter(truncate: Boolean): BufferedWriter
        fun readLines(): List<String>

        data class MediaStoreTarget(
            val context: Context,
            val uri: Uri,
        ) : OutputTarget {
            override fun openWriter(truncate: Boolean): BufferedWriter {
                val mode = if (truncate) "wt" else "wa"
                val stream = context.contentResolver.openOutputStream(uri, mode)
                    ?: error("Could not open $uri")
                return BufferedWriter(OutputStreamWriter(stream, StandardCharsets.UTF_8))
            }

            override fun readLines(): List<String> {
                val stream = context.contentResolver.openInputStream(uri) ?: return emptyList()
                return stream.bufferedReader(StandardCharsets.UTF_8).use { it.readLines() }
            }
        }

        data class FileTarget(val file: File) : OutputTarget {
            override fun openWriter(truncate: Boolean): BufferedWriter {
                val stream = FileOutputStream(file, !truncate)
                return BufferedWriter(OutputStreamWriter(stream, StandardCharsets.UTF_8))
            }

            override fun readLines(): List<String> {
                if (!file.exists()) return emptyList()
                return file.bufferedReader(StandardCharsets.UTF_8).use { it.readLines() }
            }
        }
    }

    companion object {
        const val FILE_WATCH_PPG = "watch_ppg.csv"
        const val FILE_WATCH_IMU = "watch_imu.csv"
        const val FILE_POLAR_ECG = "polar_ecg.csv"
        const val FILE_POLAR_HR = "polar_hr.csv"
        const val FILE_POLAR_ACC = "polar_acc.csv"
        const val FILE_EVENTS = "events.csv"
        const val FILE_PREPARED_SESSION = "prepared_session.csv"
        const val PREPARED_SESSION_HEADER = "time,ppg,hr,acc_x,acc_y,acc_z,ecg"
        private const val PREPARED_SESSION_NOTE =
            "prepared_session.csv is derived from raw CSV files using approximate nearest timestamp alignment for quick ML ingestion. Raw CSV files remain authoritative."

        private val CSV_HEADERS = linkedMapOf(
            FILE_WATCH_PPG to "session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,ppg_green,ppg_red,ppg_ir,status",
            FILE_WATCH_IMU to "session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,acc_x,acc_y,acc_z",
            FILE_POLAR_ECG to "session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,ecg_uv",
            FILE_POLAR_HR to "session_id,timestamp_unix_ms,elapsed_realtime_ns,sample_index,hr_bpm,rr_ms",
            FILE_POLAR_ACC to "session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,acc_x,acc_y,acc_z",
            FILE_EVENTS to "session_id,timestamp_unix_ms,elapsed_realtime_ns,event_type,event_label,notes",
        )
    }
}

private sealed interface TimestampedPreparedRow {
    val timestampUnixMs: Long
}

private data class PreparedPpgRow(
    override val timestampUnixMs: Long,
    val ppg: String,
) : TimestampedPreparedRow

private data class PreparedImuRow(
    override val timestampUnixMs: Long,
    val accX: String,
    val accY: String,
    val accZ: String,
) : TimestampedPreparedRow

private data class PreparedHrRow(
    override val timestampUnixMs: Long,
    val hr: String,
) : TimestampedPreparedRow

private data class PreparedEcgRow(
    override val timestampUnixMs: Long,
    val ecg: String,
) : TimestampedPreparedRow
