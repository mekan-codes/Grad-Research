package com.tinyppgcollector.wear.recording

import com.tinyppgcollector.wear.comm.PhoneMessageClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class WatchSampleBuffer(
    private val scope: CoroutineScope,
    private val phoneMessageClient: PhoneMessageClient,
    private val path: String,
    private val maxRows: Int = 25,
    private val flushEveryMs: Long = 1_000L,
    private val onBatchSent: suspend (path: String, rows: Int, delivered: Boolean) -> Unit = { _, _, _ -> },
) {
    private val lock = Any()
    private val rows = mutableListOf<String>()
    private var tickerJob: Job? = null

    fun start() {
        tickerJob?.cancel()
        tickerJob = scope.launch {
            while (isActive) {
                delay(flushEveryMs)
                flush()
            }
        }
    }

    fun add(row: String) {
        val batch = synchronized(lock) {
            rows += row
            if (rows.size >= maxRows) drainLocked() else null
        }
        if (batch != null) {
            scope.launch { sendRows(batch) }
        }
    }

    suspend fun flush() {
        val batch = synchronized(lock) { drainLocked() }
        if (batch != null) sendRows(batch)
    }

    suspend fun stopAndFlush() {
        tickerJob?.cancel()
        tickerJob = null
        flush()
    }

    private suspend fun sendRows(batch: List<String>) {
        if (batch.isEmpty()) return
        val payload = batch.joinToString(separator = "\n", postfix = "\n").toByteArray()
        val delivered = phoneMessageClient.send(path, payload)
        onBatchSent(path, batch.size, delivered)
    }

    private fun drainLocked(): List<String>? {
        if (rows.isEmpty()) return null
        val copy = rows.toList()
        rows.clear()
        return copy
    }
}
