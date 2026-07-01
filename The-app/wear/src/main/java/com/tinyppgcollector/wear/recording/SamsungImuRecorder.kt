package com.tinyppgcollector.wear.recording

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.util.Log
import com.tinyppgcollector.wear.model.WatchImuSample

class SamsungImuRecorder(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(SensorManager::class.java)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private var sessionId: String? = null
    private var sampleIndex = 0L
    private var onSample: ((WatchImuSample) -> Unit)? = null

    val isAvailable: Boolean
        get() = accelerometer != null

    fun start(sessionId: String, onSample: (WatchImuSample) -> Unit): RecorderStartResult {
        val sensor = accelerometer ?: run {
            Log.e(TAG, "Accelerometer unavailable")
            return RecorderStartResult(false, "Accelerometer tracker unavailable")
        }
        this.sessionId = sessionId
        this.sampleIndex = 0L
        this.onSample = onSample

        val registered = sensorManager.registerListener(
            this,
            sensor,
            TARGET_SAMPLING_PERIOD_US,
            TARGET_SAMPLING_PERIOD_US,
        )
        return if (registered) {
            Log.i(TAG, "Accelerometer listener registered session=$sessionId targetUs=$TARGET_SAMPLING_PERIOD_US")
            RecorderStartResult(true, "Watch accelerometer recording at target 25 Hz")
        } else {
            Log.e(TAG, "Accelerometer listener registration failed")
            RecorderStartResult(false, "Accelerometer tracker unavailable")
        }
    }

    fun stop() {
        Log.i(TAG, "Accelerometer stop requested")
        sensorManager.unregisterListener(this)
        onSample = null
        sessionId = null
    }

    override fun onSensorChanged(event: SensorEvent) {
        val activeSession = sessionId ?: return
        val values = event.values
        if (values.size < 3) return
        val index = sampleIndex++
        onSample?.invoke(
            WatchImuSample(
                sessionId = activeSession,
                timestampUnixMs = System.currentTimeMillis(),
                elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
                sensorTimestampNs = event.timestamp,
                sampleIndex = index,
                accX = values[0],
                accY = values[1],
                accZ = values[2],
            ),
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        private const val TAG = "TinyPPGWatch"
        private const val TARGET_SAMPLING_PERIOD_US = 40_000
    }
}
