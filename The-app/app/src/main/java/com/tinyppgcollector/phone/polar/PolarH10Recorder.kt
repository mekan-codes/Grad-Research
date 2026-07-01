package com.tinyppgcollector.phone.polar

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import androidx.core.content.ContextCompat
import com.polar.androidcommunications.api.ble.model.DisInfo
import com.polar.sdk.api.PolarBleApi
import com.polar.sdk.api.PolarBleApiCallback
import com.polar.sdk.api.PolarBleApiDefaultImpl
import com.polar.sdk.api.errors.PolarInvalidArgument
import com.polar.sdk.api.model.EcgSample
import com.polar.sdk.api.model.FecgSample
import com.polar.sdk.api.model.PolarDeviceInfo
import com.polar.sdk.api.model.PolarHealthThermometerData
import com.polar.sdk.api.model.PolarSensorSetting
import com.tinyppgcollector.phone.model.ConnectionState
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.reactive.asFlow
import kotlinx.coroutines.rx3.await
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull

data class PolarRecorderStatus(
    val state: ConnectionState = ConnectionState.Disconnected,
    val message: String = "Polar H10 not connected",
    val deviceId: String? = null,
)

sealed interface PolarSampleEvent {
    data class Ecg(
        val timestampUnixMs: Long,
        val elapsedRealtimeNs: Long,
        val sensorTimestampNs: Long?,
        val sampleIndex: Long,
        val ecgUv: Int,
    ) : PolarSampleEvent

    data class Hr(
        val timestampUnixMs: Long,
        val elapsedRealtimeNs: Long,
        val sampleIndex: Long,
        val hrBpm: Int,
        val rrMs: Int?,
    ) : PolarSampleEvent

    data class Acc(
        val timestampUnixMs: Long,
        val elapsedRealtimeNs: Long,
        val sensorTimestampNs: Long?,
        val sampleIndex: Long,
        val accX: Float,
        val accY: Float,
        val accZ: Float,
    ) : PolarSampleEvent
}

class PolarH10Recorder(context: Context) {
    private val appContext = context.applicationContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val connectMutex = Mutex()

    private val api: PolarBleApi = PolarBleApiDefaultImpl.defaultImplementation(
        appContext,
        setOf(
            PolarBleApi.PolarBleSdkFeature.FEATURE_HR,
            PolarBleApi.PolarBleSdkFeature.FEATURE_POLAR_ONLINE_STREAMING,
            PolarBleApi.PolarBleSdkFeature.FEATURE_DEVICE_INFO,
            PolarBleApi.PolarBleSdkFeature.FEATURE_BATTERY_INFO,
        ),
    )

    private val _status = MutableStateFlow(PolarRecorderStatus())
    val status: StateFlow<PolarRecorderStatus> = _status.asStateFlow()

    private val _errors = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val errors: SharedFlow<String> = _errors.asSharedFlow()

    private val _samples = MutableSharedFlow<PolarSampleEvent>()
    val samples: SharedFlow<PolarSampleEvent> = _samples.asSharedFlow()

    private var ecgJob: Job? = null
    private var hrJob: Job? = null
    private var accJob: Job? = null

    private val ecgSampleIndex = AtomicLong(0)
    private val hrSampleIndex = AtomicLong(0)
    private val accSampleIndex = AtomicLong(0)

    val polarSdkAvailable: Boolean = true
    var connectedDeviceId: String? = null
        private set
    var polarEcgHz: Int? = null
        private set
    var recordingActive: Boolean = false
        private set

    init {
        api.setAutomaticReconnection(true)
        api.setApiCallback(
            object : PolarBleApiCallback() {
                override fun blePowerStateChanged(powered: Boolean) {
                    if (!powered) {
                        _status.value = PolarRecorderStatus(ConnectionState.Unavailable, "Bluetooth is off")
                        _errors.tryEmit("Bluetooth is off")
                    }
                }

                override fun deviceConnecting(polarDeviceInfo: PolarDeviceInfo) {
                    val deviceId = polarDeviceInfo.deviceId
                    _status.value = PolarRecorderStatus(
                        state = ConnectionState.Connecting,
                        message = "Connecting to Polar H10 $deviceId",
                        deviceId = deviceId,
                    )
                }

                override fun deviceConnected(polarDeviceInfo: PolarDeviceInfo) {
                    val deviceId = polarDeviceInfo.deviceId
                    connectedDeviceId = deviceId
                    _status.value = PolarRecorderStatus(
                        state = ConnectionState.Connected,
                        message = "Polar H10 connected: $deviceId",
                        deviceId = deviceId,
                    )
                }

                override fun deviceDisconnected(polarDeviceInfo: PolarDeviceInfo) {
                    val deviceId = polarDeviceInfo.deviceId
                    if (connectedDeviceId == deviceId) {
                        connectedDeviceId = null
                    }
                    cancelStreamJobs()
                    recordingActive = false
                    _status.value = PolarRecorderStatus(ConnectionState.Disconnected, "Polar disconnected")
                    _errors.tryEmit("Polar disconnected")
                }

                override fun bleSdkFeatureReady(
                    identifier: String,
                    feature: PolarBleApi.PolarBleSdkFeature,
                ) {
                    if (identifier == connectedDeviceId && _status.value.state == ConnectionState.Connected) {
                        _status.value = _status.value.copy(message = "Polar H10 ready: $identifier")
                    }
                }

                override fun disInformationReceived(identifier: String, disInfo: DisInfo) = Unit

                override fun htsNotificationReceived(
                    identifier: String,
                    data: PolarHealthThermometerData,
                ) = Unit
            },
        )
    }

    suspend fun connect() {
        connectMutex.withLock {
            if (!hasBlePermissions()) {
                val message = "Bluetooth permission denied"
                _status.value = PolarRecorderStatus(ConnectionState.Error, message)
                _errors.emit(message)
                return
            }

            connectedDeviceId?.let { deviceId ->
                _status.value = PolarRecorderStatus(
                    state = ConnectionState.Connected,
                    message = "Polar H10 connected: $deviceId",
                    deviceId = deviceId,
                )
                return
            }

            _status.value = PolarRecorderStatus(ConnectionState.Connecting, "Searching for Polar H10")
            val device = scanForH10()
            if (device == null) {
                val message = "Polar not found"
                _status.value = PolarRecorderStatus(ConnectionState.Disconnected, message)
                _errors.emit(message)
                return
            }

            val deviceId = device.deviceId
            _status.value = PolarRecorderStatus(
                state = ConnectionState.Connecting,
                message = "Connecting to Polar H10 $deviceId",
                deviceId = deviceId,
            )

            try {
                api.connectToDevice(deviceId)
                withTimeout(CONNECTION_TIMEOUT_MS) {
                    api.waitForConnection(deviceId).await()
                }
                connectedDeviceId = deviceId
                _status.value = PolarRecorderStatus(
                    state = ConnectionState.Connected,
                    message = "Polar H10 connected: $deviceId",
                    deviceId = deviceId,
                )
            } catch (throwable: Throwable) {
                val message = when (throwable) {
                    is SecurityException -> "Bluetooth permission denied"
                    is PolarInvalidArgument -> "Polar connection failed: ${throwable.message ?: "invalid device id"}"
                    else -> "Polar connection failed: ${throwable.message ?: throwable.javaClass.simpleName}"
                }
                _status.value = PolarRecorderStatus(ConnectionState.Error, message, deviceId)
                _errors.emit(message)
            }
        }
    }

    suspend fun startRecording(sessionId: String): Boolean {
        if (!hasBlePermissions()) {
            val message = "Bluetooth permission denied"
            _status.value = PolarRecorderStatus(ConnectionState.Error, message, connectedDeviceId)
            _errors.emit(message)
            return false
        }

        if (connectedDeviceId == null) {
            connect()
        }

        val deviceId = connectedDeviceId
        if (deviceId == null) {
            _errors.emit("Polar H10 not connected")
            return false
        }

        ecgSampleIndex.set(0)
        hrSampleIndex.set(0)
        accSampleIndex.set(0)

        _status.value = PolarRecorderStatus(
            state = ConnectionState.Connected,
            message = "Polar H10 recording: $deviceId",
            deviceId = deviceId,
        )

        startEcgStreaming(deviceId)
        startHrStreaming(deviceId)
        startAccStreamingIfAvailable(deviceId)
        recordingActive = true
        return true
    }

    suspend fun stopRecording() {
        val deviceId = connectedDeviceId
        stopStreamJobsAndJoin()
        recordingActive = false
        if (deviceId != null) {
            runCatching {
                withTimeout(STOP_TIMEOUT_MS) {
                    api.stopHrStreaming(deviceId).await()
                }
            }
            _status.value = PolarRecorderStatus(
                state = ConnectionState.Connected,
                message = "Polar recording stopped: $deviceId",
                deviceId = deviceId,
            )
        } else {
            _status.value = PolarRecorderStatus(ConnectionState.Disconnected, "Polar H10 not connected")
        }
    }

    fun close() {
        cancelStreamJobs()
        runCatching { api.shutDown() }
        scope.cancel()
    }

    private suspend fun scanForH10(): PolarDeviceInfo? {
        return try {
            withTimeoutOrNull(SCAN_TIMEOUT_MS) {
                api.searchForDevice()
                    .asFlow()
                    .firstOrNull { device ->
                        device.isConnectable && device.name.contains("H10", ignoreCase = true)
                    }
            }
        } catch (throwable: Throwable) {
            if (throwable is SecurityException) {
                _errors.emit("Bluetooth permission denied")
            }
            null
        }
    }

    private fun startEcgStreaming(deviceId: String) {
        ecgJob?.cancel()
        ecgJob = scope.launch {
            try {
                val settings = api.requestStreamSettings(deviceId, PolarBleApi.PolarDeviceDataType.ECG)
                    .await()
                    .also { polarEcgHz = it.sampleRateHzOrNull() }
                    .maxSettings()

                api.startEcgStreaming(deviceId, settings)
                    .asFlow()
                    .onCompletion { cause ->
                        if (cause != null && isActive) {
                            emitError("ECG stream unavailable: ${cause.message ?: cause.javaClass.simpleName}")
                        }
                    }
                    .collect { data ->
                        data.samples.forEach { sample ->
                            val ecgUv = when (sample) {
                                is EcgSample -> sample.voltage
                                is FecgSample -> sample.ecg
                            }

                            _samples.emit(
                                PolarSampleEvent.Ecg(
                                    timestampUnixMs = System.currentTimeMillis(),
                                    elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
                                    sensorTimestampNs = sample.timeStamp,
                                    sampleIndex = ecgSampleIndex.getAndIncrement(),
                                    ecgUv = ecgUv,
                                ),
                            )
                        }
                    }
            } catch (throwable: Throwable) {
                if (isActive) {
                    emitError("ECG stream unavailable: ${throwable.message ?: throwable.javaClass.simpleName}")
                }
            }
        }
    }

    private fun startHrStreaming(deviceId: String) {
        hrJob?.cancel()
        hrJob = scope.launch {
            try {
                api.startHrStreaming(deviceId)
                    .asFlow()
                    .onCompletion { cause ->
                        if (cause != null && isActive) {
                            emitError("HR stream unavailable: ${cause.message ?: cause.javaClass.simpleName}")
                        }
                    }
                    .collect { data ->
                        data.samples.forEach { sample ->
                            val rrIntervals = sample.rrsMs
                            if (rrIntervals.isEmpty()) {
                                emitHrSample(sample.hr, rrMs = null)
                            } else {
                                rrIntervals.forEach { rrMs ->
                                    emitHrSample(sample.hr, rrMs)
                                }
                            }
                        }
                    }
            } catch (throwable: Throwable) {
                if (isActive) {
                    emitError("HR stream unavailable: ${throwable.message ?: throwable.javaClass.simpleName}")
                }
            }
        }
    }

    private fun startAccStreamingIfAvailable(deviceId: String) {
        accJob?.cancel()
        accJob = scope.launch {
            try {
                val available = api.getAvailableOnlineStreamDataTypes(deviceId).await()
                if (!available.contains(PolarBleApi.PolarDeviceDataType.ACC)) {
                    emitError("Polar ACC stream unavailable; continuing without it")
                    return@launch
                }

                val settings = api.requestStreamSettings(deviceId, PolarBleApi.PolarDeviceDataType.ACC)
                    .await()
                    .preferSampleRate(TARGET_ACC_HZ)

                api.startAccStreaming(deviceId, settings)
                    .asFlow()
                    .onCompletion { cause ->
                        if (cause != null && isActive) {
                            emitError("Polar ACC stream unavailable: ${cause.message ?: cause.javaClass.simpleName}")
                        }
                    }
                    .collect { data ->
                        data.samples.forEach { sample ->
                            _samples.emit(
                                PolarSampleEvent.Acc(
                                    timestampUnixMs = System.currentTimeMillis(),
                                    elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
                                    sensorTimestampNs = sample.timeStamp,
                                    sampleIndex = accSampleIndex.getAndIncrement(),
                                    accX = sample.x.toFloat(),
                                    accY = sample.y.toFloat(),
                                    accZ = sample.z.toFloat(),
                                ),
                            )
                        }
                    }
            } catch (throwable: Throwable) {
                if (isActive) {
                    emitError("Polar ACC stream unavailable: ${throwable.message ?: throwable.javaClass.simpleName}")
                }
            }
        }
    }

    private suspend fun emitHrSample(hrBpm: Int, rrMs: Int?) {
        _samples.emit(
            PolarSampleEvent.Hr(
                timestampUnixMs = System.currentTimeMillis(),
                elapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
                sampleIndex = hrSampleIndex.getAndIncrement(),
                hrBpm = hrBpm,
                rrMs = rrMs,
            ),
        )
    }

    private fun cancelStreamJobs() {
        ecgJob?.cancel()
        hrJob?.cancel()
        accJob?.cancel()
        ecgJob = null
        hrJob = null
        accJob = null
    }

    private suspend fun stopStreamJobsAndJoin() {
        val jobs = listOfNotNull(ecgJob, hrJob, accJob)
        ecgJob = null
        hrJob = null
        accJob = null
        jobs.forEach { job -> job.cancelAndJoin() }
    }

    private suspend fun emitError(message: String) {
        _errors.emit(message)
    }

    private fun hasBlePermissions(): Boolean {
        val required = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            listOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            listOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        return required.all { permission ->
            ContextCompat.checkSelfPermission(appContext, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun PolarSensorSetting.sampleRateHzOrNull(): Int? {
        return settings[PolarSensorSetting.SettingType.SAMPLE_RATE]?.maxOrNull()
    }

    private fun PolarSensorSetting.preferSampleRate(targetHz: Int): PolarSensorSetting {
        val selectedSettings = settings.mapNotNull { (type, values) ->
            val selected = if (type == PolarSensorSetting.SettingType.SAMPLE_RATE) {
                values.minByOrNull { value -> abs(value - targetHz) }
            } else {
                values.maxOrNull()
            }
            selected?.let { type to it }
        }.toMap()

        return if (selectedSettings.isEmpty()) {
            maxSettings()
        } else {
            PolarSensorSetting(selectedSettings)
        }
    }

    companion object {
        private const val SCAN_TIMEOUT_MS = 15_000L
        private const val CONNECTION_TIMEOUT_MS = 15_000L
        private const val STOP_TIMEOUT_MS = 2_000L
        private const val TARGET_ACC_HZ = 25
    }
}
