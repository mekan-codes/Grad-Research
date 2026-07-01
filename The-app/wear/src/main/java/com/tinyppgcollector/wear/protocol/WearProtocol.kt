package com.tinyppgcollector.wear.protocol

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
