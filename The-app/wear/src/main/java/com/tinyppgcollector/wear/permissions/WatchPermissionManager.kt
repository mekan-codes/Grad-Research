package com.tinyppgcollector.wear.permissions

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat

object WatchPermissionManager {
    const val SAMSUNG_ADDITIONAL_HEALTH_DATA =
        "com.samsung.android.hardware.sensormanager.permission.READ_ADDITIONAL_HEALTH_DATA"

    fun requiredPermissions(): Array<String> {
        val permissions = mutableListOf<String>()
        permissions += Manifest.permission.BODY_SENSORS
        permissions += SAMSUNG_ADDITIONAL_HEALTH_DATA
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            permissions += Manifest.permission.ACTIVITY_RECOGNITION
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions += Manifest.permission.POST_NOTIFICATIONS
        }
        return permissions.distinct().toTypedArray()
    }

    fun missingPermissions(context: Context): Array<String> {
        return requiredPermissions()
            .filter { permission ->
                ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED
            }
            .toTypedArray()
    }

    fun hasRequiredPermissions(context: Context): Boolean = missingPermissions(context).isEmpty()
}
