package com.tinyppgcollector.wear

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.tinyppgcollector.wear.permissions.WatchPermissionManager
import com.tinyppgcollector.wear.recording.WatchRecordingStateStore

class WearMainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            WearTheme {
                WatchStatusScreen()
            }
        }
    }
}

@Composable
private fun WearTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Color(0xFF4CA391),
            background = Color(0xFF071A17),
            surface = Color(0xFF0D2A25),
            onSurface = Color(0xFFE7FFF7),
            onPrimary = Color(0xFF071A17),
        ),
        content = content,
    )
}

@Composable
private fun WatchStatusScreen() {
    val context = LocalContext.current
    val recordingState by WatchRecordingStateStore.state.collectAsState()
    var missingPermissions by remember {
        mutableStateOf(WatchPermissionManager.missingPermissions(context).toList())
    }
    var initialPermissionRequestLaunched by remember { mutableStateOf(false) }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) {
        missingPermissions = WatchPermissionManager.missingPermissions(context).toList()
    }

    LaunchedEffect(missingPermissions) {
        if (!initialPermissionRequestLaunched && missingPermissions.isNotEmpty()) {
            initialPermissionRequestLaunched = true
            permissionLauncher.launch(missingPermissions.toTypedArray())
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    colors = listOf(Color(0xFF17443B), Color(0xFF071A17)),
                ),
            )
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Surface(
            color = Color(0xFF4CA391),
            shape = CircleShape,
        ) {
            Text(
                text = "PPG",
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                fontWeight = FontWeight.Bold,
                color = Color(0xFF071A17),
            )
        }
        Text(
            text = "TinyPPG Watch",
            modifier = Modifier.padding(top = 14.dp),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = Color(0xFFE7FFF7),
            textAlign = TextAlign.Center,
        )
        Text(
            text = if (missingPermissions.isEmpty()) {
                recordingState.readyStatus
            } else {
                "Sensor permission required"
            },
            modifier = Modifier.padding(top = 6.dp),
            style = MaterialTheme.typography.bodySmall,
            color = Color(0xFFB8D8D0),
            textAlign = TextAlign.Center,
        )
        if (missingPermissions.isNotEmpty()) {
            Button(
                modifier = Modifier.padding(top = 10.dp),
                onClick = { permissionLauncher.launch(missingPermissions.toTypedArray()) },
            ) {
                Text("Grant")
            }
        }
        Surface(
            modifier = Modifier.padding(top = 12.dp),
            color = Color(0xFF0D2A25),
            shape = MaterialTheme.shapes.small,
        ) {
            Column(
                modifier = Modifier.padding(10.dp),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                WatchValue("Last command", recordingState.lastCommand)
                WatchValue("Recording", recordingState.recording.toString())
                WatchValue("PPG available", recordingState.ppgAvailable.toString())
                WatchValue("IMU available", recordingState.imuAvailable.toString())
                WatchValue("PPG samples", recordingState.ppgSamples.toString())
                WatchValue("IMU samples", recordingState.imuSamples.toString())
                WatchValue("Last error", recordingState.lastError.ifBlank { "none" })
            }
        }
    }
}

@Composable
private fun WatchValue(label: String, value: String) {
    Text(
        text = "$label: $value",
        style = MaterialTheme.typography.bodySmall,
        color = Color(0xFFE7FFF7),
        textAlign = TextAlign.Center,
    )
}
