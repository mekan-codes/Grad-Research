package com.tinyppgcollector.phone

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
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
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.tinyppgcollector.phone.model.ActivityLabel
import com.tinyppgcollector.phone.model.ConnectionState
import com.tinyppgcollector.phone.model.RecordingUiState
import com.tinyppgcollector.phone.permissions.PermissionManager

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TinyPpgTheme {
                TinyPpgCollectorScreen()
            }
        }
    }
}

@Composable
private fun TinyPpgTheme(content: @Composable () -> Unit) {
    val colors = lightColorScheme(
        primary = Color(0xFF146C5B),
        secondary = Color(0xFF546A2F),
        tertiary = Color(0xFF9A4F38),
        background = Color(0xFFF4F7F2),
        surface = Color(0xFFFFFFFF),
        onPrimary = Color.White,
        onSecondary = Color.White,
    )
    MaterialTheme(colorScheme = colors, content = content)
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TinyPpgCollectorScreen(viewModel: RecordingViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    var initialPermissionRequestLaunched by remember { mutableStateOf(false) }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        if (result.values.any { !it }) {
            viewModel.onPermissionsDenied()
        }
    }

    LaunchedEffect(Unit) {
        if (!initialPermissionRequestLaunched) {
            val missing = PermissionManager.missingPermissions(context)
            if (missing.isNotEmpty()) {
                initialPermissionRequestLaunched = true
                permissionLauncher.launch(missing)
            }
        }
    }

    Scaffold { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        listOf(Color(0xFFEAF3ED), Color(0xFFF9FBF8)),
                    ),
                )
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = "TinyPPGCollector",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF102D28),
            )
            Text(
                text = "Phone controller for Galaxy Watch 5 PPG/IMU and Polar H10 ECG/HR sessions.",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF4B5E58),
            )

            InputSection(uiState, viewModel)

            CardSection(title = "Activity") {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    ActivityLabel.entries.forEach { label ->
                        FilterChip(
                            selected = uiState.activityLabel == label,
                            onClick = { viewModel.updateActivityLabel(label) },
                            label = { Text(label.displayName) },
                        )
                    }
                }
                if (uiState.activityLabel == ActivityLabel.Custom) {
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = uiState.customActivityLabel,
                        onValueChange = viewModel::updateCustomActivityLabel,
                        label = { Text("Custom activity label") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }

            CardSection(title = "Connections") {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    OutlinedButton(
                        onClick = { viewModel.connectWatch() },
                        enabled = !uiState.isRecording,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Connect Watch")
                    }
                    OutlinedButton(
                        onClick = {
                            val missing = PermissionManager.missingPermissions(context)
                            if (missing.isEmpty()) viewModel.connectPolar() else permissionLauncher.launch(missing)
                        },
                        enabled = !uiState.isRecording,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Connect Polar H10")
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
                StatusLine("Watch", uiState.watchConnectionState, uiState.watchStatusText)
                StatusLine("Polar", uiState.polarConnectionState, uiState.polarStatusText)
            }

            CardSection(title = "Recording") {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Button(
                        onClick = {
                            val missing = PermissionManager.missingPermissions(context)
                            if (missing.isEmpty()) viewModel.startRecording() else permissionLauncher.launch(missing)
                        },
                        enabled = !uiState.isRecording && uiState.subjectId.isNotBlank(),
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Start Recording")
                    }
                    OutlinedButton(
                        onClick = viewModel::stopRecording,
                        enabled = uiState.isRecording,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Stop Recording")
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
                StatusValue("Elapsed", formatElapsed(uiState.elapsedSeconds))
                StatusValue("Watch PPG samples", uiState.counts.watchPpg.toString())
                StatusValue("Watch IMU samples", uiState.counts.watchImu.toString())
                StatusValue("Polar ECG samples", uiState.counts.polarEcg.toString())
                StatusValue("Polar HR rows", uiState.counts.polarHr.toString())
                StatusValue("Polar ACC samples", uiState.counts.polarAcc.toString())
            }

            uiState.savedFolderPath?.let { path ->
                CardSection(title = "Saved Folder") {
                    Text(path, style = MaterialTheme.typography.bodyMedium)
                    uiState.fileValidationSummary?.let { summary ->
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(summary, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            uiState.lastError?.let { error ->
                Surface(
                    color = Color(0xFFFFE6DF),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = error,
                        modifier = Modifier.padding(12.dp),
                        color = Color(0xFF7A2F20),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }
    }
}

@Composable
private fun InputSection(uiState: RecordingUiState, viewModel: RecordingViewModel) {
    CardSection(title = "Session") {
        OutlinedTextField(
            value = uiState.subjectId,
            onValueChange = viewModel::updateSubjectId,
            label = { Text("Subject ID") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedTextField(
            value = uiState.notes,
            onValueChange = viewModel::updateNotes,
            label = { Text("Session note") },
            minLines = 3,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun CardSection(title: String, content: @Composable () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = Color(0xFF102D28),
            )
            Spacer(modifier = Modifier.height(4.dp))
            content()
        }
    }
}

@Composable
private fun StatusLine(label: String, state: ConnectionState, message: String) {
    val color = when (state) {
        ConnectionState.Connected -> Color(0xFF146C5B)
        ConnectionState.Connecting -> Color(0xFF785A00)
        ConnectionState.Unavailable, ConnectionState.Error -> Color(0xFF9A3E2E)
        ConnectionState.Disconnected -> Color(0xFF596560)
    }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, fontWeight = FontWeight.Medium)
        Text(message, color = color, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun StatusValue(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, color = Color(0xFF4B5E58))
        Text(value, fontWeight = FontWeight.SemiBold)
    }
}

private fun formatElapsed(seconds: Long): String {
    val minutes = seconds / 60
    val remainingSeconds = seconds % 60
    return "%02d:%02d".format(minutes, remainingSeconds)
}
