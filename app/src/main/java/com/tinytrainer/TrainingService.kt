package com.tinytrainer

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

/**
 * Foreground service that owns the training loop.
 *
 * Started with [ACTION_START]; the loop runs in [TrainingManager] and exits
 * when the state becomes IDLE (stop). On exit the service autosaves the
 * brain and stops itself. Pause/resume are state changes only — the loop
 * honours them between chunks.
 *
 * The service is a plain started service (no binding): the UI and the
 * service share TrainingManager in-process.
 */
class TrainingService : Service() {

    companion object {
        const val ACTION_START = "com.tinytrainer.action.START"
        const val ACTION_PAUSE = "com.tinytrainer.action.PAUSE"
        const val ACTION_RESUME = "com.tinytrainer.action.RESUME"
        const val ACTION_STOP = "com.tinytrainer.action.STOP"

        const val CHANNEL_ID = "tinytrainer_training"
        const val NOTIF_ID = 1001

        fun startIntent(context: Context): Intent =
            Intent(context, TrainingService::class.java).setAction(ACTION_START)
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var lastNotifUpdateMs = 0L
    private var loopActive = false

    override fun onCreate() {
        super.onCreate()
        TrainingManager.init(this)
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startTraining()
            ACTION_PAUSE -> {
                TrainingManager.requestPause()
                if (!loopActive) stopSelf()
            }
            ACTION_RESUME -> {
                TrainingManager.requestResume()
                if (!loopActive) stopSelf()
            }
            ACTION_STOP -> {
                TrainingManager.requestStop()
                if (!loopActive) stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun startTraining() {
        startForegroundCompat(buildNotification(TrainState.RUNNING, TrainStats()))
        loopActive = true

        // The loop itself lives in TrainingManager; when it returns, training
        // was stopped: autosave and shut the service down.
        serviceScope.launch {
            TrainingManager.trainingLoop()
            loopActive = false
            try {
                TrainingManager.autoSave()
            } catch (e: Exception) {
                Log.w("TrainingService", "autosave failed", e)
            }
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }

        // Keep the notification fresh (throttled to ~1 update/second).
        serviceScope.launch {
            combine(TrainingManager.trainState, TrainingManager.stats) { state, stats ->
                Pair(state, stats)
            }.collect { (state, stats) ->
                val now = android.os.SystemClock.elapsedRealtime()
                if (now - lastNotifUpdateMs < 1000) return@collect
                lastNotifUpdateMs = now
                val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
                nm.notify(NOTIF_ID, buildNotification(state, stats))
            }
        }
    }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIF_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            @Suppress("DEPRECATION")
            startForeground(NOTIF_ID, notification)
        }
    }

    private fun buildNotification(state: TrainState, stats: TrainStats): Notification {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pending = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val text = when (state) {
            TrainState.RUNNING -> getString(
                R.string.notif_training,
                stats.step,
                if (stats.loss.isNaN()) 0f else stats.loss
            )
            TrainState.PAUSED -> getString(R.string.notif_paused, stats.step)
            TrainState.IDLE -> getString(R.string.train_status_idle)
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentIntent(pending)
            .setOngoing(state != TrainState.IDLE)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notif_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = getString(R.string.notif_channel_desc) }
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }
}
