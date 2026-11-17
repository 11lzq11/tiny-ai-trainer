package com.tinytrainer

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.os.Build
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * UI-facing ViewModel. All native work goes through [TrainingManager];
 * this class only wires UI actions (start/pause/stop/generate/save/load)
 * and exposes the manager's flows to the fragments.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    val trainState = TrainingManager.trainState
    val stats = TrainingManager.stats
    val lossHistory = TrainingManager.lossHistory
    val brainInfo = TrainingManager.brainInfo
    val corpora = TrainingManager.corpora
    val messages = TrainingManager.messages

    /** The structure/last-used hyper-parameters (used for "new brain"). */
    val currentConfig: TrainConfig get() = TrainingManager.currentConfig

    private val _generating = MutableStateFlow(false)
    val generating: StateFlow<Boolean> = _generating.asStateFlow()

    private val _lastGeneration = MutableStateFlow("")
    val lastGeneration: StateFlow<String> = _lastGeneration.asStateFlow()

    init {
        TrainingManager.init(application)
        viewModelScope.launch(Dispatchers.Default) {
            TrainingManager.restoreAutoSaveIfAny()
        }
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    fun startTraining(config: TrainConfig) {
        if (!config.isValid()) {
            TrainingManager.postMessage("超参数超出范围，请检查后重试")
            return
        }
        if (trainState.value == TrainState.RUNNING) return
        viewModelScope.launch(Dispatchers.Default) {
            // If a brain with progress exists and the structure changed,
            // prepareForTraining creates a new brain (old progress is lost
            // unless exported — the Train page confirms this first).
            val ok = TrainingManager.prepareForTraining(config)
            if (!ok) return@launch
            TrainingManager.requestStart()
            val ctx = getApplication<Application>()
            val intent = TrainingService.startIntent(ctx)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ctx.startForegroundService(intent)
            } else {
                ctx.startService(intent)
            }
            TrainingManager.postMessage("训练已开始（后台继续，锁屏也不停）")
        }
    }

    fun pauseTraining() {
        TrainingManager.requestPause()
        getApplication<Application>().startService(
            Intent(getApplication(), TrainingService::class.java)
                .setAction(TrainingService.ACTION_PAUSE)
        )
    }

    fun resumeTraining() {
        TrainingManager.requestResume()
        getApplication<Application>().startService(
            Intent(getApplication(), TrainingService::class.java)
                .setAction(TrainingService.ACTION_RESUME)
        )
    }

    fun stopTraining() {
        TrainingManager.requestStop()
        getApplication<Application>().startService(
            Intent(getApplication(), TrainingService::class.java)
                .setAction(TrainingService.ACTION_STOP)
        )
        viewModelScope.launch(Dispatchers.Default) {
            // The service also autosaves when its loop exits; this is a
            // belt-and-braces save in case the service is already gone.
            kotlinx.coroutines.delay(500)
            if (TrainingManager.trainState.value == TrainState.IDLE) {
                TrainingManager.autoSave()
            }
        }
    }

    // ------------------------------------------------------------------
    // Generation
    // ------------------------------------------------------------------

    fun generate(prompt: String, maxTokens: Int, temperature: Float, topK: Int) {
        if (prompt.isBlank()) {
            TrainingManager.postMessage("请先输入 prompt")
            return
        }
        if (!TrainingManager.hasBrain) {
            TrainingManager.postMessage("还没有大脑，请先去「训练」页开始训练，或去「大脑」页新建/导入")
            return
        }
        if (_generating.value) return
        _generating.value = true
        viewModelScope.launch(Dispatchers.Default) {
            val out = TrainingManager.generate(prompt, maxTokens, temperature, topK)
            _lastGeneration.value = if (out.isEmpty()) "(生成失败)" else prompt + out
            _generating.value = false
        }
    }

    // ------------------------------------------------------------------
    // Corpus
    // ------------------------------------------------------------------

    fun addCorpus(name: String, text: String): Boolean =
        TrainingManager.addCorpus(name, text)

    fun importCorpus(uri: Uri, displayName: String?) {
        viewModelScope.launch(Dispatchers.Default) {
            TrainingManager.importCorpusFile(uri, displayName)
        }
    }

    fun setCorpusActive(name: String, active: Boolean) =
        TrainingManager.setCorpusActive(name, active)

    fun deleteCorpus(name: String) = TrainingManager.deleteCorpus(name)

    // ------------------------------------------------------------------
    // Brain file operations (SAF)
    // ------------------------------------------------------------------

    /** Copy the current brain to a user-picked location. */
    fun exportBrainTo(uri: Uri) {
        viewModelScope.launch(Dispatchers.Default) {
            if (!TrainingManager.hasBrain) {
                TrainingManager.postMessage("还没有大脑可以导出")
                return@launch
            }
            val ctx = getApplication<Application>()
            val tmp = File(ctx.filesDir, "export-tmp.brain")
            val saved = TrainingManager.saveBrainFile(tmp.absolutePath)
            if (!saved) return@launch
            try {
                ctx.contentResolver.openOutputStream(uri)?.use { out ->
                    tmp.inputStream().use { input -> input.copyTo(out) }
                }
                TrainingManager.postMessage("大脑已导出")
            } catch (e: Exception) {
                TrainingManager.postMessage("导出失败：${e.message}")
            } finally {
                tmp.delete()
            }
        }
    }

    /** Copy a user-picked .brain into internal storage and load it. */
    fun importBrainFrom(uri: Uri) {
        viewModelScope.launch(Dispatchers.Default) {
            val ctx = getApplication<Application>()
            val tmp = File(ctx.filesDir, "import-tmp.brain")
            try {
                ctx.contentResolver.openInputStream(uri)?.use { input ->
                    tmp.outputStream().use { out -> input.copyTo(out) }
                }
                val ok = TrainingManager.loadBrainFile(tmp.absolutePath)
                if (ok) {
                    val target = File(ctx.filesDir, "brain-imported.brain")
                    tmp.copyTo(target, overwrite = true)
                    TrainingManager.postMessage("大脑已导入")
                }
            } catch (e: Exception) {
                TrainingManager.postMessage("导入失败：${e.message}")
            } finally {
                tmp.delete()
            }
        }
    }

    fun createNewBrain(config: TrainConfig) {
        if (!config.isValid()) {
            TrainingManager.postMessage("超参数超出范围，请检查后重试")
            return
        }
        viewModelScope.launch(Dispatchers.Default) {
            val ok = TrainingManager.createNewBrain(config)
            if (ok) TrainingManager.postMessage("已新建大脑（随机初始化）")
        }
    }

    fun loadBaseBrain() {
        viewModelScope.launch(Dispatchers.Default) {
            when (TrainingManager.loadBaseBrainFromAssets()) {
                "OK" -> TrainingManager.postMessage("底座大脑已加载")
                "PLACEHOLDER" -> TrainingManager.postMessage("内置底座大脑尚未提供（当前是占位文件），请先新建大脑或导入已有的 .brain")
                "LOAD_FAILED" -> TrainingManager.postMessage("加载大脑失败：文件可能损坏或版本不匹配")
                else -> TrainingManager.postMessage("加载底座大脑失败")
            }
        }
    }
}
