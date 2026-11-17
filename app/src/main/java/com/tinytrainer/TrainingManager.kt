package com.tinytrainer

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Single owner of the native Trainer handle and all training state.
 *
 * Runs in the same process as both the UI and [TrainingService], so the
 * service and the fragments share this object directly (no Binder needed).
 *
 * Threading contract with the C engine:
 * - The training loop calls nativeTrainSteps in small chunks (CHUNK_STEPS).
 * - [trainMutex] is held ONLY during each native call. Saving a brain (for
 *   export / autosave / generate-snapshot) takes the same mutex, so it waits
 *   at most one chunk and never races a weight update.
 * - Generation during training snapshots first: save the live weights to a
 *   temp .brain under the mutex, then load that snapshot into a SECOND
 *   handle and generate from it — the training loop is never blocked.
 */
object TrainingManager {

    private const val TAG = "TrainingManager"
    private const val CHUNK_STEPS = 5
    private const val MAX_HISTORY_POINTS = 3000
    private const val MAX_CORPUS_CHARS = 5_000_000
    private const val PREFS_NAME = "tiny_trainer"
    private const val KEY_ACTIVE_CORPORA = "active_corpora"

    private lateinit var appContext: Context

    private var trainer: Trainer? = null
    private var handle: Long = 0L
    private var config: TrainConfig = TrainConfig()

    /** Serialises native calls that touch weights (train chunk / save). */
    private val trainMutex = Mutex()

    /** Tokens processed by the current brain (accumulates across runs). */
    private var totalTokens = 0L

    private val _trainState = MutableStateFlow(TrainState.IDLE)
    val trainState: StateFlow<TrainState> = _trainState.asStateFlow()

    private val _stats = MutableStateFlow(TrainStats())
    val stats: StateFlow<TrainStats> = _stats.asStateFlow()

    /** (step, loss) points for the chart, oldest first, capped. */
    private val _lossHistory = MutableStateFlow<List<Pair<Long, Float>>>(emptyList())
    val lossHistory: StateFlow<List<Pair<Long, Float>>> = _lossHistory.asStateFlow()

    private val _brainInfo = MutableStateFlow(BrainInfo())
    val brainInfo: StateFlow<BrainInfo> = _brainInfo.asStateFlow()

    private val _corpora = MutableStateFlow<List<CorpusItem>>(emptyList())
    val corpora: StateFlow<List<CorpusItem>> = _corpora.asStateFlow()

    /** One-shot UI messages (toasts). */
    private val _messages = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val messages: SharedFlow<String> = _messages.asSharedFlow()

    val currentConfig: TrainConfig get() = config
    val hasBrain: Boolean get() = handle != 0L

    fun init(context: Context) {
        if (!::appContext.isInitialized) {
            appContext = context.applicationContext
            refreshCorpora()
        }
    }

    fun postMessage(msg: String) {
        _messages.tryEmit(msg)
    }

    // ------------------------------------------------------------------
    // Brain lifecycle
    // ------------------------------------------------------------------

    /** Create a brand-new brain from [newConfig] (random init), dropping the old one. */
    suspend fun createNewBrain(newConfig: TrainConfig): Boolean = withContext(Dispatchers.Default) {
        try {
            val t = getOrCreateTrainer()
            if (handle != 0L) {
                trainMutex.withLock { t.nativeFree(handle) }
                handle = 0L
            }
            config = newConfig
            val h = t.nativeCreate(newConfig.toJson().toString())
            if (h == 0L) {
                postMessage("创建大脑失败：引擎返回空句柄")
                return@withContext false
            }
            handle = h
            resetRunStats()
            updateBrainInfo()
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "native lib missing", e)
            postMessage("引擎加载失败：${e.message}")
            false
        } catch (e: Exception) {
            Log.e(TAG, "createNewBrain failed", e)
            postMessage("创建大脑失败：${e.message}")
            false
        }
    }

    /** Load a .brain file (import / base brain / autosave). */
    suspend fun loadBrainFile(path: String): Boolean = withContext(Dispatchers.Default) {
        try {
            val t = getOrCreateTrainer()
            if (handle == 0L) {
                // No handle yet: create one with the default config. The engine
                // MUST resize the model to the file's header structure on load.
                val h = t.nativeCreate(TrainConfig().toJson().toString())
                if (h == 0L) {
                    postMessage("创建大脑失败：引擎返回空句柄")
                    return@withContext false
                }
                handle = h
            }
            val ok = trainMutex.withLock { t.nativeLoadBrain(handle, path) }
            if (!ok) {
                postMessage("加载大脑失败：文件可能损坏或版本不匹配")
                return@withContext false
            }
            // The loaded brain may have a different structure than our config.
            // The .brain header JSON (SPEC section 2) carries the structure;
            // adopt it so the Brain page shows the right architecture.
            parseBrainHeader(path)?.let { header ->
                val model = header.optJSONObject("model")
                if (model != null) {
                    config = config.copy(
                        nLayer = model.optInt("n_layer", config.nLayer),
                        nEmbd = model.optInt("n_embd", config.nEmbd),
                        nHead = model.optInt("n_head", config.nHead),
                        blockSize = model.optInt("block_size", config.blockSize),
                        vocabSize = model.optInt("vocab_size", config.vocabSize)
                    )
                }
            }
            // The loaded brain may have a different structure than our config;
            // refresh what we display from the native side.
            resetRunStats(keepStep = true)
            updateBrainInfo()
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "native lib missing", e)
            postMessage("引擎加载失败：${e.message}")
            false
        } catch (e: Exception) {
            Log.e(TAG, "loadBrainFile failed", e)
            postMessage("加载大脑失败：${e.message}")
            false
        }
    }

    /** Save the current brain to [path]. Safe to call while training. */
    suspend fun saveBrainFile(path: String): Boolean = withContext(Dispatchers.Default) {
        if (handle == 0L) {
            postMessage("还没有大脑可以保存")
            return@withContext false
        }
        try {
            val ok = trainMutex.withLock { trainer!!.nativeSaveBrain(handle, path) }
            if (!ok) postMessage("保存大脑失败")
            ok
        } catch (e: Exception) {
            Log.e(TAG, "saveBrainFile failed", e)
            postMessage("保存大脑失败：${e.message}")
            false
        }
    }

    /** Autosave used when training stops. */
    suspend fun autoSave(): Boolean {
        if (handle == 0L) return false
        val f = File(appContext.filesDir, "brain-autosave.brain")
        return saveBrainFile(f.absolutePath)
    }

    /** Restore the autosave on app start, if one exists. */
    suspend fun restoreAutoSaveIfAny(): Boolean {
        val f = File(appContext.filesDir, "brain-autosave.brain")
        if (!f.exists()) return false
        return loadBrainFile(f.absolutePath)
    }

    // ------------------------------------------------------------------
    // Corpus management (files under filesDir/corpora/)
    // ------------------------------------------------------------------

    private fun corpusDir(): File = File(appContext.filesDir, "corpora").apply { mkdirs() }

    private fun activeNames(): MutableSet<String> {
        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getStringSet(KEY_ACTIVE_CORPORA, emptySet())?.toMutableSet() ?: mutableSetOf()
    }

    private fun saveActiveNames(names: Set<String>) {
        appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putStringSet(KEY_ACTIVE_CORPORA, names).apply()
    }

    fun refreshCorpora() {
        val dir = corpusDir()
        val active = activeNames()
        val items = dir.listFiles()
            ?.filter { it.isFile && it.name.endsWith(".txt") }
            ?.sortedByDescending { it.lastModified() }
            ?.map { CorpusItem(it.nameWithoutExtension, it.length(), it.name in active) }
            ?: emptyList()
        _corpora.value = items
    }

    /** Add a corpus from pasted text. Returns false on empty text. */
    fun addCorpus(rawName: String, text: String): Boolean {
        val clean = text.trim()
        if (clean.isEmpty()) return false
        var base = rawName.trim().ifEmpty { "语料${_corpora.value.size + 1}" }
        base = base.replace(Regex("[^A-Za-z0-9_\\-\\u4e00-\\u9fa5]+"), "_").take(40)
        val dir = corpusDir()
        var file = File(dir, "$base.txt")
        var n = 2
        while (file.exists()) {
            file = File(dir, "$base($n).txt")
            n++
        }
        file.writeText(clean, Charsets.UTF_8)
        val active = activeNames().apply { add(file.name) }
        saveActiveNames(active)
        refreshCorpora()
        return true
    }

    /** Import a .txt picked with the Storage Access Framework. */
    suspend fun importCorpusFile(uri: Uri, displayName: String?): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val bytes = appContext.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: return@withContext false
                val text = String(bytes, Charsets.UTF_8)
                val name = displayName?.substringBeforeLast(".txt")
                    ?.substringBeforeLast(".TXT") ?: "导入语料"
                // addCorpus only touches files + SharedPreferences + a StateFlow,
                // so it is safe to run on the IO dispatcher.
                addCorpus(name, text)
            } catch (e: Exception) {
                Log.e(TAG, "importCorpusFile failed", e)
                postMessage("导入语料失败：${e.message}")
                false
            }
        }

    fun setCorpusActive(name: String, active: Boolean) {
        val names = activeNames()
        val fileName = "$name.txt"
        if (active) names.add(fileName) else names.remove(fileName)
        saveActiveNames(names)
        refreshCorpora()
    }

    fun deleteCorpus(name: String) {
        val file = File(corpusDir(), "$name.txt")
        if (file.exists()) file.delete()
        val names = activeNames().apply { remove("$name.txt") }
        saveActiveNames(names)
        refreshCorpora()
    }

    /** Concatenated text of all active corpora (capped for phone memory). */
    fun buildCorpusText(): String {
        val active = _corpora.value.filter { it.active }
        val sb = StringBuilder()
        for (item in active) {
            val file = File(corpusDir(), "${item.name}.txt")
            if (!file.exists()) continue
            sb.append(file.readText(Charsets.UTF_8)).append('\n')
            if (sb.length >= MAX_CORPUS_CHARS) break
        }
        val text = sb.toString()
        return if (text.length > MAX_CORPUS_CHARS) text.substring(0, MAX_CORPUS_CHARS) else text
    }

    // ------------------------------------------------------------------
    // Training control (called from the ViewModel / Service)
    // ------------------------------------------------------------------

    /**
     * Prepare everything for a training run: create or update the brain,
     * apply the live-tunable params, and push the corpus to the engine.
     * Returns false if anything fails (message already posted).
     */
    suspend fun prepareForTraining(newConfig: TrainConfig): Boolean =
        withContext(Dispatchers.Default) {
            val corpusText = buildCorpusText()
            if (corpusText.isBlank()) {
                postMessage("请先添加至少一份语料并勾选启用")
                return@withContext false
            }
            try {
                val t = getOrCreateTrainer()
                if (handle == 0L) {
                    config = newConfig
                    val h = t.nativeCreate(newConfig.toJson().toString())
                    if (h == 0L) {
                        postMessage("创建大脑失败：引擎返回空句柄")
                        return@withContext false
                    }
                    handle = h
                } else if (structureChanged(newConfig)) {
                    // Structure params cannot change on a live brain.
                    val ok = createNewBrain(newConfig)
                    if (!ok) return@withContext false
                } else {
                    config = newConfig
                    // Live-tunable params.
                    trainMutex.withLock {
                        t.nativeSetLr(handle, newConfig.lr)
                        t.nativeSetBatchSize(handle, newConfig.batchSize)
                    }
                }
                val ok = trainMutex.withLock { t.nativeSetCorpus(handle, corpusText) }
                if (!ok) {
                    postMessage("设置语料失败")
                    return@withContext false
                }
                updateBrainInfo()
                true
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "native lib missing", e)
                postMessage("引擎加载失败：${e.message}")
                false
            } catch (e: Exception) {
                Log.e(TAG, "prepareForTraining failed", e)
                postMessage("准备训练失败：${e.message}")
                false
            }
        }

    private fun structureChanged(newConfig: TrainConfig): Boolean =
        newConfig.nLayer != config.nLayer ||
            newConfig.nEmbd != config.nEmbd ||
            newConfig.nHead != config.nHead ||
            newConfig.blockSize != config.blockSize ||
            newConfig.vocabSize != config.vocabSize

    fun requestStart() {
        _trainState.value = TrainState.RUNNING
    }

    fun requestPause() {
        if (_trainState.value == TrainState.RUNNING) _trainState.value = TrainState.PAUSED
    }

    fun requestResume() {
        if (_trainState.value == TrainState.PAUSED) _trainState.value = TrainState.RUNNING
    }

    fun requestStop() {
        _trainState.value = TrainState.IDLE
    }

    /**
     * The training loop. Runs inside [TrainingService]'s coroutine scope.
     * Exits when the state becomes IDLE (stop). Pause is honoured between
     * chunks, so pause/stop take effect within one chunk (a few seconds).
     */
    suspend fun trainingLoop() {
        var elapsedMs = _stats.value.elapsedMs
        while (_trainState.value == TrainState.RUNNING ||
            _trainState.value == TrainState.PAUSED
        ) {
            if (_trainState.value == TrainState.PAUSED) {
                delay(200)
                continue
            }
            val h = handle
            val t = trainer
            if (h == 0L || t == null) {
                delay(500)
                continue
            }
            val chunkStart = android.os.SystemClock.elapsedRealtime()
            val losses: FloatArray = try {
                withContext(Dispatchers.Default) {
                    trainMutex.withLock { t.nativeTrainSteps(h, CHUNK_STEPS) }
                }
            } catch (e: Exception) {
                Log.e(TAG, "nativeTrainSteps failed", e)
                postMessage("训练出错：${e.message}")
                _trainState.value = TrainState.IDLE
                return
            }
            val chunkMs = android.os.SystemClock.elapsedRealtime() - chunkStart
            if (losses.isEmpty()) {
                delay(500)
                continue
            }
            elapsedMs += chunkMs
            totalTokens += losses.size.toLong() * config.batchSize * config.blockSize
            val step = (_stats.value.step + losses.size)
            val loss = losses.last()
            val tps = if (elapsedMs > 0) totalTokens * 1000.0 / elapsedMs else 0.0
            _stats.value = TrainStats(step, loss, tps, elapsedMs)
            appendLossHistory(step - losses.size + 1, losses)
            _brainInfo.value = _brainInfo.value.copy(step = step, loss = loss)
        }
    }

    private fun appendLossHistory(firstStep: Long, losses: FloatArray) {
        val current = _lossHistory.value.toMutableList()
        losses.forEachIndexed { i, loss ->
            current.add(Pair(firstStep + i, loss))
        }
        while (current.size > MAX_HISTORY_POINTS) current.removeAt(0)
        _lossHistory.value = current
    }

    // ------------------------------------------------------------------
    // Generation (snapshot-safe during training)
    // ------------------------------------------------------------------

    /**
     * Generate text. If training is running, the live weights are first
     * snapshotted to a temp .brain (under the mutex), then loaded into a
     * SECOND handle and generated from there — training is never blocked.
     * When idle, generate directly from the live handle.
     */
    suspend fun generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int
    ): String = withContext(Dispatchers.Default) {
        val h = handle
        val t = trainer
        if (h == 0L || t == null) return@withContext ""
        try {
            if (_trainState.value == TrainState.RUNNING ||
                _trainState.value == TrainState.PAUSED
            ) {
                // Snapshot path.
                val snap = File(appContext.filesDir, "snapshot.brain")
                val saved = trainMutex.withLock { t.nativeSaveBrain(h, snap.absolutePath) }
                if (!saved) return@withContext ""
                val h2 = t.nativeCreate(config.toJson().toString())
                if (h2 == 0L) return@withContext ""
                try {
                    if (!t.nativeLoadBrain(h2, snap.absolutePath)) return@withContext ""
                    t.nativeGenerate(
                        h2, prompt, maxTokens, temperature, topK,
                        System.currentTimeMillis()
                    )
                } finally {
                    t.nativeFree(h2)
                    snap.delete()
                }
            } else {
                // Idle: generate from the live handle, still under the mutex
                // so a training run starting mid-generation cannot race it.
                trainMutex.withLock {
                    t.nativeGenerate(
                        h, prompt, maxTokens, temperature, topK,
                        System.currentTimeMillis()
                    )
                }
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "native lib missing", e)
            ""
        } catch (e: Exception) {
            Log.e(TAG, "generate failed", e)
            ""
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private fun getOrCreateTrainer(): Trainer {
        if (trainer == null) trainer = Trainer()
        return trainer!!
    }

    /**
     * Read the .brain header JSON (SPEC section 2):
     * magic "TBRN" (4B) | version u32 LE | header_len u32 LE | header JSON.
     * Returns null if the file is not a valid .brain.
     */
    private fun parseBrainHeader(path: String): JSONObject? {
        return try {
            File(path).inputStream().use { input ->
                val magic = ByteArray(4)
                if (input.read(magic) != 4) return null
                if (!magic.contentEquals("TBRN".toByteArray(Charsets.US_ASCII))) return null
                val lenBytes = ByteArray(8)
                if (input.read(lenBytes) != 8) return null
                val buf = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN)
                buf.int // version, ignored here
                val headerLen = buf.int
                if (headerLen <= 0 || headerLen > 1_000_000) return null
                val headerBytes = ByteArray(headerLen)
                var read = 0
                while (read < headerLen) {
                    val n = input.read(headerBytes, read, headerLen - read)
                    if (n < 0) return null
                    read += n
                }
                JSONObject(String(headerBytes, Charsets.UTF_8))
            }
        } catch (e: Exception) {
            Log.w(TAG, "parseBrainHeader failed", e)
            null
        }
    }

    private fun resetRunStats(keepStep: Boolean = false) {
        val step = if (keepStep && handle != 0L) {
            runCatching { trainer!!.nativeGetStep(handle) }.getOrDefault(0L)
        } else 0L
        totalTokens = 0L
        _stats.value = TrainStats(step = step)
        _lossHistory.value = emptyList()
    }

    private fun updateBrainInfo() {
        val t = trainer ?: return
        if (handle == 0L) {
            _brainInfo.value = BrainInfo()
            return
        }
        val step = runCatching { t.nativeGetStep(handle) }.getOrDefault(0L)
        val params = runCatching { t.nativeParamCount(handle) }.getOrDefault(0L)
        val vocab = runCatching { t.nativeVocabSize(handle) }.getOrDefault(0)
        _brainInfo.value = BrainInfo(
            step = step,
            paramCount = params,
            vocabSize = vocab,
            nLayer = config.nLayer,
            nEmbd = config.nEmbd,
            nHead = config.nHead,
            blockSize = config.blockSize,
            loss = _stats.value.loss
        )
    }

    /** Load the bundled base brain (assets/base.brain). Returns a status string. */
    suspend fun loadBaseBrainFromAssets(): String = withContext(Dispatchers.IO) {
        try {
            val out = File(appContext.filesDir, "base.brain")
            appContext.assets.open("base.brain").use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            }
            // The first release ships a placeholder, not a real .brain.
            // A real .brain starts with the magic "TBRN".
            val magic = ByteArray(4)
            out.inputStream().use { it.read(magic) }
            if (!magic.contentEquals("TBRN".toByteArray(Charsets.US_ASCII))) {
                out.delete()
                return@withContext "PLACEHOLDER"
            }
            val ok = loadBrainFile(out.absolutePath)
            if (ok) "OK" else "LOAD_FAILED"
        } catch (e: Exception) {
            Log.e(TAG, "loadBaseBrainFromAssets failed", e)
            "ERROR"
        }
    }
}
