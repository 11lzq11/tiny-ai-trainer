package com.tinytrainer.ui

import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.tinytrainer.MainViewModel
import com.tinytrainer.R
import com.tinytrainer.TrainConfig
import com.tinytrainer.TrainState
import com.tinytrainer.TrainingManager
import kotlinx.coroutines.launch

/**
 * Train page: corpus management, hyper-parameters, start/pause/stop,
 * live stats and the LOSS curve.
 */
class TrainFragment : Fragment() {

    private val viewModel: MainViewModel by activityViewModels()

    private lateinit var etCorpusName: EditText
    private lateinit var etCorpusText: EditText
    private lateinit var rvCorpus: RecyclerView
    private lateinit var tvCorpusEmpty: TextView
    private lateinit var etLr: EditText
    private lateinit var etBatch: EditText
    private lateinit var etBlock: EditText
    private lateinit var etLayers: EditText
    private lateinit var etEmbd: EditText
    private lateinit var etHeads: EditText
    private lateinit var etVocab: EditText
    private lateinit var etWarmup: EditText
    private lateinit var etWd: EditText
    private lateinit var btnStart: Button
    private lateinit var btnPause: Button
    private lateinit var btnStop: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvStep: TextView
    private lateinit var tvLoss: TextView
    private lateinit var tvTps: TextView
    private lateinit var tvElapsed: TextView
    private lateinit var lossChart: LossChartView

    private lateinit var corpusAdapter: CorpusAdapter

    private val importTxtLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        val name = queryDisplayName(uri) ?: "导入语料"
        viewModel.importCorpus(uri, name)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.fragment_train, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        etCorpusName = view.findViewById(R.id.et_corpus_name)
        etCorpusText = view.findViewById(R.id.et_corpus_text)
        rvCorpus = view.findViewById(R.id.rv_corpus)
        tvCorpusEmpty = view.findViewById(R.id.tv_corpus_empty)
        etLr = view.findViewById(R.id.et_lr)
        etBatch = view.findViewById(R.id.et_batch)
        etBlock = view.findViewById(R.id.et_block)
        etLayers = view.findViewById(R.id.et_layers)
        etEmbd = view.findViewById(R.id.et_embd)
        etHeads = view.findViewById(R.id.et_heads)
        etVocab = view.findViewById(R.id.et_vocab)
        etWarmup = view.findViewById(R.id.et_warmup)
        etWd = view.findViewById(R.id.et_wd)
        btnStart = view.findViewById(R.id.btn_start)
        btnPause = view.findViewById(R.id.btn_pause)
        btnStop = view.findViewById(R.id.btn_stop)
        tvStatus = view.findViewById(R.id.tv_status)
        tvStep = view.findViewById(R.id.tv_step)
        tvLoss = view.findViewById(R.id.tv_loss)
        tvTps = view.findViewById(R.id.tv_tps)
        tvElapsed = view.findViewById(R.id.tv_elapsed)
        lossChart = view.findViewById(R.id.loss_chart)

        corpusAdapter = CorpusAdapter(
            onToggle = { item, active ->
                viewModel.setCorpusActive(item.name, active)
                if (viewModel.trainState.value != TrainState.IDLE) {
                    Toast.makeText(
                        requireContext(),
                        "语料改动将在下次开始训练时生效",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            },
            onDelete = { item -> viewModel.deleteCorpus(item.name) }
        )
        rvCorpus.layoutManager = LinearLayoutManager(requireContext())
        rvCorpus.adapter = corpusAdapter

        view.findViewById<Button>(R.id.btn_add_corpus).setOnClickListener {
            val text = etCorpusText.text.toString()
            if (text.isBlank()) {
                Toast.makeText(requireContext(), "请先粘贴语料文本", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val ok = viewModel.addCorpus(etCorpusName.text.toString(), text)
            if (ok) {
                etCorpusText.text.clear()
                etCorpusName.text.clear()
                Toast.makeText(requireContext(), "语料已添加", Toast.LENGTH_SHORT).show()
            }
        }
        view.findViewById<Button>(R.id.btn_import_txt).setOnClickListener {
            importTxtLauncher.launch(arrayOf("text/plain", "*/*"))
        }

        btnStart.setOnClickListener { onStartClicked() }
        btnPause.setOnClickListener {
            if (viewModel.trainState.value == TrainState.PAUSED) {
                viewModel.resumeTraining()
            } else {
                viewModel.pauseTraining()
            }
        }
        btnStop.setOnClickListener { viewModel.stopTraining() }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.corpora.collect { items ->
                corpusAdapter.submitList(items)
                tvCorpusEmpty.visibility = if (items.isEmpty()) View.VISIBLE else View.GONE
            }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.trainState.collect { state -> updateControls(state) }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.stats.collect { stats ->
                tvStep.text = stats.step.toString()
                tvLoss.text = if (stats.loss.isNaN()) "—" else "%.3f".format(stats.loss)
                tvTps.text = if (stats.tokensPerSec > 0) "%.0f".format(stats.tokensPerSec) else "—"
                tvElapsed.text = formatElapsed(stats.elapsedMs)
            }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.lossHistory.collect { points ->
                lossChart.points = points
            }
        }
    }

    private fun onStartClicked() {
        val config = readConfig() ?: run {
            Toast.makeText(
                requireContext(),
                getString(R.string.train_bad_params),
                Toast.LENGTH_LONG
            ).show()
            return
        }
        // Changing the model structure discards the current brain's progress
        // (unless exported). Confirm first.
        val current = viewModel.currentConfig
        val brain = viewModel.brainInfo.value
        val structureChanged = TrainingManager.hasBrain && brain.step > 0 &&
            (config.nLayer != current.nLayer || config.nEmbd != current.nEmbd ||
                config.nHead != current.nHead || config.blockSize != current.blockSize ||
                config.vocabSize != current.vocabSize)
        if (structureChanged) {
            AlertDialog.Builder(requireContext())
                .setMessage("结构参数变了，开始训练会新建大脑，当前大脑的进度将丢失（除非已导出）。确定吗？")
                .setPositiveButton(R.string.dialog_confirm) { _, _ ->
                    viewModel.startTraining(config)
                }
                .setNegativeButton(R.string.dialog_cancel, null)
                .show()
        } else {
            viewModel.startTraining(config)
        }
    }

    /** Parse the hyper-parameter fields; null if any value is missing/invalid. */
    private fun readConfig(): TrainConfig? {
        val lr = etLr.text.toString().toFloatOrNull() ?: return null
        val batch = etBatch.text.toString().toIntOrNull() ?: return null
        val block = etBlock.text.toString().toIntOrNull() ?: return null
        val layers = etLayers.text.toString().toIntOrNull() ?: return null
        val embd = etEmbd.text.toString().toIntOrNull() ?: return null
        val heads = etHeads.text.toString().toIntOrNull() ?: return null
        val vocab = etVocab.text.toString().toIntOrNull() ?: return null
        val warmup = etWarmup.text.toString().toIntOrNull() ?: return null
        val wd = etWd.text.toString().toFloatOrNull() ?: return null
        val config = TrainConfig(
            lr = lr, batchSize = batch, blockSize = block, nLayer = layers,
            nEmbd = embd, nHead = heads, vocabSize = vocab,
            warmupSteps = warmup, weightDecay = wd
        )
        return if (config.isValid()) config else null
    }

    private fun updateControls(state: TrainState) {
        when (state) {
            TrainState.IDLE -> {
                tvStatus.text = getString(R.string.train_status_idle)
                btnStart.isEnabled = true
                btnPause.isEnabled = false
                btnPause.text = getString(R.string.train_pause)
                btnStop.isEnabled = false
            }
            TrainState.RUNNING -> {
                tvStatus.text = getString(R.string.train_status_running)
                btnStart.isEnabled = false
                btnPause.isEnabled = true
                btnPause.text = getString(R.string.train_pause)
                btnStop.isEnabled = true
            }
            TrainState.PAUSED -> {
                tvStatus.text = getString(R.string.train_status_paused)
                btnStart.isEnabled = false
                btnPause.isEnabled = true
                btnPause.text = getString(R.string.train_resume)
                btnStop.isEnabled = true
            }
        }
    }

    private fun formatElapsed(ms: Long): String {
        val totalSeconds = ms / 1000
        val h = totalSeconds / 3600
        val m = (totalSeconds % 3600) / 60
        val s = totalSeconds % 60
        return if (h > 0) "%d:%02d:%02d".format(h, m, s) else "%02d:%02d".format(m, s)
    }

    private fun queryDisplayName(uri: Uri): String? {
        return try {
            requireContext().contentResolver.query(
                uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME))
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }
}
