package com.tinytrainer.ui

import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.tinytrainer.BrainInfo
import com.tinytrainer.MainViewModel
import com.tinytrainer.R
import com.tinytrainer.TrainingManager
import kotlinx.coroutines.launch

/**
 * Brain page: current brain info, export / import (.brain), new brain,
 * load the bundled base brain.
 */
class BrainFragment : Fragment() {

    private val viewModel: MainViewModel by activityViewModels()

    private lateinit var tvBrainInfo: TextView

    private val exportLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/octet-stream")
    ) { uri: Uri? ->
        if (uri != null) viewModel.exportBrainTo(uri)
    }

    private val importLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) viewModel.importBrainFrom(uri)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.fragment_brain, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        tvBrainInfo = view.findViewById(R.id.tv_brain_info)

        view.findViewById<Button>(R.id.btn_export).setOnClickListener {
            val step = viewModel.brainInfo.value.step
            exportLauncher.launch("brain-$step.brain")
        }
        view.findViewById<Button>(R.id.btn_import).setOnClickListener {
            importLauncher.launch(arrayOf("*/*"))
        }
        view.findViewById<Button>(R.id.btn_new).setOnClickListener { onNewBrainClicked() }
        view.findViewById<Button>(R.id.btn_load_base).setOnClickListener {
            viewModel.loadBaseBrain()
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.brainInfo.collect { info ->
                tvBrainInfo.text = formatBrainInfo(info)
            }
        }
    }

    private fun onNewBrainClicked() {
        val info = viewModel.brainInfo.value
        if (TrainingManager.hasBrain && info.step > 0) {
            AlertDialog.Builder(requireContext())
                .setMessage(R.string.brain_new_confirm)
                .setPositiveButton(R.string.dialog_confirm) { _, _ ->
                    viewModel.createNewBrain(viewModel.currentConfig)
                }
                .setNegativeButton(R.string.dialog_cancel, null)
                .show()
        } else {
            viewModel.createNewBrain(viewModel.currentConfig)
        }
    }

    private fun formatBrainInfo(info: BrainInfo): String {
        if (!TrainingManager.hasBrain || info.paramCount == 0L) {
            return getString(R.string.brain_none)
        }
        val params = when {
            info.paramCount >= 1_000_000 ->
                "%.2fM".format(info.paramCount / 1_000_000.0)
            info.paramCount >= 1_000 ->
                "%.1fK".format(info.paramCount / 1_000.0)
            else -> info.paramCount.toString()
        }
        val loss = if (info.loss.isNaN()) "—" else "%.3f".format(info.loss)
        return getString(R.string.brain_step) + ": " + info.step + "\n" +
            getString(R.string.brain_params) + ": " + params + "\n" +
            getString(R.string.brain_vocab) + ": " + info.vocabSize + "\n" +
            getString(R.string.brain_arch) + ": " + info.archString() + "\n" +
            getString(R.string.brain_loss) + ": " + loss
    }
}
