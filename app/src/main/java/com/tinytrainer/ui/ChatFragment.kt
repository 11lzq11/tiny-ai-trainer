package com.tinytrainer.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.tinytrainer.MainViewModel
import com.tinytrainer.R
import com.tinytrainer.TrainingManager
import kotlinx.coroutines.launch

/**
 * Chat page: prompt + temperature / Top-K sliders + generate.
 * Works while training is running (TrainingManager snapshots the weights
 * first, so generation never blocks the training loop).
 */
class ChatFragment : Fragment() {

    private val viewModel: MainViewModel by activityViewModels()

    private lateinit var etPrompt: EditText
    private lateinit var sbTemperature: SeekBar
    private lateinit var sbTopK: SeekBar
    private lateinit var tvTempLabel: TextView
    private lateinit var tvTopKLabel: TextView
    private lateinit var etMaxTokens: EditText
    private lateinit var btnGenerate: Button
    private lateinit var tvOutput: TextView

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.fragment_chat, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        etPrompt = view.findViewById(R.id.et_prompt)
        sbTemperature = view.findViewById(R.id.sb_temperature)
        sbTopK = view.findViewById(R.id.sb_topk)
        tvTempLabel = view.findViewById(R.id.tv_temp_label)
        tvTopKLabel = view.findViewById(R.id.tv_topk_label)
        etMaxTokens = view.findViewById(R.id.et_max_tokens)
        btnGenerate = view.findViewById(R.id.btn_generate)
        tvOutput = view.findViewById(R.id.tv_output)

        updateTempLabel()
        updateTopKLabel()
        sbTemperature.setOnSeekBarChangeListener(simpleListener { updateTempLabel() })
        sbTopK.setOnSeekBarChangeListener(simpleListener { updateTopKLabel() })

        btnGenerate.setOnClickListener {
            val prompt = etPrompt.text.toString()
            val maxTokens = etMaxTokens.text.toString().toIntOrNull() ?: 100
            val temperature = 0.1f + sbTemperature.progress / 100f
            val topK = sbTopK.progress
            viewModel.generate(prompt, maxTokens.coerceIn(1, 500), temperature, topK)
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.generating.collect { generating ->
                btnGenerate.isEnabled = !generating
                btnGenerate.text = if (generating) {
                    getString(R.string.chat_generating)
                } else {
                    getString(R.string.chat_generate)
                }
            }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.lastGeneration.collect { text ->
                if (text.isNotEmpty()) tvOutput.text = text
            }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.brainInfo.collect { info ->
                if (!TrainingManager.hasBrain) {
                    tvOutput.text = getString(R.string.chat_no_brain)
                }
            }
        }
    }

    private fun updateTempLabel() {
        val temp = 0.1f + sbTemperature.progress / 100f
        tvTempLabel.text = "${getString(R.string.chat_temperature)}：%.2f".format(temp)
    }

    private fun updateTopKLabel() {
        tvTopKLabel.text = "${getString(R.string.chat_topk)}：${sbTopK.progress}"
    }

    private fun simpleListener(onChange: () -> Unit) =
        object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                onChange()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        }
}
