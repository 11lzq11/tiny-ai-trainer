package com.tinytrainer.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.tinytrainer.CorpusItem
import com.tinytrainer.R

/**
 * Corpus list rows: enable checkbox + name + size + delete button.
 * The checkbox only reports the tap; the checked state comes from the
 * ViewModel's flow (single source of truth, no stale UI state).
 */
class CorpusAdapter(
    private val onToggle: (CorpusItem, Boolean) -> Unit,
    private val onDelete: (CorpusItem) -> Unit
) : ListAdapter<CorpusItem, CorpusAdapter.ViewHolder>(DIFF) {

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<CorpusItem>() {
            override fun areItemsTheSame(oldItem: CorpusItem, newItem: CorpusItem): Boolean =
                oldItem.name == newItem.name

            override fun areContentsTheSame(oldItem: CorpusItem, newItem: CorpusItem): Boolean =
                oldItem == newItem
        }
    }

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val checkBox: CheckBox = view.findViewById(R.id.cb_active)
        val nameView: TextView = view.findViewById(R.id.tv_name)
        val sizeView: TextView = view.findViewById(R.id.tv_size)
        val deleteButton: Button = view.findViewById(R.id.btn_delete)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_corpus, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = getItem(position)
        holder.nameView.text = item.name
        holder.sizeView.text = formatSize(item.sizeBytes)
        // Detach the listener before setting the state, or the programmatic
        // setChecked would fire a toggle event.
        holder.checkBox.setOnCheckedChangeListener(null)
        holder.checkBox.isChecked = item.active
        holder.checkBox.setOnCheckedChangeListener { _, checked ->
            onToggle(item, checked)
        }
        holder.deleteButton.setOnClickListener { onDelete(item) }
    }

    /** File size in bytes (shown as B/KB/MB — byte count, not char count). */
    private fun formatSize(sizeBytes: Long): String = when {
        sizeBytes < 1024 -> "$sizeBytes 字节"
        sizeBytes < 1024 * 1024 -> "%.1f KB".format(sizeBytes / 1024.0)
        else -> "%.2f MB".format(sizeBytes / 1024.0 / 1024.0)
    }
}
