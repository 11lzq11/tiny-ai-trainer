package com.tinytrainer

import org.json.JSONObject

/** Hyper-parameters the user can edit on the Train page (SPEC section 1). */
data class TrainConfig(
    val lr: Float = 3e-4f,
    val batchSize: Int = 32,
    val blockSize: Int = 128,
    val nLayer: Int = 4,
    val nEmbd: Int = 192,
    val nHead: Int = 4,
    val vocabSize: Int = 3000,
    val warmupSteps: Int = 100,
    val weightDecay: Float = 0.01f,
    val seed: Long = 1337L
) {
    /** SPEC ranges: vocab 500–8000, block 64–256, layers 1–6, embd 64–384,
     *  heads 2–8 and must divide nEmbd, lr 1e-5–1e-2, batch 1–128. */
    fun isValid(): Boolean {
        if (lr !in 1e-5f..1e-2f) return false
        if (batchSize !in 1..128) return false
        if (blockSize !in 64..256) return false
        if (nLayer !in 1..6) return false
        if (nEmbd !in 64..384) return false
        if (nHead !in 2..8) return false
        if (nEmbd % nHead != 0) return false
        if (vocabSize !in 500..8000) return false
        if (warmupSteps !in 0..5000) return false
        if (weightDecay !in 0f..0.5f) return false
        return true
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("n_layer", nLayer)
        put("n_embd", nEmbd)
        put("n_head", nHead)
        put("block_size", blockSize)
        put("vocab_size", vocabSize)
        put("dropout", 0.0)
        put("lr", lr.toDouble())
        put("batch_size", batchSize)
        put("warmup_steps", warmupSteps)
        put("weight_decay", weightDecay.toDouble())
        put("seed", seed)
    }
}

/** Info shown on the Brain page. */
data class BrainInfo(
    val step: Long = 0,
    val paramCount: Long = 0,
    val vocabSize: Int = 0,
    val nLayer: Int = 0,
    val nEmbd: Int = 0,
    val nHead: Int = 0,
    val blockSize: Int = 0,
    val loss: Float = Float.NaN
) {
    fun archString(): String =
        if (nLayer == 0) "—"
        else "${nLayer}层 · 隐藏层 $nEmbd · ${nHead}头 · 序列 $blockSize"
}

/** One corpus file stored under filesDir/corpora/. */
data class CorpusItem(
    val name: String,
    val sizeBytes: Long,
    val active: Boolean
)

/** Live training numbers shown on the Train page. */
data class TrainStats(
    val step: Long = 0,
    val loss: Float = Float.NaN,
    val tokensPerSec: Double = 0.0,
    val elapsedMs: Long = 0
)

enum class TrainState { IDLE, RUNNING, PAUSED }
