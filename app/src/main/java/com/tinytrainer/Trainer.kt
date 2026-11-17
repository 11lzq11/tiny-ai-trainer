package com.tinytrainer

import org.json.JSONObject

/**
 * JNI bridge to the C training engine (libtinytrainer.so).
 *
 * The method signatures below MUST match SPEC section 3 exactly — the C side
 * implements them as JNI entry points named:
 *   Java_com_tinytrainer_Trainer_nativeCreate, ... (etc.)
 * Do not rename the class, the package, or any method: the JNI symbol names
 * are derived from them.
 *
 * Contract notes for the C engine (see also SPEC.md):
 * - nativeCreate(configJson): config JSON keys: n_layer, n_embd, n_head,
 *   block_size, vocab_size, dropout, lr, batch_size, warmup_steps,
 *   weight_decay, seed. Returns a native handle (long), 0 on failure.
 * - nativeLoadBrain(handle, path): loads a .brain file into the handle. The
 *   file's header JSON carries the model structure; the engine MUST resize
 *   the model to match the file (the config passed to nativeCreate is only
 *   the default for a brand-new brain).
 * - nativeSetCorpus(handle, text): UTF-8 text. First call builds the vocab
 *   (most frequent N chars + <pad>=0 <bos>=1 <eos>=2 <unk>=3); later calls
 *   only swap the corpus. Characters outside the vocab become <unk>.
 * - nativeTrainSteps(handle, steps): blocking; returns one loss per step.
 *   Call it from a background thread / coroutine only.
 * - nativeGenerate: samples with temperature / topK / seed from the live
 *   weights. For generation *during* training, callers snapshot first
 *   (see TrainingManager.generate) instead of calling this on the live
 *   handle.
 */
class Trainer {

    companion object {
        init {
            System.loadLibrary("tinytrainer")
        }

        /** Default hyper-parameters (SPEC section 1). */
        fun defaultConfig(): JSONObject = JSONObject().apply {
            put("n_layer", 4)
            put("n_embd", 192)
            put("n_head", 4)
            put("block_size", 128)
            put("vocab_size", 3000)
            put("dropout", 0.0)
            put("lr", 3e-4)
            put("batch_size", 32)
            put("warmup_steps", 100)
            put("weight_decay", 0.01)
            put("seed", 1337)
        }
    }

    // 返回 native handle（long），0 表示失败
    external fun nativeCreate(configJson: String): Long
    external fun nativeFree(handle: Long)
    external fun nativeLoadBrain(handle: Long, path: String): Boolean
    external fun nativeSaveBrain(handle: Long, path: String): Boolean
    external fun nativeSetCorpus(handle: Long, text: String): Boolean   // 分词并建词表（首次）/ 仅换语料
    external fun nativeTrainSteps(handle: Long, steps: Int): FloatArray  // 返回每步 loss
    external fun nativeGenerate(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topK: Int, seed: Long): String
    external fun nativeGetStep(handle: Long): Long
    external fun nativeSetLr(handle: Long, lr: Float)
    external fun nativeSetBatchSize(handle: Long, batch: Int)
    external fun nativeVocabSize(handle: Long): Int
    external fun nativeParamCount(handle: Long): Long
}
