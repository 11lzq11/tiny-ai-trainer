# TinyTrainer — 口袋里的极小 AI 训练器

[English below](#tinytrainer--a-tiny-ai-trainer-in-your-pocket)

一个能在手机上从零训练一个极小 Transformer 的 Android 应用：可调学习率、
batch、模型结构、语料、温度、Top-K；实时看 LOSS 曲线、Step、tokens/s；
随时对话；大脑可以导出成一个 `.brain` 文件、导入继续训。

> **定位**：玩具级小模型（约 2–4M 参数），学的是“像你的语料一样说话”，
> 不是懂知识的百科全书。训几千到几万步后，在语料的领域里能输出通顺的
> 句子；不要指望它回答事实问题。

## 怎么装

1. 从 GitHub Actions 的 Artifacts 下载 `tiny-trainer-debug-apk`（`app-debug.apk`），
   或从 Releases 下载。
2. 在手机上允许“安装未知来源的应用”，安装 APK 即可（未签名，可侧载）。
3. 需要 Android 8.0+（API 24），建议 4GB 以上内存。

## 怎么训

1. 打开「训练」页，粘贴文本或导入 `.txt` 文件作为语料（可多份，勾选启用）。
   语料越集中越好（同一个领域/风格），模型学的是那个领域说话的方式。
2. 调超参数（默认已填好能用的值）：
   学习率 3e-4、batch 32、序列长度 128、4 层、隐藏层 192、4 头、词表 3000。
3. 点「开始训练」。LOSS 曲线、Step、tokens/s 实时显示。
4. 锁屏、切后台都会继续训（前台服务 + 通知栏显示进度）。
   暂停/停止后会自动保存大脑到应用内部存储。
5. 去「对话」页输入开头，调温度/Top-K，点生成。训练中也可以生成，
   应用会先给当前权重复制一份快照再生成，不打断训练。
6. 「大脑」页可以导出 `.brain`（分享/保存）、导入别人的 `.brain`、
   新建大脑、加载内置底座大脑（底座文件后续提供）。

### 关于“底座大脑”

首版 APK 内置的 `base.brain` 是占位文件。正式的底座大脑（用几 MB 干净
中文语料轻度预训练）会在 B 完成前提供并替换。从底座微调几千步，比从零
训同等步数通顺得多。

## 性能预期（骁龙 6 系）

默认配置（约 2.4M 参数，batch 32，序列 128）每步约 70 GFLOP。
骁龙 6 系（4×A78 + 4×A55）配多线程 C 引擎的**估计**速度：

| 情况 | 速度 | 3000 步约需 |
| --- | --- | --- |
| 引擎矩阵乘优化较好 | 20–35 步/分钟 | 1.5–2.5 小时 |
| 引擎矩阵乘较朴素 | 8–20 步/分钟 | 2.5–6 小时 |

> 具体数字以真机实测为准；上面是按 FLOPs 的估算，不是实测。
> 校准参考：桌面冒烟测试（小配置，约 0.5M 参数，batch 4）在单核
> 虚拟机上跑到 1.08 步/秒，200 步后 loss 从 4.68 降到 0.31，生成已通顺；
> 手机是 8 核，默认配置每步计算量约为小配置的 36 倍。
> 小 batch（如 8）步数上得快，但每步看到的 token 少，学到的东西也少。

内存：fp32 训练约 150–250MB（含 Adam 状态），4GB 内存的手机够用。

## 已知限制

- 训练是前台服务，耗电明显；建议插着充电器训。
- 进程被系统杀死会丢失未保存的进度（停止训练时会自动保存；导出大脑可备份）。
- 词表是字符级：不在词表里的字符会被当成 `<unk>`。
- 暂停/停止在当前一批步数跑完后生效（最多几秒到几十秒）。
- 生成长度上限 500 token；温度 0.1–2.0，Top-K 0–200（0 = 不限制）。
- 底座大脑尚未提供前，从零训需要更多步数才能通顺。

## 项目结构

```
tiny-ai-trainer/
├── SPEC.md                  规格（模型 / .brain 格式 / JNI 接口 / 验收标准）
├── app/                     Android 应用（Kotlin + Gradle Kotlin DSL）
│   └── src/main/
│       ├── java/com/tinytrainer/   Activity / ViewModel / Service / UI
│       ├── cpp/CMakeLists.txt      编译 engine/ 的 C 引擎
│       └── assets/base.brain       底座大脑（首版为占位文件）
├── engine/                  C 训练引擎（手写 Transformer 前向/反向 + JNI）
└── .github/workflows/       CI：assembleDebug，产物为 debug APK
```

## 开发

```bash
./gradlew :app:assembleDebug
```

需要 Android SDK（API 34、build-tools 34.0.0、NDK 26.1.10909125、
CMake 3.22.1）与 JDK 17。CI 会自动装这些。

---

# TinyTrainer — A Tiny AI Trainer in Your Pocket

An Android app that trains a tiny Transformer from scratch on your phone:
tunable learning rate, batch, architecture, corpus, temperature, Top-K;
live LOSS curve, Step, tokens/s; chat any time; export/import the whole
brain as a single `.brain` file.

> **Positioning**: a toy-scale model (~2–4M params) that learns to *talk like
> your corpus*, not a knowledgeable encyclopedia. After a few thousand steps
> it produces coherent sentences inside the corpus domain; don't expect it
> to answer factual questions.

## Install

1. Download `tiny-trainer-debug-apk` (`app-debug.apk`) from GitHub Actions
   Artifacts, or from Releases.
2. Allow "install unknown apps" and install the APK (unsigned, sideloadable).
3. Android 8.0+ (API 24); 4GB+ RAM recommended.

## Train

1. On the Train tab, paste text or import `.txt` files as corpus (multiple
   files supported; tick to enable). Keep the corpus focused (one domain /
   style) — the model learns how that domain talks.
2. Tune hyper-parameters (sane defaults pre-filled): lr 3e-4, batch 32,
   sequence 128, 4 layers, hidden 192, 4 heads, vocab 3000.
3. Tap Start. LOSS curve, Step and tokens/s update live.
4. Training continues with the screen locked or the app in the background
   (foreground service + notification). Pause/Stop autosaves the brain.
5. On the Chat tab, type a prompt, set temperature / Top-K, and generate.
   Generation during training snapshots the weights first, so it never
   interrupts training.
6. On the Brain tab, export the `.brain` (share/save), import one, create a
   new brain, or load the bundled base brain (provided later).

### Base brain

The first APK ships a placeholder `base.brain`. The real base brain (lightly
pre-trained on a few MB of clean Chinese text) will replace it before the
first release. Fine-tuning from the base for a few thousand steps is much
more coherent than training the same steps from scratch.

## Performance (Snapdragon 6)

Default config (~2.4M params, batch 32, seq 128) costs ~70 GFLOP/step.
Estimated speed on a Snapdragon 6 (4×A78 + 4×A55) with the multi-threaded
C engine:

| Case | Speed | 3000 steps ≈ |
| --- | --- | --- |
| Well-optimised matmuls | 20–35 steps/min | 1.5–2.5 h |
| Naive matmuls | 8–20 steps/min | 2.5–6 h |

> These are FLOPs-based estimates, not measurements — real numbers to be
> confirmed on a real phone. Calibration: the desktop smoke test (small
> config, ~0.5M params, batch 4) ran at 1.08 steps/sec on a single-core VM,
> loss dropping from 4.68 to 0.31 in 200 steps with coherent generation;
> the phone has 8 cores while the default config costs ~36× the FLOPs per
> step of the small one. A small batch (e.g. 8) steps faster but sees
> fewer tokens per step.

Memory: ~150–250MB fp32 training (incl. Adam states); fine on a 4GB phone.

## Known limitations

- Training is a foreground service and drains battery; keep the phone plugged in.
- If the OS kills the process, unsaved progress is lost (Stop autosaves;
  export the brain to back up).
- Character-level vocab: characters outside the vocab become `<unk>`.
- Pause/Stop take effect after the current chunk finishes (seconds to tens
  of seconds).
- Generation capped at 500 tokens; temperature 0.1–2.0, Top-K 0–200
  (0 = unlimited).
- Without the base brain, training from scratch needs more steps to become coherent.

## Layout

```
tiny-ai-trainer/
├── SPEC.md                  Spec (model / .brain format / JNI / acceptance)
├── app/                     Android app (Kotlin + Gradle Kotlin DSL)
│   └── src/main/
│       ├── java/com/tinytrainer/   Activity / ViewModel / Service / UI
│       ├── cpp/CMakeLists.txt      Builds the C engine in engine/
│       └── assets/base.brain       Base brain (placeholder in v0.1.0)
├── engine/                  C training engine (hand-written Transformer + JNI)
└── .github/workflows/       CI: assembleDebug, debug APK artifact
```

## Development

```bash
./gradlew :app:assembleDebug
```

Requires Android SDK (API 34, build-tools 34.0.0, NDK 26.1.10909125,
CMake 3.22.1) and JDK 17. CI installs these automatically.
