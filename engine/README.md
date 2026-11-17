# TinyTrainer C 训练引擎

纯 C 手写的极小 Transformer（GPT 式）训练引擎，不依赖任何 ML 框架。
对应 SPEC.md 第 1、2、3、6 节。

## 文件

| 文件 | 说明 |
|---|---|
| `tinygpt.h` | 公共 API（配置/模型/语料/训练/生成/大脑文件） |
| `tinygpt.c` | 全部实现：前向/反向、AdamW、字符级 tokenizer、.brain 读写 |
| `jni.c` | JNI 桥接（Kotlin `com.tinytrainer.Trainer` ↔ C） |
| `CMakeLists.txt` | NDK 与桌面构建共用 |
| `tests/smoke.c` | 桌面冒烟测试：训 200 步看 loss 下降 + 生成 + 保存/加载往返 |
| `tests/corpus.txt` | 冒烟测试用中文语料（约 4KB，重复句式） |
| `tests/jni_stub.h` | 仅供桌面语法检查的最小 JNI 存根（非真实 jni.h） |

## 模型

- 解码器 Transformer（GPT 式），前向 + 反向手写；交叉熵损失；AdamW。
- 默认：`n_layer=4, n_embd=192, n_head=4, block_size=128, vocab_size=3000`，
  约 2.38M 参数（vocab=3000 时）；输出头与 token embedding 共享权重。
- 结构全部可配：层数 1–6、隐藏层 64–384、头数 2–8（需整除隐藏层）、
  序列 64–256、词表 500–8000；dropout 恒为 0（不实现）。
- 学习率：warmup（默认 100 步）+ 余弦衰减到 10%，之后保持 10%；
  梯度裁剪 1.0；AdamW（β1=0.9，β2=0.95，eps=1e-8，wd=0.01，仅矩阵衰减）。

## 词表

字符级。首次 `tg_set_corpus` 时统计语料字符频率，取最高频的
`vocab_size-4` 个字符；特殊 token `<pad>=0, <bos>=1, <eos>=2, <unk>=3`。
之后换语料只重新分词，不重建词表（新字符 → `<unk>`）。
若语料去重后字符少于上限，实际词表会小于上限（embedding 多余行闲置，
不影响正确性；大脑文件里如实保存）。

## 大脑文件（`.brain`）

单个二进制文件：`magic "TBRN" | version u32=1 | header_len u32 | header JSON |
权重 fp32 LE | AdamW m fp32 LE | AdamW v fp32 LE`。

header JSON：`{model:{n_layer,n_embd,n_head,block_size,vocab_size,vocab_cap},
vocab:["<pad>",...], train:{step,loss,lr,batch_size,seed,rng_state}}`。
其中 `vocab_cap` 为扩展字段（embedding 分配行数；缺失时视为 =vocab_size）。
保存 AdamW 的 m/v 与训练 RNG 状态，因此加载后继续训练与生成的结果
与保存前逐 bit 一致；若文件只有权重段（外部工具生成），m/v 按 0 初始化，
仍可加载（继续训练的轨迹会略有不同）。

**注意：大脑文件不存语料。** 加载后如需继续训练，必须先重新设置同一份
（或新的）语料（`tg_set_corpus` / `nativeSetCorpus`），否则训练会拒绝
（无 token 数据）。生成不需要语料，加载后可直接生成。

## 多线程

矩阵乘与大循环逐元素操作用 OpenMP 并行（`#pragma omp parallel for`，
带规模阈值，小循环不并行以免开销）。不用 `-fopenmp` 编译时 pragma 被
忽略，退化为串行，仍可正确运行。NDK 下需 `find_package(OpenMP)` 链接
`OpenMP::OpenMP_C`（NDK r21+ 自带 libomp）。

## JNI

- Kotlin 侧类必须是 `com.tinytrainer.Trainer`，否则须按实际包名重命名
  `jni.c` 里的函数（`Java_<包>_<类>_<方法>`）。
- 每个 handle 带一把互斥锁，训练/生成/保存/加载互斥。`nativeTrainSteps`
  是阻塞调用，Kotlin 层应在后台线程以小块（10–50 步）循环调用，
  以便生成在训练块之间插空执行、UI 保持响应。
- 字符串用 `GetStringUTFChars`（标准 UTF-8；BMP 内字符无损；增补平面
  字符如 emoji 会被当作 `<unk>`，已知限制）。

## 桌面构建与测试

```sh
# 冒烟测试（小配置，200 步，约 1 分钟）
gcc -O2 -fopenmp -Wall -Wextra -std=c99 -o /tmp/smoke tests/smoke.c tinygpt.c -lm
./smoke --config small --steps 200 --corpus tests/corpus.txt

# 默认结构验证（2.38M 参数，batch 8，50 步）
./smoke --config default --steps 50 --corpus tests/corpus.txt

# JNI 语法检查（对存根头文件；真实构建用 NDK 的 jni.h）
cp tests/jni_stub.h /tmp/jni.h && gcc -O2 -Wall -Wextra -std=c99 -I/tmp -c jni.c -o /tmp/jni.o

# CMake（桌面）
cmake -S . -B /tmp/build && cmake --build /tmp/build
```

## 已知限制

1. JNI 未在真实 NDK 下编译验证（本机无 NDK/JDK）；只做了存根语法检查。
   NDK 构建由应用侧 CI 验证。
2. 训练与生成在 JNI 层用互斥锁串行（生成会等当前训练块跑完；块小则无感）。
3. 语料/生成中的增补平面字符（emoji 等）会被当作 `<unk>`。
4. 词表小于上限时，embedding 多余行闲置并随大脑文件保存（最多约 2MB）。
5. 注意力反向用固定 1024 的栈数组存中间量（block_size 上限即 1024）。
6. 训练为 fp32 全精度；未做量化/混合精度（骁龙 6 上够用是设计目标，
   但步/秒仍需在真机上实测）。
