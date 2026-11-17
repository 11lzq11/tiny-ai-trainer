/*
 * tinygpt.h — 极小 Transformer 训练引擎（纯 C，无 ML 框架依赖）
 *
 * 解码器 Transformer（GPT 式）：前向 + 反向 + AdamW + 交叉熵。
 * 字符级 tokenizer，词表由语料统计生成；大脑文件为单个 .brain 二进制。
 *
 * 设计要点（与 SPEC.md 一致）：
 *  - 输出头与 token embedding 共享权重（weight tying），省约 vocab*n_embd 参数
 *  - dropout 恒为 0（不实现）
 *  - 词表：<pad>=0, <bos>=1, <eos>=2, <unk>=3，其后为按频率降序的字符
 *  - 大脑文件：magic "TBRN" | version u32 | header_len u32 | header JSON |
 *    权重 fp32 LE | AdamW m fp32 LE | AdamW v fp32 LE（后两段用于“加载后继续
 *    训练结果一致”；若文件只有权重段，m/v 按 0 初始化，仍可加载）。
 *    header 里同时保存学习率调度与优化器超参数（warmup/decay/wd/裁剪/
 *    beta/eps）与训练 RNG 状态，否则加载后续训轨迹会与保存前不一致。
 *  - 多线程：矩阵乘与逐元素大循环用 OpenMP 并行；不用 -fopenmp 编译时
 *    pragma 会被忽略，退化为串行（仍可正确运行）
 *  - 仅用标准 C99 + pthread（JNI 互斥）+ 可选 OpenMP；无 glibc 专属 API，
 *    可在 Android NDK 下编译。
 */

#ifndef TINYGPT_H
#define TINYGPT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- 配置 ---------------- */

typedef struct {
    int n_layer;       /* 层数，默认 4（1–6） */
    int n_embd;        /* 隐藏层宽，默认 192（64–384） */
    int n_head;        /* 注意力头数，默认 4（2–8，需整除 n_embd） */
    int block_size;    /* 序列长度，默认 128（64–256） */
    int vocab_size;    /* 词表上限，默认 3000（500–8000）；实际词表可能更小 */
    float lr;          /* 峰值学习率，默认 3e-4 */
    int batch_size;    /* 默认 8（骁龙 6 上小 batch 步数更快，见 README） */
    int warmup_steps;  /* warmup 步数，默认 100 */
    int decay_steps;   /* 余弦衰减到 10% 所用的步数，默认 10000；之后保持 10% */
    float weight_decay;/* 默认 0.01（仅对矩阵参数） */
    float grad_clip;   /* 梯度裁剪（全局 L2 范数），默认 1.0；<=0 表示不裁剪 */
    float beta1;       /* 默认 0.9 */
    float beta2;       /* 默认 0.95 */
    float eps;         /* 默认 1e-8 */
    uint64_t seed;     /* 随机种子，默认 1337 */
} TGConfig;

/* 默认配置 */
void tg_config_default(TGConfig *cfg);

/* 校验配置是否合法（n_head 整除 n_embd 等），合法返回 0 */
int tg_config_validate(const TGConfig *cfg);

/* ---------------- 模型 ---------------- */

typedef struct TGModel TGModel;

/* 按配置创建模型并随机初始化（N(0, 0.02)，偏置 0，LN 权重 1）。
 * 此时词表为空，需先 tg_set_corpus 才能训练/生成。失败返回 NULL。 */
TGModel *tg_create(const TGConfig *cfg);

/* 释放模型（含词表、参数、梯度、Adam 状态、激活缓存、语料） */
void tg_free(TGModel *m);

/* 参数量（按当前结构；含未使用的 embedding 行，若词表小于上限） */
long tg_param_count(const TGModel *m);

/* 当前词表大小（set_corpus 之后才有意义，未设语料返回 0） */
int tg_vocab_size(const TGModel *m);

/* 当前训练步数 / 最近一次 loss */
long tg_get_step(const TGModel *m);
float tg_get_loss(const TGModel *m);

/* 设置峰值学习率 / batch 大小（立即生效） */
void tg_set_lr(TGModel *m, float lr);
void tg_set_batch_size(TGModel *m, int batch);

/* ---------------- 语料与词表 ---------------- */

/* 设置语料（UTF-8 文本）。
 * 首次调用：统计字符频率，建词表（取最高频的 vocab_size-4 个字符），
 *           并分词存入模型；同时把 embedding 的有效行数定为实际词表大小。
 * 之后调用：只用已有词表重新分词换语料（新字符 → <unk>）。
 * 成功返回 0；文本为空或无有效字符返回 -1。 */
int tg_set_corpus(TGModel *m, const char *utf8_text);

/* 语料 token 总数（set_corpus 之后） */
long tg_corpus_tokens(const TGModel *m);

/* 取词表字符串（UTF-8，调用者不拥有；index 越界返回 NULL） */
const char *tg_vocab_str(const TGModel *m, int index);

/* 编码一段 UTF-8 文本为 token（输出到 out，容量 cap；返回 token 数；
 * 超出 cap 的部分截断；不在词表的字符 → <unk>=3） */
int tg_encode(const TGModel *m, const char *utf8_text, int *out, int cap);

/* 解码 token 序列为 UTF-8 文本（输出到 out，容量 cap；返回写入字节数，
 * 不含结尾 \0；特殊 token <pad>/<bos>/<eos> 跳过，<unk> 输出 U+FFFD） */
int tg_decode(const TGModel *m, const int *tokens, int n, char *out, int cap);

/* ---------------- 训练与生成 ---------------- */

/* 训练 steps 步，每步返回平均交叉熵 loss（写入 losses[steps]）。
 * 要求已 set_corpus。成功返回 0，失败返回 -1。 */
int tg_train_steps(TGModel *m, int steps, float *losses);

/* 生成：prompt 为 UTF-8 文本；max_tokens 最多生成 token 数；
 * temperature 温度（<=0 为贪心）；top_k（<=0 为不限制）；
 * seed 为采样种子。输出写入 out（UTF-8，容量 cap），返回写入字节数。
 * 要求已 set_corpus。失败返回 -1。 */
int tg_generate(TGModel *m, const char *prompt, int max_tokens,
                float temperature, int top_k, uint64_t seed,
                char *out, int cap);

/* ---------------- 大脑文件 ---------------- */

/* 保存大脑到 path（单个 .brain 文件）。成功返回 0。 */
int tg_save_brain(const TGModel *m, const char *path);

/* 从 path 加载大脑，创建新模型并返回；失败返回 NULL。
 * 加载后 step/loss/lr/batch/seed/Adam 状态/RNG 状态全部恢复，
 * 继续训练与生成的结果与保存前一致。 */
TGModel *tg_load_brain(const char *path);

/* 从 JSON 配置字符串创建模型（供 JNI nativeCreate 用）。
 * 支持的字段：n_layer, n_embd, n_head, block_size, vocab_size, lr,
 * batch_size, warmup_steps, decay_steps, weight_decay, grad_clip,
 * beta1, beta2, eps, seed。缺失字段用默认值；非法返回 NULL。 */
TGModel *tg_create_from_json(const char *json);

#ifdef __cplusplus
}
#endif

#endif /* TINYGPT_H */
