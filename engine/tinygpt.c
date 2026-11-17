/*
 * tinygpt.c — 极小 Transformer 训练引擎（纯 C，无 ML 框架依赖）
 * 手写实现：解码器 Transformer 前向/反向、AdamW、交叉熵、字符级 tokenizer、
 * .brain 文件读写。仅用标准 C99 + pthread + 可选 OpenMP。
 */
#define _POSIX_C_SOURCE 200809L
#include "tinygpt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------- RNG：splitmix64（训练/初始化用） ---------------- */
static uint64_t rng_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static float rng_uniform(uint64_t *state) { /* [0,1) */
    return (float)((rng_next(state) >> 11) * (1.0 / 9007199254740992.0));
}
static float rng_normal(uint64_t *state) { /* Box-Muller，标准正态 */
    float u1 = rng_uniform(state);
    float u2 = rng_uniform(state);
    if (u1 < 1e-12f) u1 = 1e-12f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---------------- UTF-8 编解码 ---------------- */
/* 解码一个 UTF-8 码点：返回码点值，*adv 为消耗字节数；非法返回 0xFFFD，adv=1 */
static uint32_t utf8_decode(const unsigned char *s, int avail, int *adv) {
    unsigned char c = s[0];
    if (c < 0x80) { *adv = 1; return c; }
    if ((c & 0xE0) == 0xC0 && avail >= 2) {
        uint32_t cp = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        if (cp >= 0x80) { *adv = 2; return cp; }
    } else if ((c & 0xF0) == 0xE0 && avail >= 3) {
        uint32_t cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) { *adv = 3; return cp; }
    } else if ((c & 0xF8) == 0xF0 && avail >= 4) {
        uint32_t cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) |
                      ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (cp >= 0x10000 && cp <= 0x10FFFF) { *adv = 4; return cp; }
    }
    *adv = 1; return 0xFFFD; /* 非法字节 → 替换字符 */
}
/* 编码一个码点为 UTF-8，返回写入字节数 */
static int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

/* ---------------- 极小 JSON 工具（只够读配置/大脑头） ---------------- */
/* 在 JSON 文本里找 "key" 并解析其后的整数；找不到返回 def */
static long json_get_long(const char *json, const char *key, long def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    return strtol(p + 1, NULL, 10);
}
static double json_get_double(const char *json, const char *key, double def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    return strtod(p + 1, NULL);
}
/* JSON 字符串转义后追加到动态 buffer（用于写 header 里的 vocab） */
typedef struct { char *buf; size_t len, cap; } StrBuf;
static void sb_putc(StrBuf *sb, char c) {
    if (sb->len + 2 > sb->cap) {
        sb->cap = sb->cap ? sb->cap * 2 : 256;
        sb->buf = (char *)realloc(sb->buf, sb->cap);
    }
    sb->buf[sb->len++] = c; sb->buf[sb->len] = '\0';
}
static void sb_puts(StrBuf *sb, const char *s) { while (*s) sb_putc(sb, *s++); }
static void sb_put_json_string(StrBuf *sb, const char *s) {
    sb_putc(sb, '"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"') sb_puts(sb, "\\\"");
        else if (*p == '\\') sb_puts(sb, "\\\\");
        else if (*p < 0x20) { char tmp[8]; snprintf(tmp, sizeof(tmp), "\\u%04x", *p); sb_puts(sb, tmp); }
        else sb_putc(sb, (char)*p);
    }
    sb_putc(sb, '"');
}
/* 解析 JSON 字符串（带转义，含 \uXXXX）为 UTF-8；src 指向开头的 '"'；
 * 返回解码后的 malloc 字符串，*end 指向闭合引号之后；失败返回 NULL */
static char *json_parse_string(const char *src, const char **end) {
    if (*src != '"') return NULL;
    src++;
    size_t cap = 16, len = 0;
    char *out = (char *)malloc(cap);
    while (*src && *src != '"') {
        uint32_t cp;
        if (*src == '\\') {
            src++;
            switch (*src) {
                case 'n': cp = '\n'; src++; break;
                case 't': cp = '\t'; src++; break;
                case 'r': cp = '\r'; src++; break;
                case 'b': cp = '\b'; src++; break;
                case 'f': cp = '\f'; src++; break;
                case '/': cp = '/'; src++; break;
                case '\\': cp = '\\'; src++; break;
                case '"': cp = '"'; src++; break;
                case 'u': {
                    unsigned v = 0;
                    for (int i = 0; i < 4 && src[1 + i]; i++) {
                        char h = src[1 + i]; v <<= 4;
                        if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                    }
                    src += 5; cp = v; break;
                }
                default: cp = (unsigned char)*src; src++; break;
            }
        } else {
            int adv = 0; cp = utf8_decode((const unsigned char *)src, 4, &adv); src += adv;
        }
        char tmp[4]; int n = utf8_encode(cp, tmp);
        if (len + (size_t)n + 1 > cap) { cap = (len + n + 1) * 2; out = (char *)realloc(out, cap); }
        memcpy(out + len, tmp, (size_t)n); len += (size_t)n;
    }
    if (*src != '"') { free(out); return NULL; }
    out[len] = '\0';
    if (end) *end = src + 1;
    return out;
}

/* ---------------- 模型结构 ---------------- */

struct TGModel {
    TGConfig cfg;
    /* 词表 */
    char **vocab;      /* vocab_size 条 UTF-8 字符串（拥有） */
    int vocab_size;    /* 有效词表大小（set_corpus 后） */
    int vocab_cap;     /* embedding 分配的行数（= cfg.vocab_size） */
    uint32_t *vocab_cp;/* 每条词表对应的码点（特殊 token 为 0） */
    /* 字符 → token 的哈希表（编码用） */
    uint32_t *h_keys; int *h_vals; int h_cap; /* h_keys[i]=码点+1（0=空） */
    /* 参数 / 梯度 / AdamW 状态 */
    float *params, *grads, *m, *v;
    long n_params;
    /* 参数偏移（相对 params 起点） */
    long off_tok, off_pos, off_lnf_w, off_lnf_b;
    long *layer_base;  /* [n_layer] 每层起点 */
    /* 每层内字段偏移（相对层起点） */
    long o_ln1w, o_ln1b, o_qkvw, o_qkvb, o_attw, o_attb;
    long o_ln2w, o_ln2b, o_fcw, o_fcb, o_projw, o_projb;
    long layer_size;
    /* 激活缓存（按最大 batch/block_size 分配） */
    float *acts;       /* 全部前向中间量 */
    float *scratch;    /* 反向临时梯度 */
    /* 语料 */
    int *tokens; long n_tokens;
    /* 训练状态 */
    long step;
    float last_loss;
    uint64_t rng_state; /* 训练采样 RNG */
    int vocab_built;
};

/* 每层参数量 = 12*D^2 + 13*D */
static long tg_layer_param_size(const TGConfig *c) {
    long D = c->n_embd;
    return 12 * D * D + 13 * D;
}
static long tg_total_params(const TGConfig *c) {
    long D = c->n_embd, V = c->vocab_size, T = c->block_size;
    return V * D + T * D + (long)c->n_layer * tg_layer_param_size(c) + 2 * D;
}

/* 计算各偏移并填入模型（在分配 params 之前调用） */
static void tg_compute_offsets(TGModel *m) {
    const TGConfig *c = &m->cfg;
    long D = c->n_embd;
    m->off_tok = 0;
    m->off_pos = (long)c->vocab_size * D;
    long lb = m->off_pos + (long)c->block_size * D;
    m->layer_base = (long *)malloc(sizeof(long) * (size_t)c->n_layer);
    for (int l = 0; l < c->n_layer; l++)
        m->layer_base[l] = lb + (long)l * tg_layer_param_size(c);
    m->layer_size = tg_layer_param_size(c);
    long o = 0;
    m->o_ln1w = o; o += D;
    m->o_ln1b = o; o += D;
    m->o_qkvw = o; o += 3 * D * D;
    m->o_qkvb = o; o += 3 * D;
    m->o_attw = o; o += D * D;
    m->o_attb = o; o += D;
    m->o_ln2w = o; o += D;
    m->o_ln2b = o; o += D;
    m->o_fcw  = o; o += 4 * D * D;
    m->o_fcb  = o; o += 4 * D;
    m->o_projw = o; o += 4 * D * D;
    m->o_projb = o; o += D;
    long lnf = lb + (long)c->n_layer * tg_layer_param_size(c);
    m->off_lnf_w = lnf;
    m->off_lnf_b = lnf + D;
    m->n_params = tg_total_params(c);
}

/* 激活缓存大小（float 数）：按最大 batch 与 block_size */
static size_t tg_acts_size(const TGModel *m) {
    const TGConfig *c = &m->cfg;
    long B = c->batch_size, T = c->block_size, D = c->n_embd, H = c->n_head, V = m->vocab_size > 0 ? m->vocab_size : c->vocab_size;
    long L = c->n_layer;
    long per_layer = 0;
    per_layer += B * T * D;      /* lin（每层输入，复用流） */
    per_layer += B * T * D;      /* ln1 */
    per_layer += B * T * 3 * D;  /* qkv */
    per_layer += B * H * T * T;  /* att probs */
    per_layer += B * T * D;      /* attout */
    per_layer += B * T * D;      /* x2 */
    per_layer += B * T * D;      /* ln2 */
    per_layer += B * T * 4 * D;  /* fcpre */
    per_layer += B * T * 4 * D;  /* fc */
    per_layer += 4 * B * T;      /* ln1/ln2 mean+rstd */
    long total = L * per_layer;
    total += B * T * D;          /* 最终流（lnf 输入） */
    total += B * T * D;          /* lnf 输出 */
    total += 2 * B * T;          /* lnf mean+rstd */
    total += B * T * V;          /* logits */
    total += B * T * D;          /* emb（lin[0] 的源，单独算一份） */
    return (size_t)total;
}
/* 反向临时缓存大小（float 数） */
static size_t tg_scratch_size(const TGModel *m) {
    const TGConfig *c = &m->cfg;
    long B = c->batch_size, T = c->block_size, D = c->n_embd, H = c->n_head, V = m->vocab_size > 0 ? m->vocab_size : c->vocab_size;
    size_t total = 0;
    total += (size_t)(B * T * V);     /* dlogits */
    total += (size_t)(B * T * D);     /* dres（残差流梯度） */
    total += (size_t)(B * T * 3 * D); /* gqkv */
    total += (size_t)(B * H * T * T); /* gatt */
    total += (size_t)(B * T * 4 * D); /* gfc */
    total += (size_t)(B * T * D) * 3; /* g1,g2,g3 */
    return total;
}

/* ---------------- 配置 ---------------- */
void tg_config_default(TGConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layer = 4; cfg->n_embd = 192; cfg->n_head = 4;
    cfg->block_size = 128; cfg->vocab_size = 3000;
    cfg->lr = 3e-4f; cfg->batch_size = 8;
    cfg->warmup_steps = 100; cfg->decay_steps = 10000;
    cfg->weight_decay = 0.01f; cfg->grad_clip = 1.0f;
    cfg->beta1 = 0.9f; cfg->beta2 = 0.95f; cfg->eps = 1e-8f;
    cfg->seed = 1337;
}
int tg_config_validate(const TGConfig *cfg) {
    if (cfg->n_layer < 1 || cfg->n_layer > 12) return -1;
    if (cfg->n_embd < 16 || cfg->n_embd > 1024) return -1;
    if (cfg->n_head < 1 || cfg->n_head > 16) return -1;
    if (cfg->n_embd % cfg->n_head != 0) return -1;
    if (cfg->block_size < 8 || cfg->block_size > 1024) return -1;
    if (cfg->vocab_size < 16 || cfg->vocab_size > 65536) return -1;
    if (cfg->batch_size < 1 || cfg->batch_size > 256) return -1;
    if (cfg->lr <= 0 || cfg->lr > 0.1f) return -1;
    if (cfg->warmup_steps < 0 || cfg->decay_steps < 1) return -1;
    return 0;
}

/* ---------------- 创建 / 释放 ---------------- */
TGModel *tg_create(const TGConfig *cfg) {
    if (!cfg || tg_config_validate(cfg) != 0) return NULL;
    TGModel *m = (TGModel *)calloc(1, sizeof(TGModel));
    if (!m) return NULL;
    m->cfg = *cfg;
    m->vocab_cap = cfg->vocab_size;
    m->rng_state = cfg->seed ? cfg->seed : 0x9E3779B97F4A7C15ULL;
    tg_compute_offsets(m);
    m->params = (float *)malloc(sizeof(float) * (size_t)m->n_params);
    m->grads  = (float *)calloc((size_t)m->n_params, sizeof(float));
    m->m      = (float *)calloc((size_t)m->n_params, sizeof(float));
    m->v      = (float *)calloc((size_t)m->n_params, sizeof(float));
    if (!m->params || !m->grads || !m->m || !m->v) { tg_free(m); return NULL; }
    /* 随机初始化：权重 N(0,0.02)，偏置 0，LN 权重 1 */
    uint64_t rs = m->rng_state ^ 0xA0761D6478BD642FULL;
    long D = cfg->n_embd;
    for (long i = 0; i < m->n_params; i++)
        m->params[i] = 0.02f * rng_normal(&rs);
    /* 偏置与 LN 归零/置 1：按偏移逐段处理 */
    for (int l = 0; l < cfg->n_layer; l++) {
        float *base = m->params + m->layer_base[l];
        for (long i = 0; i < D; i++) { base[m->o_ln1w + i] = 1.0f; base[m->o_ln1b + i] = 0.0f; }
        for (long i = 0; i < 3 * D; i++) base[m->o_qkvb + i] = 0.0f;
        for (long i = 0; i < D; i++) base[m->o_attb + i] = 0.0f;
        for (long i = 0; i < D; i++) { base[m->o_ln2w + i] = 1.0f; base[m->o_ln2b + i] = 0.0f; }
        for (long i = 0; i < 4 * D; i++) base[m->o_fcb + i] = 0.0f;
        for (long i = 0; i < D; i++) base[m->o_projb + i] = 0.0f;
    }
    for (long i = 0; i < D; i++) { m->params[m->off_lnf_w + i] = 1.0f; m->params[m->off_lnf_b + i] = 0.0f; }
    /* 激活缓存按最大 batch 分配（词表未知时按上限估算；set_corpus 后若
     * 词表更小，logits 部分用不满，无妨） */
    m->vocab_size = cfg->vocab_size; /* 临时值，set_corpus 后修正 */
    m->acts = (float *)malloc(sizeof(float) * tg_acts_size(m));
    m->scratch = (float *)malloc(sizeof(float) * tg_scratch_size(m));
    if (!m->acts || !m->scratch) { tg_free(m); return NULL; }
    m->vocab_size = 0; /* 尚未建词表 */
    return m;
}

void tg_free(TGModel *m) {
    if (!m) return;
    if (m->vocab) {
        for (int i = 0; i < m->vocab_size; i++) free(m->vocab[i]);
        free(m->vocab);
    }
    free(m->vocab_cp); free(m->h_keys); free(m->h_vals);
    free(m->params); free(m->grads); free(m->m); free(m->v);
    free(m->layer_base);
    free(m->acts); free(m->scratch);
    free(m->tokens);
    free(m);
}

long tg_param_count(const TGModel *m) { return m ? m->n_params : 0; }
int tg_vocab_size(const TGModel *m) { return m ? m->vocab_size : 0; }
long tg_get_step(const TGModel *m) { return m ? m->step : 0; }
float tg_get_loss(const TGModel *m) { return m ? m->last_loss : 0.0f; }
void tg_set_lr(TGModel *m, float lr) { if (m && lr > 0) m->cfg.lr = lr; }
void tg_set_batch_size(TGModel *m, int batch) {
    if (!m || batch < 1 || batch > 256) return;
    if (batch == m->cfg.batch_size) return;
    /* batch 变大需要更大的激活缓存：重新分配 */
    m->cfg.batch_size = batch;
    free(m->acts); free(m->scratch);
    m->acts = (float *)malloc(sizeof(float) * tg_acts_size(m));
    m->scratch = (float *)malloc(sizeof(float) * tg_scratch_size(m));
}
long tg_corpus_tokens(const TGModel *m) { return m ? m->n_tokens : 0; }
const char *tg_vocab_str(const TGModel *m, int index) {
    if (!m || !m->vocab || index < 0 || index >= m->vocab_size) return NULL;
    return m->vocab[index];
}

/* ---------------- 字符 → token 哈希表 ---------------- */
static uint32_t hash_cp(uint32_t cp) {
    cp ^= cp >> 16; cp *= 0x7feb352dU; cp ^= cp >> 15; cp *= 0x846ca68bU; cp ^= cp >> 16;
    return cp;
}
static void vocab_hash_build(TGModel *m) {
    int cap = 1024;
    while (cap < m->vocab_size * 4) cap *= 2;
    free(m->h_keys); free(m->h_vals);
    m->h_cap = cap;
    m->h_keys = (uint32_t *)calloc((size_t)cap, sizeof(uint32_t));
    m->h_vals = (int *)malloc(sizeof(int) * (size_t)cap);
    for (int i = 4; i < m->vocab_size; i++) {
        uint32_t cp = m->vocab_cp[i];
        uint32_t h = hash_cp(cp) & (uint32_t)(cap - 1);
        while (m->h_keys[h] != 0) h = (h + 1) & (uint32_t)(cap - 1);
        m->h_keys[h] = cp + 1; /* +1 使 0 可表示空槽 */
        m->h_vals[h] = i;
    }
}
static int vocab_lookup(const TGModel *m, uint32_t cp) {
    if (!m->h_keys) return 3;
    uint32_t h = hash_cp(cp) & (uint32_t)(m->h_cap - 1);
    while (m->h_keys[h] != 0) {
        if (m->h_keys[h] == cp + 1) return m->h_vals[h];
        h = (h + 1) & (uint32_t)(m->h_cap - 1);
    }
    return 3; /* <unk> */
}

/* 频率统计用的排序项 */
typedef struct { uint32_t cp; long count; } CpCount;
static int cmp_cpcount(const void *a, const void *b) {
    const CpCount *x = (const CpCount *)a, *y = (const CpCount *)b;
    if (y->count != x->count) return y->count > x->count ? 1 : -1;
    return x->cp < y->cp ? -1 : (x->cp > y->cp ? 1 : 0);
}
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* ---------------- 语料与词表 ---------------- */
int tg_set_corpus(TGModel *m, const char *utf8_text) {
    if (!m || !utf8_text || !*utf8_text) return -1;
    size_t len = strlen(utf8_text);
    /* 解码全部码点 */
    uint32_t *cps = (uint32_t *)malloc(sizeof(uint32_t) * (len + 1));
    if (!cps) return -1;
    long n_cp = 0;
    for (size_t i = 0; i < len;) {
        int adv = 0;
        uint32_t cp = utf8_decode((const unsigned char *)utf8_text + i, (int)(len - i), &adv);
        if (adv <= 0) adv = 1;
        /* 跳过 \r（统一换行），其余全部保留 */
        if (cp != '\r') cps[n_cp++] = cp;
        i += (size_t)adv;
    }
    if (n_cp == 0) { free(cps); return -1; }

    if (!m->vocab_built) {
        /* 首次：统计频率并建词表 */
        uint32_t *sorted = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)n_cp);
        memcpy(sorted, cps, sizeof(uint32_t) * (size_t)n_cp);
        /* 简单排序后计数（语料通常 < 1MB，qsort 足够） */
        qsort(sorted, (size_t)n_cp, sizeof(uint32_t), cmp_u32);
        CpCount *counts = (CpCount *)malloc(sizeof(CpCount) * (size_t)(n_cp + 1));
        long n_distinct = 0;
        for (long i = 0; i < n_cp;) {
            long j = i;
            while (j < n_cp && sorted[j] == sorted[i]) j++;
            counts[n_distinct].cp = sorted[i];
            counts[n_distinct].count = j - i;
            n_distinct++;
            i = j;
        }
        free(sorted);
        qsort(counts, (size_t)n_distinct, sizeof(CpCount), cmp_cpcount);
        int want = m->cfg.vocab_size - 4;
        int take = (int)(n_distinct < want ? n_distinct : want);
        m->vocab_size = 4 + take;
        m->vocab = (char **)malloc(sizeof(char *) * (size_t)m->vocab_size);
        m->vocab_cp = (uint32_t *)calloc((size_t)m->vocab_size, sizeof(uint32_t));
        static const char *special[4] = {"<pad>", "<bos>", "<eos>", "<unk>"};
        for (int i = 0; i < 4; i++) {
            size_t sl = strlen(special[i]) + 1;
            m->vocab[i] = (char *)malloc(sl); memcpy(m->vocab[i], special[i], sl);
            m->vocab_cp[i] = 0;
        }
        for (int i = 0; i < take; i++) {
            char tmp[4]; int nb = utf8_encode(counts[i].cp, tmp);
            m->vocab[4 + i] = (char *)malloc((size_t)nb + 1);
            memcpy(m->vocab[4 + i], tmp, (size_t)nb); m->vocab[4 + i][nb] = '\0';
            m->vocab_cp[4 + i] = counts[i].cp;
        }
        free(counts);
        vocab_hash_build(m);
        m->vocab_built = 1;
    }
    /* 分词：码点 → token（不在词表 → <unk>=3） */
    free(m->tokens);
    m->tokens = (int *)malloc(sizeof(int) * (size_t)(n_cp + 2));
    if (!m->tokens) { free(cps); return -1; }
    long nt = 0;
    m->tokens[nt++] = 1; /* <bos> 开头 */
    for (long i = 0; i < n_cp; i++) m->tokens[nt++] = vocab_lookup(m, cps[i]);
    free(cps);
    m->n_tokens = nt;
    return 0;
}

int tg_encode(const TGModel *m, const char *utf8_text, int *out, int cap) {
    if (!m || !m->vocab_built || !utf8_text || !out || cap <= 0) return -1;
    size_t len = strlen(utf8_text);
    int n = 0;
    for (size_t i = 0; i < len && n < cap;) {
        int adv = 0;
        uint32_t cp = utf8_decode((const unsigned char *)utf8_text + i, (int)(len - i), &adv);
        if (adv <= 0) adv = 1;
        if (cp != '\r') out[n++] = vocab_lookup(m, cp);
        i += (size_t)adv;
    }
    return n;
}

int tg_decode(const TGModel *m, const int *tokens, int n, char *out, int cap) {
    if (!m || !m->vocab_built || !out || cap <= 0) return -1;
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int t = tokens[i];
        const char *s = NULL;
        char unk[4] = {(char)0xEF, (char)0xBF, (char)0xBD, '\0'}; /* U+FFFD */
        if (t == 3) s = unk;
        else if (t >= 4 && t < m->vocab_size) s = m->vocab[t];
        else continue; /* <pad>/<bos>/<eos>/非法 → 跳过 */
        size_t sl = strlen(s);
        if (pos + (int)sl >= cap) break;
        memcpy(out + pos, s, sl); pos += (int)sl;
    }
    out[pos] = '\0';
    return pos;
}

/* ================= 核心数学算子 ================= */

/* matmul 前向：out[bt,n] = bias[n] + sum_k inp[bt,k]*W[n,k]（W 为 [N,K] 行主序） */
static void matmul_forward(float *out, const float *inp, const float *W, const float *bias,
                           int BT, int N, int K) {
#pragma omp parallel for if (BT > 16)
    for (int bt = 0; bt < BT; bt++) {
        const float *x = inp + (size_t)bt * K;
        float *o = out + (size_t)bt * N;
        for (int n = 0; n < N; n++) {
            const float *w = W + (size_t)n * K;
            float s = bias ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++) s += x[k] * w[k];
            o[n] = s;
        }
    }
}
/* matmul 反向求 dW：dW[n,k] += sum_bt dout[bt,n]*inp[bt,k] */
static void matmul_backward_dW(float *dW, const float *dout, const float *inp,
                               int BT, int N, int K) {
#pragma omp parallel for if (N > 32)
    for (int n = 0; n < N; n++) {
        float *dw = dW + (size_t)n * K;
        for (int bt = 0; bt < BT; bt++) {
            float d = dout[(size_t)bt * N + n];
            const float *x = inp + (size_t)bt * K;
            for (int k = 0; k < K; k++) dw[k] += d * x[k];
        }
    }
}
/* matmul 反向求 dx：dx[bt,k] = sum_n dout[bt,n]*W[n,k] */
static void matmul_backward_dx(float *dx, const float *dout, const float *W,
                               int BT, int N, int K) {
#pragma omp parallel for if (BT > 16)
    for (int bt = 0; bt < BT; bt++) {
        const float *d = dout + (size_t)bt * N;
        float *x = dx + (size_t)bt * K;
        for (int k = 0; k < K; k++) x[k] = 0.0f;
        for (int n = 0; n < N; n++) {
            float dv = d[n];
            const float *w = W + (size_t)n * K;
            for (int k = 0; k < K; k++) x[k] += dv * w[k];
        }
    }
}
/* 偏置梯度：db[n] += sum_bt dout[bt,n] */
static void bias_backward(float *db, const float *dout, int BT, int N) {
#pragma omp parallel for if (N > 32)
    for (int n = 0; n < N; n++) {
        float s = 0.0f;
        for (int bt = 0; bt < BT; bt++) s += dout[(size_t)bt * N + n];
        db[n] += s;
    }
}

/* LayerNorm 前向（逐行）：out = (x-mean)*rstd*w + b；同时存 mean/rstd */
static void layernorm_forward(float *out, float *mean, float *rstd,
                              const float *inp, const float *w, const float *b,
                              int BT, int D) {
#pragma omp parallel for if (BT > 16)
    for (int bt = 0; bt < BT; bt++) {
        const float *x = inp + (size_t)bt * D;
        float *o = out + (size_t)bt * D;
        float mu = 0.0f, var = 0.0f;
        for (int d = 0; d < D; d++) mu += x[d];
        mu /= (float)D;
        for (int d = 0; d < D; d++) { float t = x[d] - mu; var += t * t; }
        var /= (float)D;
        float r = 1.0f / sqrtf(var + 1e-5f);
        mean[bt] = mu; rstd[bt] = r;
        for (int d = 0; d < D; d++) o[d] = (x[d] - mu) * r * w[d] + b[d];
    }
}
/* LayerNorm 反向求 dx（add=1 时累加到 dx，否则覆盖） */
static void layernorm_backward_dx(float *dx, const float *dout, const float *inp,
                                  const float *w, const float *mean, const float *rstd,
                                  int BT, int D, int add) {
#pragma omp parallel for if (BT > 16)
    for (int bt = 0; bt < BT; bt++) {
        const float *do_ = dout + (size_t)bt * D;
        const float *x = inp + (size_t)bt * D;
        float *dx_ = dx + (size_t)bt * D;
        float r = rstd[bt], mu = mean[bt];
        float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
        for (int d = 0; d < D; d++) {
            float xhat = (x[d] - mu) * r;
            float dxhat = do_[d] * w[d];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
        }
        float m1 = sum_dxhat / (float)D;
        float m2 = sum_dxhat_xhat / (float)D;
        for (int d = 0; d < D; d++) {
            float xhat = (x[d] - mu) * r;
            float dxhat = do_[d] * w[d];
            float v = r * (dxhat - m1 - xhat * m2);
            dx_[d] = add ? dx_[d] + v : v;
        }
    }
}
/* LayerNorm 反向求 dw/db（累加） */
static void layernorm_backward_wb(float *dw, float *db, const float *dout,
                                  const float *inp, const float *mean, const float *rstd,
                                  int BT, int D) {
#pragma omp parallel for if (D > 32)
    for (int d = 0; d < D; d++) {
        float sw = 0.0f, sb = 0.0f;
        for (int bt = 0; bt < BT; bt++) {
            float xhat = (inp[(size_t)bt * D + d] - mean[bt]) * rstd[bt];
            float do_ = dout[(size_t)bt * D + d];
            sw += do_ * xhat;
            sb += do_;
        }
        dw[d] += sw; db[d] += sb;
    }
}

/* GELU（tanh 近似）及其导数 */
static float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static float gelu_df(float x) {
    float c = 0.7978845608f;
    float t = tanhf(c * (x + 0.044715f * x * x * x));
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * x * x);
}

/* 因果自注意力前向：att 为 softmax 概率 [B,H,T,T]，attout 为拼接输出 [B,T,D] */
static void attention_forward(float *att, float *attout, const float *qkv,
                              int B, int T, int H, int D) {
    int dh = D / H;
    float scale = 1.0f / sqrtf((float)dh);
#pragma omp parallel for collapse(2) if (B * H > 1)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int t1 = 0; t1 < T; t1++) {
                float *prow = att + ((size_t)b * H + h) * (size_t)T * T + (size_t)t1 * T;
                const float *q = qkv + ((size_t)b * T + t1) * (size_t)3 * D + h * dh;
                float maxv = -1e30f;
                for (int t2 = 0; t2 <= t1; t2++) {
                    const float *k = qkv + ((size_t)b * T + t2) * (size_t)3 * D + D + h * dh;
                    float s = 0.0f;
                    for (int d = 0; d < dh; d++) s += q[d] * k[d];
                    s *= scale;
                    prow[t2] = s;
                    if (s > maxv) maxv = s;
                }
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++) { float e = expf(prow[t2] - maxv); prow[t2] = e; sum += e; }
                float inv = 1.0f / sum;
                for (int t2 = 0; t2 <= t1; t2++) prow[t2] *= inv;
                float *o = attout + ((size_t)b * T + t1) * (size_t)D + h * dh;
                for (int d = 0; d < dh; d++) o[d] = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++) {
                    const float *v = qkv + ((size_t)b * T + t2) * (size_t)3 * D + 2 * D + h * dh;
                    float p = prow[t2];
                    for (int d = 0; d < dh; d++) o[d] += p * v[d];
                }
            }
        }
    }
}
/* 因果自注意力反向：dqkv 须由调用者先清零 */
static void attention_backward(float *dqkv, const float *dattout, const float *att,
                               const float *qkv, int B, int T, int H, int D) {
    int dh = D / H;
    float scale = 1.0f / sqrtf((float)dh);
#pragma omp parallel for collapse(2) if (B * H > 1)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            float dprow[1024]; /* T <= 1024（配置校验保证） */
            for (int t1 = 0; t1 < T; t1++) {
                const float *prow = att + ((size_t)b * H + h) * (size_t)T * T + (size_t)t1 * T;
                const float *do_ = dattout + ((size_t)b * T + t1) * (size_t)D + h * dh;
                float s = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++) {
                    const float *v = qkv + ((size_t)b * T + t2) * (size_t)3 * D + 2 * D + h * dh;
                    float *dv = dqkv + ((size_t)b * T + t2) * (size_t)3 * D + 2 * D + h * dh;
                    float dp = 0.0f;
                    for (int d = 0; d < dh; d++) { dp += do_[d] * v[d]; dv[d] += prow[t2] * do_[d]; }
                    dprow[t2] = dp;
                    s += dp * prow[t2];
                }
                const float *q = qkv + ((size_t)b * T + t1) * (size_t)3 * D + h * dh;
                float *dq = dqkv + ((size_t)b * T + t1) * (size_t)3 * D + h * dh;
                for (int t2 = 0; t2 <= t1; t2++) {
                    float ds = prow[t2] * (dprow[t2] - s) * scale;
                    const float *k = qkv + ((size_t)b * T + t2) * (size_t)3 * D + D + h * dh;
                    float *dk = dqkv + ((size_t)b * T + t2) * (size_t)3 * D + D + h * dh;
                    for (int d = 0; d < dh; d++) { dq[d] += ds * k[d]; dk[d] += ds * q[d]; }
                }
            }
        }
    }
}

/* 交叉熵前向：logits 原地变为 softmax 概率；返回平均 loss */
static float crossentropy_forward(float *logits, const int *targets, int B, int T, int V) {
    long BT = (long)B * T;
    double total = 0.0;
#pragma omp parallel for reduction(+:total) if (BT > 64)
    for (long bt = 0; bt < BT; bt++) {
        float *lg = logits + bt * (long)V;
        float mx = lg[0];
        for (int v = 1; v < V; v++) if (lg[v] > mx) mx = lg[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++) { float e = expf(lg[v] - mx); lg[v] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++) lg[v] *= inv;
        int t = targets[bt];
        float p = (t >= 0 && t < V) ? lg[t] : 1e-9f;
        if (p < 1e-12f) p = 1e-12f;
        total += -log((double)p);
    }
    return (float)(total / (double)BT);
}
/* 交叉熵反向：dlogits = (probs - onehot) / (B*T) */
static void crossentropy_backward(float *dlogits, const float *probs, const int *targets,
                                  int B, int T, int V) {
    long BT = (long)B * T;
    float scale = 1.0f / (float)BT;
#pragma omp parallel for if (BT > 64)
    for (long bt = 0; bt < BT; bt++) {
        const float *p = probs + bt * (long)V;
        float *d = dlogits + bt * (long)V;
        int t = targets[bt];
        for (int v = 0; v < V; v++) d[v] = p[v] * scale;
        if (t >= 0 && t < V) d[t] -= scale;
    }
}

/* ---------------- 激活缓存布局 ---------------- */
#define TG_MAX_LAYERS 12
typedef struct {
    float *lin, *ln1, *qkv, *att, *attout, *x2, *ln2, *fcpre, *fc;
    float *ln1_mean, *ln1_rstd, *ln2_mean, *ln2_rstd;
} TGLayerActs;
typedef struct {
    float *emb, *xout, *lnf, *lnf_mean, *lnf_rstd, *logits;
    TGLayerActs layer[TG_MAX_LAYERS];
} TGActs;
typedef struct {
    float *dlogits, *dres, *gqkv, *gatt, *gfc, *g1, *g2, *g3;
} TGScratch;

static void tg_acts_layout(TGModel *m, int B, int T, TGActs *a) {
    const TGConfig *c = &m->cfg;
    int D = c->n_embd, H = c->n_head, V = m->vocab_size, L = c->n_layer;
    float *p = m->acts;
    a->emb = p; p += (size_t)B * T * D;
    for (int l = 0; l < L; l++) {
        TGLayerActs *la = &a->layer[l];
        la->lin = p; p += (size_t)B * T * D;
        la->ln1 = p; p += (size_t)B * T * D;
        la->qkv = p; p += (size_t)B * T * 3 * D;
        la->att = p; p += (size_t)B * H * T * T;
        la->attout = p; p += (size_t)B * T * D;
        la->x2 = p; p += (size_t)B * T * D;
        la->ln2 = p; p += (size_t)B * T * D;
        la->fcpre = p; p += (size_t)B * T * 4 * D;
        la->fc = p; p += (size_t)B * T * 4 * D;
        la->ln1_mean = p; p += (size_t)B * T;
        la->ln1_rstd = p; p += (size_t)B * T;
        la->ln2_mean = p; p += (size_t)B * T;
        la->ln2_rstd = p; p += (size_t)B * T;
    }
    a->xout = p; p += (size_t)B * T * D;
    a->lnf = p; p += (size_t)B * T * D;
    a->lnf_mean = p; p += (size_t)B * T;
    a->lnf_rstd = p; p += (size_t)B * T;
    a->logits = p; p += (size_t)B * T * V;
}
static void tg_scratch_layout(TGModel *m, int B, int T, TGScratch *s) {
    const TGConfig *c = &m->cfg;
    int D = c->n_embd, H = c->n_head, V = m->vocab_size;
    float *p = m->scratch;
    s->dlogits = p; p += (size_t)B * T * V;
    s->dres = p; p += (size_t)B * T * D;
    s->gqkv = p; p += (size_t)B * T * 3 * D;
    s->gatt = p; p += (size_t)B * H * T * T;
    s->gfc = p; p += (size_t)B * T * 4 * D;
    s->g1 = p; p += (size_t)B * T * D;
    s->g2 = p; p += (size_t)B * T * D;
    s->g3 = p; p += (size_t)B * T * D;
}

/* ---------------- 前向 ---------------- */
/* inputs [B,T]，targets [B,T] 或 NULL（生成时只算 logits）。
 * 返回平均交叉熵（targets 为 NULL 时返回 0）。 */
static float tg_forward(TGModel *m, const int *inputs, const int *targets, int B, int T) {
    const TGConfig *c = &m->cfg;
    int D = c->n_embd, H = c->n_head, V = m->vocab_size, L = c->n_layer;
    TGActs a; tg_acts_layout(m, B, T, &a);
    long BT = (long)B * T;
    float *tok = m->params + m->off_tok;
    float *pos = m->params + m->off_pos;
    /* embedding */
#pragma omp parallel for if (BT > 16)
    for (long bt = 0; bt < BT; bt++) {
        int t = (int)(bt % T);
        int id = inputs[bt];
        if (id < 0 || id >= V) id = 3;
        const float *te = tok + (size_t)id * D;
        const float *pe = pos + (size_t)t * D;
        float *o = a.emb + bt * (long)D;
        for (int d = 0; d < D; d++) o[d] = te[d] + pe[d];
    }
    memcpy(a.layer[0].lin, a.emb, sizeof(float) * (size_t)BT * D);
    for (int l = 0; l < L; l++) {
        TGLayerActs *la = &a.layer[l];
        float *lbase = m->params + m->layer_base[l];
        layernorm_forward(la->ln1, la->ln1_mean, la->ln1_rstd, la->lin,
                          lbase + m->o_ln1w, lbase + m->o_ln1b, (int)BT, D);
        matmul_forward(la->qkv, la->ln1, lbase + m->o_qkvw, lbase + m->o_qkvb,
                       (int)BT, 3 * D, D);
        attention_forward(la->att, la->attout, la->qkv, B, T, H, D);
        matmul_forward(la->x2, la->attout, lbase + m->o_attw, lbase + m->o_attb,
                       (int)BT, D, D);
#pragma omp parallel for if (BT * D > 4096)
        for (long i = 0; i < BT * D; i++) la->x2[i] += la->lin[i];
        layernorm_forward(la->ln2, la->ln2_mean, la->ln2_rstd, la->x2,
                          lbase + m->o_ln2w, lbase + m->o_ln2b, (int)BT, D);
        matmul_forward(la->fcpre, la->ln2, lbase + m->o_fcw, lbase + m->o_fcb,
                       (int)BT, 4 * D, D);
        long nfc = BT * 4 * D;
#pragma omp parallel for if (nfc > 4096)
        for (long i = 0; i < nfc; i++) la->fc[i] = gelu_f(la->fcpre[i]);
        float *out = (l + 1 < L) ? a.layer[l + 1].lin : a.xout;
        matmul_forward(out, la->fc, lbase + m->o_projw, lbase + m->o_projb,
                       (int)BT, D, 4 * D);
#pragma omp parallel for if (BT * D > 4096)
        for (long i = 0; i < BT * D; i++) out[i] += la->x2[i];
    }
    layernorm_forward(a.lnf, a.lnf_mean, a.lnf_rstd, a.xout,
                      m->params + m->off_lnf_w, m->params + m->off_lnf_b, (int)BT, D);
    matmul_forward(a.logits, a.lnf, tok, NULL, (int)BT, V, D);
    if (targets) return crossentropy_forward(a.logits, targets, B, T, V);
    return 0.0f;
}

/* ---------------- 反向 ---------------- */
/* 要求 tg_forward 刚跑过同一批数据（激活缓存有效）。梯度累加进 m->grads。 */
static void tg_backward(TGModel *m, const int *inputs, const int *targets, int B, int T) {
    const TGConfig *c = &m->cfg;
    int D = c->n_embd, H = c->n_head, V = m->vocab_size, L = c->n_layer;
    TGActs a; tg_acts_layout(m, B, T, &a);
    TGScratch s; tg_scratch_layout(m, B, T, &s);
    long BT = (long)B * T;
    float *tok = m->params + m->off_tok;
    float *gtok = m->grads + m->off_tok;
    /* 交叉熵 → dlogits */
    crossentropy_backward(s.dlogits, a.logits, targets, B, T, V);
    /* 输出头（共享 embedding）：dlnf = dlogits @ tok；dTokEmbd += dlogits^T @ lnf */
    matmul_backward_dx(s.g1, s.dlogits, tok, (int)BT, V, D);
    matmul_backward_dW(gtok, s.dlogits, a.lnf, (int)BT, V, D);
    /* 最终 LN：dlnf → d(xout)，写进 dres（覆盖） */
    layernorm_backward_dx(s.dres, s.g1, a.xout, m->params + m->off_lnf_w,
                          a.lnf_mean, a.lnf_rstd, (int)BT, D, 0);
    layernorm_backward_wb(m->grads + m->off_lnf_w, m->grads + m->off_lnf_b, s.g1,
                          a.xout, a.lnf_mean, a.lnf_rstd, (int)BT, D);
    for (int l = L - 1; l >= 0; l--) {
        TGLayerActs *la = &a.layer[l];
        float *lbase = m->params + m->layer_base[l];
        float *gbase = m->grads + m->layer_base[l];
        /* s.dres = d(out)；out = x2 + proj(fc) */
        matmul_backward_dx(s.gfc, s.dres, lbase + m->o_projw, (int)BT, D, 4 * D);
        matmul_backward_dW(gbase + m->o_projw, s.dres, la->fc, (int)BT, D, 4 * D);
        bias_backward(gbase + m->o_projb, s.dres, (int)BT, D);
        /* GELU 反向：gfc 原地变为 d(fcpre) */
        long nfc = BT * 4 * D;
#pragma omp parallel for if (nfc > 4096)
        for (long i = 0; i < nfc; i++) s.gfc[i] *= gelu_df(la->fcpre[i]);
        /* fc 层：d(fcpre) → d(ln2) */
        matmul_backward_dx(s.g1, s.gfc, lbase + m->o_fcw, (int)BT, 4 * D, D);
        matmul_backward_dW(gbase + m->o_fcw, s.gfc, la->ln2, (int)BT, 4 * D, D);
        bias_backward(gbase + m->o_fcb, s.gfc, (int)BT, 4 * D);
        /* ln2：d(ln2) → d(x2)，累加进 dres（残差部分已在其中） */
        layernorm_backward_dx(s.dres, s.g1, la->x2, lbase + m->o_ln2w,
                              la->ln2_mean, la->ln2_rstd, (int)BT, D, 1);
        layernorm_backward_wb(gbase + m->o_ln2w, gbase + m->o_ln2b, s.g1,
                              la->x2, la->ln2_mean, la->ln2_rstd, (int)BT, D);
        /* s.dres 现为总 d(x2)；x2 = lin + attproj */
        matmul_backward_dx(s.g2, s.dres, lbase + m->o_attw, (int)BT, D, D);
        matmul_backward_dW(gbase + m->o_attw, s.dres, la->attout, (int)BT, D, D);
        bias_backward(gbase + m->o_attb, s.dres, (int)BT, D);
        /* 注意力反向：d(attout) → d(qkv) */
        memset(s.gqkv, 0, sizeof(float) * (size_t)BT * 3 * D);
        attention_backward(s.gqkv, s.g2, la->att, la->qkv, B, T, H, D);
        /* qkv 层：d(qkv) → d(ln1) */
        matmul_backward_dx(s.g3, s.gqkv, lbase + m->o_qkvw, (int)BT, 3 * D, D);
        matmul_backward_dW(gbase + m->o_qkvw, s.gqkv, la->ln1, (int)BT, 3 * D, D);
        bias_backward(gbase + m->o_qkvb, s.gqkv, (int)BT, 3 * D);
        /* ln1：d(ln1) → d(lin)，累加进 dres（注意力残差部分已在其中） */
        layernorm_backward_dx(s.dres, s.g3, la->lin, lbase + m->o_ln1w,
                              la->ln1_mean, la->ln1_rstd, (int)BT, D, 1);
        layernorm_backward_wb(gbase + m->o_ln1w, gbase + m->o_ln1b, s.g3,
                              la->lin, la->ln1_mean, la->ln1_rstd, (int)BT, D);
        /* s.dres 现为 d(lin[l])，即上一层的 d(out)，继续循环 */
    }
    /* embedding 反向：dres 为 d(emb) */
    /* dTokEmbd：按 token 位置散射累加（串行，避免竞态） */
    for (long bt = 0; bt < BT; bt++) {
        int id = inputs[bt];
        if (id < 0 || id >= V) id = 3;
        float *g = gtok + (size_t)id * D;
        const float *d = s.dres + bt * (long)D;
        for (int k = 0; k < D; k++) g[k] += d[k];
    }
    /* dPosEmbd：对 batch 求和 */
    float *gpos = m->grads + m->off_pos;
#pragma omp parallel for if (T > 16)
    for (int t = 0; t < T; t++) {
        float *g = gpos + (size_t)t * D;
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int b = 0; b < B; b++) sum += s.dres[((size_t)b * T + t) * D + d];
            g[d] += sum;
        }
    }
}

/* ---------------- 学习率调度：warmup + 余弦衰减到 10% ---------------- */
static float tg_current_lr(TGModel *m) {
    const TGConfig *c = &m->cfg;
    long step = m->step;
    float peak = c->lr;
    if (c->warmup_steps > 0 && step < c->warmup_steps)
        return peak * (float)(step + 1) / (float)c->warmup_steps;
    long span = c->decay_steps - c->warmup_steps;
    if (span < 1) span = 1;
    float prog = (float)(step - c->warmup_steps) / (float)span;
    if (prog > 1.0f) prog = 1.0f;
    if (prog < 0.0f) prog = 0.0f;
    float co = 0.5f * (1.0f + cosf((float)M_PI * prog));
    return peak * (0.1f + 0.9f * co);
}

/* AdamW 单段更新（wd>0 时对该段做权重衰减） */
static void adamw_segment(float *p, float *g, float *mm, float *vv, long n,
                          float lr, float b1, float b2, float eps, float wd,
                          long t, float bc1, float bc2) {
    (void)t; /* 时间步已折进 bc1/bc2，参数保留作接口说明 */
#pragma omp parallel for if (n > 4096)
    for (long i = 0; i < n; i++) {
        float grad = g[i];
        mm[i] = b1 * mm[i] + (1.0f - b1) * grad;
        vv[i] = b2 * vv[i] + (1.0f - b2) * grad * grad;
        float mh = mm[i] / bc1;
        float vh = vv[i] / bc2;
        p[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
    }
}
/* AdamW 整步：先全局梯度裁剪，再按段更新（仅矩阵做权重衰减） */
static void tg_adamw_step(TGModel *m, float lr) {
    const TGConfig *c = &m->cfg;
    long n = m->n_params;
    /* 全局梯度范数 */
    double norm2 = 0.0;
#pragma omp parallel for reduction(+:norm2) if (n > 4096)
    for (long i = 0; i < n; i++) norm2 += (double)m->grads[i] * (double)m->grads[i];
    if (c->grad_clip > 0) {
        double norm = sqrt(norm2);
        if (norm > c->grad_clip) {
            float scale = (float)(c->grad_clip / norm);
#pragma omp parallel for if (n > 4096)
            for (long i = 0; i < n; i++) m->grads[i] *= scale;
        }
    }
    long t = m->step + 1; /* Adam 时间步（1 起） */
    float b1 = c->beta1, b2 = c->beta2, eps = c->eps;
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    float wd = c->weight_decay;
    long D = c->n_embd;
    float *p = m->params, *g = m->grads, *mm = m->m, *vv = m->v;
    /* token / position embedding：矩阵，衰减 */
    adamw_segment(p + m->off_tok, g + m->off_tok, mm + m->off_tok, vv + m->off_tok,
                  (long)m->vocab_cap * D, lr, b1, b2, eps, wd, t, bc1, bc2);
    adamw_segment(p + m->off_pos, g + m->off_pos, mm + m->off_pos, vv + m->off_pos,
                  (long)c->block_size * D, lr, b1, b2, eps, wd, t, bc1, bc2);
    for (int l = 0; l < c->n_layer; l++) {
        float *pb = p + m->layer_base[l], *gb = g + m->layer_base[l];
        float *mb = mm + m->layer_base[l], *vb = vv + m->layer_base[l];
        adamw_segment(pb + m->o_ln1w, gb + m->o_ln1w, mb + m->o_ln1w, vb + m->o_ln1w, D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_ln1b, gb + m->o_ln1b, mb + m->o_ln1b, vb + m->o_ln1b, D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_qkvw, gb + m->o_qkvw, mb + m->o_qkvw, vb + m->o_qkvw, 3 * D * D, lr, b1, b2, eps, wd, t, bc1, bc2);
        adamw_segment(pb + m->o_qkvb, gb + m->o_qkvb, mb + m->o_qkvb, vb + m->o_qkvb, 3 * D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_attw, gb + m->o_attw, mb + m->o_attw, vb + m->o_attw, D * D, lr, b1, b2, eps, wd, t, bc1, bc2);
        adamw_segment(pb + m->o_attb, gb + m->o_attb, mb + m->o_attb, vb + m->o_attb, D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_ln2w, gb + m->o_ln2w, mb + m->o_ln2w, vb + m->o_ln2w, D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_ln2b, gb + m->o_ln2b, mb + m->o_ln2b, vb + m->o_ln2b, D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_fcw, gb + m->o_fcw, mb + m->o_fcw, vb + m->o_fcw, 4 * D * D, lr, b1, b2, eps, wd, t, bc1, bc2);
        adamw_segment(pb + m->o_fcb, gb + m->o_fcb, mb + m->o_fcb, vb + m->o_fcb, 4 * D, lr, b1, b2, eps, 0, t, bc1, bc2);
        adamw_segment(pb + m->o_projw, gb + m->o_projw, mb + m->o_projw, vb + m->o_projw, 4 * D * D, lr, b1, b2, eps, wd, t, bc1, bc2);
        adamw_segment(pb + m->o_projb, gb + m->o_projb, mb + m->o_projb, vb + m->o_projb, D, lr, b1, b2, eps, 0, t, bc1, bc2);
    }
    adamw_segment(p + m->off_lnf_w, g + m->off_lnf_w, mm + m->off_lnf_w, vv + m->off_lnf_w, D, lr, b1, b2, eps, 0, t, bc1, bc2);
    adamw_segment(p + m->off_lnf_b, g + m->off_lnf_b, mm + m->off_lnf_b, vv + m->off_lnf_b, D, lr, b1, b2, eps, 0, t, bc1, bc2);
}

/* ---------------- 训练 ---------------- */
/* 单步：采样一批随机窗口 → 前向 → 反向 → AdamW。返回 loss。 */
static float tg_train_one_step(TGModel *m, int *batch_in, int *batch_tgt) {
    const TGConfig *c = &m->cfg;
    int B = c->batch_size, T = c->block_size;
    long nt = m->n_tokens;
    for (int b = 0; b < B; b++) {
        long start = (long)(rng_next(&m->rng_state) % (uint64_t)nt);
        for (int t = 0; t < T; t++) {
            long idx = (start + t) % nt;
            long idx2 = (start + t + 1) % nt;
            batch_in[(size_t)b * T + t] = m->tokens[idx];
            batch_tgt[(size_t)b * T + t] = m->tokens[idx2];
        }
    }
    memset(m->grads, 0, sizeof(float) * (size_t)m->n_params);
    float loss = tg_forward(m, batch_in, batch_tgt, B, T);
    tg_backward(m, batch_in, batch_tgt, B, T);
    float lr = tg_current_lr(m);
    tg_adamw_step(m, lr);
    m->step++;
    m->last_loss = loss;
    return loss;
}

int tg_train_steps(TGModel *m, int steps, float *losses) {
    if (!m || !m->vocab_built || !m->tokens || m->n_tokens < 2 || steps < 1 || !losses)
        return -1;
    int B = m->cfg.batch_size, T = m->cfg.block_size;
    int *batch_in = (int *)malloc(sizeof(int) * (size_t)B * T);
    int *batch_tgt = (int *)malloc(sizeof(int) * (size_t)B * T);
    if (!batch_in || !batch_tgt) { free(batch_in); free(batch_tgt); return -1; }
    for (int s = 0; s < steps; s++)
        losses[s] = tg_train_one_step(m, batch_in, batch_tgt);
    free(batch_in); free(batch_tgt);
    return 0;
}

/* ---------------- 生成 ---------------- */
int tg_generate(TGModel *m, const char *prompt, int max_tokens,
                float temperature, int top_k, uint64_t seed,
                char *out, int cap) {
    if (!m || !m->vocab_built || !out || cap <= 0 || max_tokens < 1) return -1;
    int T = m->cfg.block_size;
    int V = m->vocab_size;
    int *ctx = (int *)malloc(sizeof(int) * (size_t)(T + max_tokens + 2));
    if (!ctx) return -1;
    int n = 0;
    ctx[n++] = 1; /* <bos> */
    if (prompt && *prompt) {
        int got = tg_encode(m, prompt, ctx + n, T - 1);
        if (got > 0) n += got;
    }
    uint64_t gs = seed ? seed : 0x123456789ABCDEFULL;
    int gen = 0;
    int *out_tokens = (int *)malloc(sizeof(int) * (size_t)(max_tokens + 1));
    if (!out_tokens) { free(ctx); return -1; }
    while (gen < max_tokens) {
        int clen = n < T ? n : T; /* 只用最近 block_size 个 token */
        int *win = ctx + (n - clen);
        /* 前向（B=1，T=clen） */
        float loss_unused = tg_forward(m, win, NULL, 1, clen);
        (void)loss_unused;
        TGActs a; tg_acts_layout(m, 1, clen, &a);
        float *last_logits = a.logits + (size_t)(clen - 1) * V;
        int next;
        if (temperature <= 0.0f || top_k == 1) {
            /* 贪心 */
            next = 0;
            for (int v = 1; v < V; v++) if (last_logits[v] > last_logits[next]) next = v;
        } else {
            /* 温度 + Top-K + 采样 */
            float *tmp = (float *)malloc(sizeof(float) * (size_t)V);
            if (!tmp) { free(ctx); free(out_tokens); return -1; }
            for (int v = 0; v < V; v++) tmp[v] = last_logits[v] / temperature;
            if (top_k > 0 && top_k < V) {
                /* 找第 top_k 大的阈值（简单选择法，V 很小） */
                float *vals = (float *)malloc(sizeof(float) * (size_t)V);
                memcpy(vals, tmp, sizeof(float) * (size_t)V);
                for (int i = 0; i < top_k; i++) {
                    int bi = i;
                    for (int j = i + 1; j < V; j++) if (vals[j] > vals[bi]) bi = j;
                    float t2 = vals[i]; vals[i] = vals[bi]; vals[bi] = t2;
                }
                float thr = vals[top_k - 1];
                free(vals);
                for (int v = 0; v < V; v++) if (tmp[v] < thr) tmp[v] = -1e30f;
            }
            float mx = tmp[0];
            for (int v = 1; v < V; v++) if (tmp[v] > mx) mx = tmp[v];
            double sum = 0.0;
            for (int v = 0; v < V; v++) { double e = exp((double)(tmp[v] - mx)); tmp[v] = (float)e; sum += e; }
            double r = rng_uniform(&gs) * sum;
            double acc = 0.0;
            next = V - 1;
            for (int v = 0; v < V; v++) { acc += tmp[v]; if (r <= acc) { next = v; break; } }
            free(tmp);
        }
        out_tokens[gen++] = next;
        if (n < T + max_tokens + 1) ctx[n++] = next;
        else { memmove(ctx, ctx + 1, sizeof(int) * (size_t)(T + max_tokens)); ctx[T + max_tokens] = next; n = T + max_tokens + 1; }
        if (next == 2) break; /* <eos> 停 */
    }
    int written = tg_decode(m, out_tokens, gen, out, cap);
    free(ctx); free(out_tokens);
    return written;
}

/* ---------------- 大脑文件（.brain） ---------------- */
/* 布局：magic "TBRN" | version u32=1 | header_len u32 | header JSON |
 *       权重 fp32 LE | AdamW m fp32 LE | AdamW v fp32 LE
 * header JSON：{"model":{n_layer,n_embd,n_head,block_size,vocab_size,vocab_cap},
 *   "vocab":[...], "train":{step,loss,lr,batch_size,seed,rng_state}}
 * vocab_cap 为扩展字段（= embedding 分配行数）；缺失时视为 = vocab_size。
 * m/v 段缺失时按 0 初始化，仍可加载（继续训练的结果会与保存前略有不同，
 * 因为 Adam 状态丢失；带 m/v 段时继续训练逐 bit 一致）。 */

static void sb_put_long(StrBuf *sb, long v) { char t[32]; snprintf(t, sizeof(t), "%ld", v); sb_puts(sb, t); }
static void sb_put_ull(StrBuf *sb, unsigned long long v) { char t[32]; snprintf(t, sizeof(t), "%llu", v); sb_puts(sb, t); }
static void sb_put_double(StrBuf *sb, double v) { char t[40]; snprintf(t, sizeof(t), "%.9g", v); sb_puts(sb, t); }

static int write_u32_le(FILE *f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v & 0xFF), (unsigned char)((v >> 8) & 0xFF),
                          (unsigned char)((v >> 16) & 0xFF), (unsigned char)((v >> 24) & 0xFF)};
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}
static int read_u32_le(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
/* 读/写 float 数组（本机即小端；用 memcpy 保证格式与内存一致） */
static int write_floats(FILE *f, const float *p, long n) {
    return fwrite(p, sizeof(float), (size_t)n, f) == (size_t)n ? 0 : -1;
}
static int read_floats(FILE *f, float *p, long n) {
    return fread(p, sizeof(float), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int tg_save_brain(const TGModel *m, const char *path) {
    if (!m || !m->vocab_built || !path) return -1;
    StrBuf sb = {0};
    sb_puts(&sb, "{\"model\":{\"n_layer\":");
    sb_put_long(&sb, m->cfg.n_layer);
    sb_puts(&sb, ",\"n_embd\":"); sb_put_long(&sb, m->cfg.n_embd);
    sb_puts(&sb, ",\"n_head\":"); sb_put_long(&sb, m->cfg.n_head);
    sb_puts(&sb, ",\"block_size\":"); sb_put_long(&sb, m->cfg.block_size);
    sb_puts(&sb, ",\"vocab_size\":"); sb_put_long(&sb, m->vocab_size);
    sb_puts(&sb, ",\"vocab_cap\":"); sb_put_long(&sb, m->vocab_cap);
    sb_puts(&sb, "},\"vocab\":[");
    for (int i = 0; i < m->vocab_size; i++) {
        if (i) sb_putc(&sb, ',');
        sb_put_json_string(&sb, m->vocab[i]);
    }
    sb_puts(&sb, "],\"train\":{\"step\":");
    sb_put_long(&sb, m->step);
    sb_puts(&sb, ",\"loss\":"); sb_put_double(&sb, m->last_loss);
    sb_puts(&sb, ",\"lr\":"); sb_put_double(&sb, m->cfg.lr);
    sb_puts(&sb, ",\"batch_size\":"); sb_put_long(&sb, m->cfg.batch_size);
    sb_puts(&sb, ",\"seed\":"); sb_put_ull(&sb, m->cfg.seed);
    sb_puts(&sb, ",\"rng_state\":"); sb_put_ull(&sb, m->rng_state);
    /* 学习率调度与优化器超参数也必须保存，否则加载后继续训练的学习率
     * 会回到默认值，续训轨迹与保存前不一致 */
    sb_puts(&sb, ",\"warmup_steps\":"); sb_put_long(&sb, m->cfg.warmup_steps);
    sb_puts(&sb, ",\"decay_steps\":"); sb_put_long(&sb, m->cfg.decay_steps);
    sb_puts(&sb, ",\"weight_decay\":"); sb_put_double(&sb, m->cfg.weight_decay);
    sb_puts(&sb, ",\"grad_clip\":"); sb_put_double(&sb, m->cfg.grad_clip);
    sb_puts(&sb, ",\"beta1\":"); sb_put_double(&sb, m->cfg.beta1);
    sb_puts(&sb, ",\"beta2\":"); sb_put_double(&sb, m->cfg.beta2);
    sb_puts(&sb, ",\"eps\":"); sb_put_double(&sb, m->cfg.eps);
    sb_puts(&sb, "}}");
    FILE *f = fopen(path, "wb");
    if (!f) { free(sb.buf); return -1; }
    int rc = 0;
    if (fwrite("TBRN", 1, 4, f) != 4) rc = -1;
    if (rc == 0 && write_u32_le(f, 1) != 0) rc = -1;
    if (rc == 0 && write_u32_le(f, (uint32_t)sb.len) != 0) rc = -1;
    if (rc == 0 && fwrite(sb.buf, 1, sb.len, f) != sb.len) rc = -1;
    if (rc == 0 && write_floats(f, m->params, m->n_params) != 0) rc = -1;
    if (rc == 0 && write_floats(f, m->m, m->n_params) != 0) rc = -1;
    if (rc == 0 && write_floats(f, m->v, m->n_params) != 0) rc = -1;
    free(sb.buf);
    if (fclose(f) != 0) rc = -1;
    return rc;
}

/* 安装词表（接管 vocab 字符串与 vocab_cp 数组的所有权） */
static void tg_install_vocab(TGModel *m, char **vocab, uint32_t *vocab_cp, int vocab_size) {
    m->vocab = vocab;
    m->vocab_cp = vocab_cp;
    m->vocab_size = vocab_size;
    vocab_hash_build(m);
    m->vocab_built = 1;
}

static unsigned long long json_get_ull(const char *json, const char *key, unsigned long long def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    return strtoull(p + 1, NULL, 10);
}

TGModel *tg_load_brain(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    uint32_t version = 0, header_len = 0;
    int ok = 1;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TBRN", 4) != 0) ok = 0;
    if (ok && read_u32_le(f, &version) != 0) ok = 0;
    if (ok && version != 1) ok = 0;
    if (ok && read_u32_le(f, &header_len) != 0) ok = 0;
    if (ok && (header_len == 0 || header_len > (1u << 26))) ok = 0;
    char *header = NULL;
    if (ok) {
        header = (char *)malloc((size_t)header_len + 1);
        if (!header) ok = 0;
        else if (fread(header, 1, header_len, f) != header_len) ok = 0;
        else header[header_len] = '\0';
    }
    TGModel *m = NULL;
    if (ok) {
        TGConfig cfg;
        tg_config_default(&cfg);
        cfg.n_layer = (int)json_get_long(header, "n_layer", cfg.n_layer);
        cfg.n_embd = (int)json_get_long(header, "n_embd", cfg.n_embd);
        cfg.n_head = (int)json_get_long(header, "n_head", cfg.n_head);
        cfg.block_size = (int)json_get_long(header, "block_size", cfg.block_size);
        long vs = json_get_long(header, "vocab_size", cfg.vocab_size);
        long vc = json_get_long(header, "vocab_cap", vs);
        cfg.vocab_size = (int)vc;
        cfg.lr = (float)json_get_double(header, "lr", cfg.lr);
        cfg.batch_size = (int)json_get_long(header, "batch_size", cfg.batch_size);
        cfg.seed = json_get_ull(header, "seed", cfg.seed);
        /* 学习率调度与优化器超参数（旧文件缺失时用默认值，保持兼容） */
        cfg.warmup_steps = (int)json_get_long(header, "warmup_steps", cfg.warmup_steps);
        cfg.decay_steps = (int)json_get_long(header, "decay_steps", cfg.decay_steps);
        cfg.weight_decay = (float)json_get_double(header, "weight_decay", cfg.weight_decay);
        cfg.grad_clip = (float)json_get_double(header, "grad_clip", cfg.grad_clip);
        cfg.beta1 = (float)json_get_double(header, "beta1", cfg.beta1);
        cfg.beta2 = (float)json_get_double(header, "beta2", cfg.beta2);
        cfg.eps = (float)json_get_double(header, "eps", cfg.eps);
        if (tg_config_validate(&cfg) != 0) ok = 0;
        else {
            m = tg_create(&cfg);
            if (!m) ok = 0;
        }
    }
    /* 解析 vocab 数组 */
    if (ok) {
        const char *p = strstr(header, "\"vocab\"");
        if (!p) ok = 0;
        else {
            p = strchr(p, '[');
            if (!p) ok = 0;
            else {
                p++;
                int cap = 64, n = 0;
                char **vocab = (char **)malloc(sizeof(char *) * (size_t)cap);
                uint32_t *vocab_cp = (uint32_t *)calloc((size_t)cap, sizeof(uint32_t));
                if (!vocab || !vocab_cp) { free(vocab); free(vocab_cp); ok = 0; }
                else {
                    while (*p && *p != ']') {
                        if (*p == '"') {
                            const char *end = NULL;
                            char *s = json_parse_string(p, &end);
                            if (!s) { ok = 0; break; }
                            if (n >= cap) {
                                cap *= 2;
                                vocab = (char **)realloc(vocab, sizeof(char *) * (size_t)cap);
                                vocab_cp = (uint32_t *)realloc(vocab_cp, sizeof(uint32_t) * (size_t)cap);
                                if (!vocab || !vocab_cp) { free(s); ok = 0; break; }
                            }
                            vocab[n] = s;
                            if (n >= 4) {
                                int adv = 0;
                                vocab_cp[n] = utf8_decode((const unsigned char *)s, 4, &adv);
                            } else vocab_cp[n] = 0;
                            n++;
                            p = end;
                        } else p++;
                    }
                    if (ok) {
                        if (n == 0) { free(vocab); free(vocab_cp); ok = 0; }
                        else tg_install_vocab(m, vocab, vocab_cp, n);
                    } else {
                        for (int i = 0; i < n; i++) free(vocab[i]);
                        free(vocab); free(vocab_cp);
                    }
                }
            }
        }
    }
    /* 读权重 + Adam 状态 */
    if (ok) {
        if (read_floats(f, m->params, m->n_params) != 0) ok = 0;
        else {
            /* 若文件还有 m/v 段则读入，否则保持 0 */
            if (read_floats(f, m->m, m->n_params) != 0) {
                memset(m->m, 0, sizeof(float) * (size_t)m->n_params);
                memset(m->v, 0, sizeof(float) * (size_t)m->n_params);
            } else if (read_floats(f, m->v, m->n_params) != 0) {
                memset(m->v, 0, sizeof(float) * (size_t)m->n_params);
            }
        }
    }
    if (ok) {
        m->step = json_get_long(header, "step", 0);
        m->last_loss = (float)json_get_double(header, "loss", 0.0);
        m->rng_state = json_get_ull(header, "rng_state", m->cfg.seed);
    }
    free(header);
    fclose(f);
    if (!ok) { tg_free(m); return NULL; }
    return m;
}

/* ---------------- 从 JSON 配置创建 ---------------- */
TGModel *tg_create_from_json(const char *json) {
    if (!json) return NULL;
    TGConfig cfg;
    tg_config_default(&cfg);
    cfg.n_layer = (int)json_get_long(json, "n_layer", cfg.n_layer);
    cfg.n_embd = (int)json_get_long(json, "n_embd", cfg.n_embd);
    cfg.n_head = (int)json_get_long(json, "n_head", cfg.n_head);
    cfg.block_size = (int)json_get_long(json, "block_size", cfg.block_size);
    cfg.vocab_size = (int)json_get_long(json, "vocab_size", cfg.vocab_size);
    cfg.lr = (float)json_get_double(json, "lr", cfg.lr);
    cfg.batch_size = (int)json_get_long(json, "batch_size", cfg.batch_size);
    cfg.warmup_steps = (int)json_get_long(json, "warmup_steps", cfg.warmup_steps);
    cfg.decay_steps = (int)json_get_long(json, "decay_steps", cfg.decay_steps);
    cfg.weight_decay = (float)json_get_double(json, "weight_decay", cfg.weight_decay);
    cfg.grad_clip = (float)json_get_double(json, "grad_clip", cfg.grad_clip);
    cfg.beta1 = (float)json_get_double(json, "beta1", cfg.beta1);
    cfg.beta2 = (float)json_get_double(json, "beta2", cfg.beta2);
    cfg.eps = (float)json_get_double(json, "eps", cfg.eps);
    cfg.seed = json_get_ull(json, "seed", cfg.seed);
    return tg_create(&cfg);
}
