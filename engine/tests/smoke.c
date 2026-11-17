/*
 * smoke.c — 桌面冒烟测试
 *
 * 用法：smoke [--config small|default] [--steps N] [--corpus PATH] [--seed N]
 * 默认：small 配置，200 步，tests/corpus.txt。
 *
 * 流程：
 *  1. 读语料，建模型，建词表
 *  2. 训 N 步，打印每步 loss；检查后 10 步均值 < 前 10 步均值 * 0.75
 *  3. 生成一段文本并打印
 *  4. 保存 .brain → 释放 → 加载 → 同种子生成必须逐字相同；
 *     再各训 5 步，loss 必须逐 bit 相同（验证 Adam/RNG 状态往返）
 * 任一检查失败返回非零退出码。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../tinygpt.h"

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = '\0';
    fclose(f);
    if (out_len) *out_len = n;
    return buf;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *config_name = "small";
    const char *corpus_path = "tests/corpus.txt";
    int steps = 200;
    unsigned long long seed = 1337;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config_name = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "未知参数：%s\n", argv[i]); return 2; }
    }

    TGConfig cfg;
    tg_config_default(&cfg);
    cfg.seed = seed;
    cfg.lr = 3e-3f;          /* 冒烟测试用较大学习率，让 loss 下降更明显 */
    cfg.warmup_steps = 10;
    cfg.decay_steps = 1000;
    if (!strcmp(config_name, "small")) {
        cfg.n_layer = 2; cfg.n_embd = 128; cfg.n_head = 4;
        cfg.block_size = 64; cfg.vocab_size = 1000; cfg.batch_size = 4;
    } else if (!strcmp(config_name, "default")) {
        cfg.batch_size = 8;  /* 默认结构，batch 8（桌面验证用） */
    } else {
        fprintf(stderr, "未知配置：%s（用 small 或 default）\n", config_name);
        return 2;
    }

    char *corpus = read_file(corpus_path, NULL);
    if (!corpus) { fprintf(stderr, "读不到语料：%s\n", corpus_path); return 2; }

    TGModel *m = tg_create(&cfg);
    if (!m) { fprintf(stderr, "创建模型失败\n"); return 2; }
    if (tg_set_corpus(m, corpus) != 0) { fprintf(stderr, "设置语料失败\n"); return 2; }
    printf("配置：%s 层=%d 宽=%d 头=%d 序列=%d batch=%d lr=%g\n",
           config_name, cfg.n_layer, cfg.n_embd, cfg.n_head,
           cfg.block_size, cfg.batch_size, (double)cfg.lr);
    printf("参数量：%ld  词表：%d  语料 token：%ld\n",
           tg_param_count(m), tg_vocab_size(m), tg_corpus_tokens(m));
    /* 注意：corpus 先不释放，加载后的模型续训前还要再 set 一次语料
     *（大脑文件不存语料，加载后需重新设置语料才能继续训练） */

    float *losses = (float *)malloc(sizeof(float) * (size_t)steps);
    if (!losses) return 2;
    double t0 = now_sec();
    if (tg_train_steps(m, steps, losses) != 0) {
        fprintf(stderr, "训练失败\n"); return 2;
    }
    double dt = now_sec() - t0;
    for (int i = 0; i < steps; i++)
        printf("step %4d  loss %.4f\n", i + 1, losses[i]);
    printf("训练用时 %.1f 秒，%.2f 步/秒\n", dt, steps / dt);

    double first10 = 0, last10 = 0;
    for (int i = 0; i < 10; i++) first10 += losses[i];
    for (int i = steps - 10; i < steps; i++) last10 += losses[i];
    first10 /= 10.0; last10 /= 10.0;
    printf("前 10 步平均 loss：%.4f\n后 10 步平均 loss：%.4f\n", first10, last10);
    int loss_ok = (last10 < first10 * 0.75) && (last10 < first10 - 1.0);
    printf("loss 下降检查：%s\n", loss_ok ? "通过" : "失败");

    /* 生成 */
    char gen1[4096];
    int rc = tg_generate(m, "小白猫", 60, 0.8f, 40, 42, gen1, sizeof(gen1));
    if (rc < 0) { fprintf(stderr, "生成失败\n"); return 2; }
    printf("生成（prompt=小白猫，温度 0.8，Top-K 40）：\n%s\n", gen1);

    /* 保存 → 释放 → 加载 → 生成必须相同；再各训 5 步 loss 必须相同 */
    const char *brain_path = "/tmp/tinygpt_smoke.brain";
    if (tg_save_brain(m, brain_path) != 0) { fprintf(stderr, "保存大脑失败\n"); return 2; }
    printf("已保存大脑到 %s\n", brain_path);

    /* 先在原模型上多训 5 步，记录 loss */
    float la[5];
    if (tg_train_steps(m, 5, la) != 0) { fprintf(stderr, "续训失败\n"); return 2; }
    char genA[4096];
    /* 注意：生成用的是多训 5 步之后的权重；为保证可比，加载后的模型也先训 5 步再生成 */
    TGModel *m2 = tg_load_brain(brain_path);
    if (!m2) { fprintf(stderr, "加载大脑失败\n"); return 2; }
    printf("加载后：步数=%ld 词表=%d 参数量=%ld\n",
           tg_get_step(m2), tg_vocab_size(m2), tg_param_count(m2));
    /* 大脑文件不存语料：加载后需重新设置同一份语料才能继续训练 */
    if (tg_set_corpus(m2, corpus) != 0) { fprintf(stderr, "加载后设置语料失败\n"); return 2; }
    printf("加载后重设语料：语料 token=%ld\n", tg_corpus_tokens(m2));
    float lb[5];
    if (tg_train_steps(m2, 5, lb) != 0) { fprintf(stderr, "加载后续训失败\n"); return 2; }
    int train_rt_ok = 1;
    for (int i = 0; i < 5; i++) {
        printf("续训步 %d：原模型 loss %.6f，加载后 loss %.6f %s\n",
               i + 1, la[i], lb[i], la[i] == lb[i] ? "一致" : "不一致");
        if (la[i] != lb[i]) train_rt_ok = 0;
    }
    /* 生成一致性：原模型（多训 5 步后）与加载后模型（多训 5 步后）同种子生成 */
    if (tg_generate(m, "小白猫", 60, 0.8f, 40, 42, genA, sizeof(genA)) < 0) {
        fprintf(stderr, "生成失败\n"); return 2;
    }
    char gen2[4096];
    if (tg_generate(m2, "小白猫", 60, 0.8f, 40, 42, gen2, sizeof(gen2)) < 0) {
        fprintf(stderr, "生成失败\n"); return 2;
    }
    int gen_ok = strcmp(genA, gen2) == 0;
    printf("加载后生成：\n%s\n生成一致性：%s\n", gen2, gen_ok ? "通过" : "失败");
    printf("续训一致性：%s\n", train_rt_ok ? "通过" : "失败");

    tg_free(m);
    tg_free(m2);
    free(losses);
    free(corpus);
    int all_ok = loss_ok && gen_ok && train_rt_ok;
    printf("SMOKE %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
