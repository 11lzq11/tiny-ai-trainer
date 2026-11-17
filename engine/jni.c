/*
 * jni.c — JNI 桥接（Kotlin Trainer 类 ↔ C 引擎）
 *
 * 重要约定：Kotlin 侧的类必须是 com.tinytrainer.Trainer，
 * 否则须按实际包名重命名下面的 JNI 函数（Java_<包>_<类>_<方法>）。
 *
 * 线程模型：每个 handle 带一把互斥锁，训练/生成/保存/加载互斥。
 * nativeTrainSteps 是阻塞调用，Kotlin 层应在后台线程以小块（例如 10–50 步）
 * 循环调用，以便生成与 UI 保持响应；生成在训练块之间插空执行。
 *
 * 字符串用 GetStringUTFChars（标准 UTF-8，BMP 内字符无损；增补平面字符
 * 如 emoji 会被当作 <unk>，属已知限制，见 README）。
 *
 * 本文件只依赖 <jni.h> 与标准 C + pthread，可在 Android NDK 下编译。
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "tinygpt.h"

typedef struct {
    TGModel *model;
    pthread_mutex_t lock;
} TGHandle;

static TGHandle *handle_from_jlong(jlong h) { return (TGHandle *)(uintptr_t)h; }

/* ---------------- 生命周期 ---------------- */

JNIEXPORT jlong JNICALL
Java_com_tinytrainer_Trainer_nativeCreate(JNIEnv *env, jobject thiz, jstring configJson) {
    (void)env; (void)thiz;
    const char *json = (*env)->GetStringUTFChars(env, configJson, NULL);
    if (!json) return 0;
    TGModel *model = tg_create_from_json(json);
    (*env)->ReleaseStringUTFChars(env, configJson, json);
    if (!model) return 0;
    TGHandle *h = (TGHandle *)calloc(1, sizeof(TGHandle));
    if (!h) { tg_free(model); return 0; }
    h->model = model;
    pthread_mutex_init(&h->lock, NULL);
    return (jlong)(uintptr_t)h;
}

JNIEXPORT void JNICALL
Java_com_tinytrainer_Trainer_nativeFree(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h) return;
    pthread_mutex_lock(&h->lock);
    tg_free(h->model);
    h->model = NULL;
    pthread_mutex_unlock(&h->lock);
    pthread_mutex_destroy(&h->lock);
    free(h);
}

/* ---------------- 大脑文件 ---------------- */

JNIEXPORT jboolean JNICALL
Java_com_tinytrainer_Trainer_nativeLoadBrain(JNIEnv *env, jobject thiz, jlong handle, jstring path) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h) return JNI_FALSE;
    const char *cpath = (*env)->GetStringUTFChars(env, path, NULL);
    if (!cpath) return JNI_FALSE;
    pthread_mutex_lock(&h->lock);
    TGModel *loaded = tg_load_brain(cpath);
    if (loaded) {
        tg_free(h->model);
        h->model = loaded;
    }
    pthread_mutex_unlock(&h->lock);
    (*env)->ReleaseStringUTFChars(env, path, cpath);
    return loaded ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_tinytrainer_Trainer_nativeSaveBrain(JNIEnv *env, jobject thiz, jlong handle, jstring path) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h) return JNI_FALSE;
    const char *cpath = (*env)->GetStringUTFChars(env, path, NULL);
    if (!cpath) return JNI_FALSE;
    pthread_mutex_lock(&h->lock);
    int rc = h->model ? tg_save_brain(h->model, cpath) : -1;
    pthread_mutex_unlock(&h->lock);
    (*env)->ReleaseStringUTFChars(env, path, cpath);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ---------------- 语料 ---------------- */

JNIEXPORT jboolean JNICALL
Java_com_tinytrainer_Trainer_nativeSetCorpus(JNIEnv *env, jobject thiz, jlong handle, jstring text) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h) return JNI_FALSE;
    const char *ctext = (*env)->GetStringUTFChars(env, text, NULL);
    if (!ctext) return JNI_FALSE;
    pthread_mutex_lock(&h->lock);
    int rc = h->model ? tg_set_corpus(h->model, ctext) : -1;
    pthread_mutex_unlock(&h->lock);
    (*env)->ReleaseStringUTFChars(env, text, ctext);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ---------------- 训练 ---------------- */

JNIEXPORT jfloatArray JNICALL
Java_com_tinytrainer_Trainer_nativeTrainSteps(JNIEnv *env, jobject thiz, jlong handle, jint steps) {
    (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || steps < 1) return NULL;
    float *losses = (float *)malloc(sizeof(float) * (size_t)steps);
    if (!losses) return NULL;
    pthread_mutex_lock(&h->lock);
    int rc = h->model ? tg_train_steps(h->model, (int)steps, losses) : -1;
    pthread_mutex_unlock(&h->lock);
    if (rc != 0) { free(losses); return NULL; }
    jfloatArray out = (*env)->NewFloatArray(env, steps);
    if (!out) { free(losses); return NULL; }
    (*env)->SetFloatArrayRegion(env, out, 0, steps, losses);
    free(losses);
    return out;
}

/* ---------------- 生成 ---------------- */

JNIEXPORT jstring JNICALL
Java_com_tinytrainer_Trainer_nativeGenerate(JNIEnv *env, jobject thiz, jlong handle,
                                           jstring prompt, jint maxTokens,
                                           jfloat temperature, jint topK, jlong seed) {
    (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || maxTokens < 1) return NULL;
    const char *cprompt = (*env)->GetStringUTFChars(env, prompt, NULL);
    if (!cprompt) return NULL;
    /* 每个 token 最多 4 字节 UTF-8 */
    size_t cap = (size_t)maxTokens * 4 + 16;
    char *out = (char *)malloc(cap);
    if (!out) { (*env)->ReleaseStringUTFChars(env, prompt, cprompt); return NULL; }
    pthread_mutex_lock(&h->lock);
    int rc = h->model ? tg_generate(h->model, cprompt, (int)maxTokens,
                                    (float)temperature, (int)topK, (uint64_t)seed,
                                    out, (int)cap) : -1;
    pthread_mutex_unlock(&h->lock);
    (*env)->ReleaseStringUTFChars(env, prompt, cprompt);
    if (rc < 0) { free(out); return NULL; }
    jstring result = (*env)->NewStringUTF(env, out);
    free(out);
    return result;
}

/* ---------------- 状态查询 / 参数 ---------------- */

JNIEXPORT jlong JNICALL
Java_com_tinytrainer_Trainer_nativeGetStep(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || !h->model) return 0;
    pthread_mutex_lock(&h->lock);
    long s = tg_get_step(h->model);
    pthread_mutex_unlock(&h->lock);
    return (jlong)s;
}

JNIEXPORT void JNICALL
Java_com_tinytrainer_Trainer_nativeSetLr(JNIEnv *env, jobject thiz, jlong handle, jfloat lr) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || !h->model) return;
    pthread_mutex_lock(&h->lock);
    tg_set_lr(h->model, (float)lr);
    pthread_mutex_unlock(&h->lock);
}

JNIEXPORT void JNICALL
Java_com_tinytrainer_Trainer_nativeSetBatchSize(JNIEnv *env, jobject thiz, jlong handle, jint batch) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || !h->model) return;
    pthread_mutex_lock(&h->lock);
    tg_set_batch_size(h->model, (int)batch);
    pthread_mutex_unlock(&h->lock);
}

JNIEXPORT jint JNICALL
Java_com_tinytrainer_Trainer_nativeVocabSize(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || !h->model) return 0;
    pthread_mutex_lock(&h->lock);
    int v = tg_vocab_size(h->model);
    pthread_mutex_unlock(&h->lock);
    return (jint)v;
}

JNIEXPORT jlong JNICALL
Java_com_tinytrainer_Trainer_nativeParamCount(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    TGHandle *h = handle_from_jlong(handle);
    if (!h || !h->model) return 0;
    pthread_mutex_lock(&h->lock);
    long n = tg_param_count(h->model);
    pthread_mutex_unlock(&h->lock);
    return (jlong)n;
}
