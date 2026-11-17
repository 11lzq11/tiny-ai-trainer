/*
 * jni_stub.h — 仅用于桌面语法检查的最小 JNI 存根。
 * ！！！这不是真正的 jni.h，仅声明 jni.c 用到的几个函数，
 * 用于在没有 Android NDK/JDK 的机器上做 `gcc -c` 语法检查。
 * 真实构建（NDK）时用 NDK 自带的 jni.h，本文件不会被包含。
 */
#ifndef TINYGPT_JNI_STUB_H
#define TINYGPT_JNI_STUB_H

#ifdef __cplusplus
extern "C" {
#endif

#define JNIEXPORT
#define JNICALL
#define JNI_TRUE 1
#define JNI_FALSE 0

typedef unsigned char jboolean;
typedef signed char jbyte;
typedef unsigned short jchar;
typedef short jshort;
typedef int jint;
typedef long long jlong;
typedef float jfloat;
typedef double jdouble;
typedef int jsize;

typedef void *_jobject;
typedef _jobject jobject;
typedef _jobject jclass;
typedef _jobject jstring;
typedef _jobject jarray;
typedef _jobject jfloatArray;

struct JNINativeInterface_;
typedef const struct JNINativeInterface_ *JNIEnv;

struct JNINativeInterface_ {
    void *reserved0;
    void *reserved1;
    void *reserved2;
    void *reserved3;
    const char *(JNICALL *GetStringUTFChars)(JNIEnv *, jstring, jboolean *);
    void (JNICALL *ReleaseStringUTFChars)(JNIEnv *, jstring, const char *);
    jstring(JNICALL *NewStringUTF)(JNIEnv *, const char *);
    jfloatArray(JNICALL *NewFloatArray)(JNIEnv *, jsize);
    void (JNICALL *SetFloatArrayRegion)(JNIEnv *, jfloatArray, jsize, jsize, const jfloat *);
};

#ifdef __cplusplus
}
#endif

#endif /* TINYGPT_JNI_STUB_H */
