# Keep the JNI bridge class and its native methods (release builds only;
# debug builds do not minify, but keep this file valid for both).
-keep class com.tinytrainer.Trainer {
    <methods>;
}
