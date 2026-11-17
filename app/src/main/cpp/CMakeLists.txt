# TinyTrainer native library (libtinytrainer.so).
#
# The C training engine lives in <repo>/engine/ and is owned by the engine
# task (do NOT edit it from the Android side). The engine sources below are
# listed explicitly — if the engine task renames/moves them, update this list.
#
# Engine files compiled here:
#   engine/tinygpt.c  — hand-written Transformer (forward/backward/AdamW)
#   engine/jni.c      — JNI bridge for com.tinytrainer.Trainer
# Deliberately NOT compiled: engine/tests/* (desktop smoke test, has its own
# main()) and engine/CMakeLists.txt (standalone/desktop build definition).
#
# JNI CONTRACT (must match the Kotlin side exactly):
#   Kotlin class: com.tinytrainer.Trainer
#   Library name: "tinytrainer"  (System.loadLibrary("tinytrainer"))
#   The engine provides these JNI entry points (verified in engine/jni.c):
#     Java_com_tinytrainer_Trainer_nativeCreate
#     Java_com_tinytrainer_Trainer_nativeFree
#     Java_com_tinytrainer_Trainer_nativeLoadBrain
#     Java_com_tinytrainer_Trainer_nativeSaveBrain
#     Java_com_tinytrainer_Trainer_nativeSetCorpus
#     Java_com_tinytrainer_Trainer_nativeTrainSteps
#     Java_com_tinytrainer_Trainer_nativeGenerate
#     Java_com_tinytrainer_Trainer_nativeGetStep
#     Java_com_tinytrainer_Trainer_nativeSetLr
#     Java_com_tinytrainer_Trainer_nativeSetBatchSize
#     Java_com_tinytrainer_Trainer_nativeVocabSize
#     Java_com_tinytrainer_Trainer_nativeParamCount
#
# The engine's training code uses OpenMP (#pragma omp) for matmuls; the
# flags below enable it. If the engine is built without -fopenmp the pragmas
# are ignored and it degrades to serial (still correct, just slower).

cmake_minimum_required(VERSION 3.18.1)

project("tinytrainer")

set(ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine")

set(ENGINE_SOURCES
    "${ENGINE_DIR}/tinygpt.c"
    "${ENGINE_DIR}/jni.c"
)

foreach(src ${ENGINE_SOURCES})
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR
            "TinyTrainer: missing engine source: ${src}. "
            "The engine task must land engine/tinygpt.c and engine/jni.c.")
    endif()
endforeach()
message(STATUS "TinyTrainer engine sources: ${ENGINE_SOURCES}")

add_library(tinytrainer SHARED ${ENGINE_SOURCES})

# The engine is written as C99 (see engine/CMakeLists.txt).
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

target_include_directories(tinytrainer PRIVATE "${ENGINE_DIR}")

# -O2: release-grade optimisation for the training loop.
# -fopenmp: multi-threaded matmuls (at least 4 cores on a Snapdragon 6).
# -Wall -Wextra: keep the engine warning-clean.
target_compile_options(tinytrainer PRIVATE
    -O2
    -Wall
    -Wextra
    -fopenmp
)

# -fopenmp must ALSO be a link option: it is what makes Clang link the
# OpenMP runtime (libomp) at all. -static-openmp alone only says "when
# you link OpenMP, use the static lib" — without -fopenmp at link time
# nothing is linked and the __kmpc_* symbols stay undefined (this broke
# CI: ld.lld "undefined symbol: __kmpc_global_thread_num" etc.).
# Statically link the OpenMP runtime so the .so has no extra .so dependency.
target_link_options(tinytrainer PRIVATE -fopenmp -static-openmp)

find_library(log-lib log)

target_link_libraries(tinytrainer PRIVATE
    android
    ${log-lib}
    m
)
