# TinyTrainer 引擎 CMake（供 Android NDK 与桌面构建共用）
cmake_minimum_required(VERSION 3.18)
project(tinygpt C)

# C99 + O2；NDK 下 -Wall -Wextra 且把警告视为失败由外层控制
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP（矩阵乘并行）：NDK 需 NDK r21+，找不到则退化为串行
find_package(OpenMP)
if(OpenMP_C_FOUND)
  message(STATUS "OpenMP found, enabling multi-threaded matmul")
endif()

add_library(tinygpt STATIC tinygpt.c)
target_include_directories(tinygpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tinygpt PUBLIC m)
if(OpenMP_C_FOUND)
  target_link_libraries(tinygpt PUBLIC OpenMP::OpenMP_C)
endif()
if(ANDROID)
  # JNI 桥接只在 Android 下编进库
  add_library(tinygpt_jni SHARED jni.c)
  target_link_libraries(tinygpt_jni PRIVATE tinygpt)
endif()

# 桌面冒烟测试（仅非 Android）
if(NOT ANDROID)
  add_executable(smoke tests/smoke.c)
  target_link_libraries(smoke PRIVATE tinygpt)
endif()
