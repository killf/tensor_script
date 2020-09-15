#ifndef TENSORSCRIPT_ALLOC_HPP
#define TENSORSCRIPT_ALLOC_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include "device.hpp"
#include "exception.hpp"


namespace ts {

  template<typename T>
  T *malloc(std::size_t num, const Device &device) {
    T *ptr = nullptr;
    if (device.is_cpu()) {
      CHECK(cudaMallocHost(&ptr, sizeof(T) * num));
    } else if (device.is_cuda()) {
      if (device != default_device) cudaSetDevice(device.index());
      CHECK(cudaMalloc(&ptr, sizeof(T) * num));
      if (device != default_device) cudaSetDevice(default_device.index());
    }

    std::cout << "malloc: " << (void *) ptr << std::endl;
    return ptr;
  }

  template<typename T>
  void free(T *ptr, const Device &device) {
    if (device.is_cpu()) {
      CHECK(cudaFreeHost(ptr));
    } else if (device.is_cuda()) {
      CHECK(cudaFree(ptr));
    }

    std::cout << "free: " << (void *) ptr << std::endl;
  }

  template<typename T>
  void memcpy(T *dst, T *src, std::size_t num, const Device &dst_dev, const Device &src_dev) {
    if (dst_dev.is_cpu() && src_dev.is_cpu()) {
      CHECK(cudaMemcpy(dst, src, sizeof(T) * num, cudaMemcpyHostToHost));
    } else if (dst_dev.is_cuda() && src_dev.is_cpu()) {
      CHECK(cudaMemcpy(dst, src, sizeof(T) * num, cudaMemcpyHostToDevice));
    } else if (dst_dev.is_cpu() && src_dev.is_cuda()) {
      CHECK(cudaMemcpy(dst, src, sizeof(T) * num, cudaMemcpyDeviceToHost));
    } else {
      CHECK(cudaMemcpy(dst, src, sizeof(T) * num, cudaMemcpyDeviceToDevice));
    }
  }
}

#endif //TENSORSCRIPT_ALLOC_HPP
