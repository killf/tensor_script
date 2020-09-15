#ifndef TENSORSCRIPT_TENSOR_HPP
#define TENSORSCRIPT_TENSOR_HPP

#include <iostream>
#include <memory>

#include "memory.hpp"
#include "shape.hpp"
#include "device.hpp"

namespace ts {

  template<class T>
  class Tensor {
  public:
    Tensor() : device_(default_device), data_(nullptr) {};

    explicit Tensor(TensorShape shape) : shape_(std::move(shape)),
                                         device_(default_device),
                                         data_(malloc<T>(shape_.size(), device_),
                                               [&](T *ptr) { free(ptr, device_); }) {}

    Tensor(TensorShape shape, Device device) : shape_(std::move(shape)),
                                               device_(device),
                                               data_(malloc<T>(shape_.size(), device_),
                                                     [&](T *ptr) { free(ptr, device_); }) {}

    Tensor(const Tensor &rhs) : shape_(rhs.shape_),
                                device_(rhs.device_),
                                data_(rhs.data_) {}

    Tensor(Tensor &&rhs) noexcept: shape_(rhs.shape_),
                                   device_(rhs.device_),
                                   data_(rhs.data_) {
      rhs.data_ = nullptr;
    }

    ~Tensor() = default;

    void swap(Tensor &rhs) {
      std::swap(rhs.device_, device_);
      std::swap(rhs.shape_, shape_);
      std::swap(rhs.data_, data_);
    }

    Tensor &operator=(const Tensor &rhs) {
      if (this == &rhs)return *this;

      Tensor tmp(rhs);
      swap(tmp);

      return *this;
    }

  public:
    void reshape(const TensorShape &shape) {
      if (shape.size() != shape_.size()) {
        throw ValueError("cannot reshape tensor of size ", shape_.size(), " into shape ", shape);
      }

      shape_ = shape;
    }

  public:
    Tensor to(const Device &device) {
      if (device == device_) return *this;

      Tensor result(shape_, device);
      memcpy(result.data_.get(), data_.get(), shape_.size(), device, device_);

      return result;
    }

    Tensor cuda() {
      if (is_cuda())return *this;
      if (default_device.is_cuda()) return to(default_device);
      return to(Device(CUDA));
    }

    Tensor cpu() {
      if (is_cpu())return *this;
      if (default_device.is_cpu()) return to(default_device);
      return to(Device(CPU));
    }

    std::string str() {
      std::stringstream ss;
      ss << "Tensor<" << device_.str() << ">" << shape_;
      return ss.str();
    }

  public:

    inline size_t ndim() const { return shape_.ndim(); }

    inline TensorShape shape() const { return shape_; }

    inline std::size_t size() const { return shape_.size(); }

    inline std::size_t size(size_t axis) const { return shape_[axis]; }

    inline Device device() const { return device_; }

    inline bool is_cpu() const noexcept { return device_.is_cpu(); }

    inline bool is_cuda() const noexcept { return device_.is_cuda(); }

    inline const T *data() const { return data_.get(); }

    inline T *data_mutable() { return data_.get(); }

  private:
    Device device_;
    TensorShape shape_;
    std::shared_ptr<T> data_;
  };

  template<typename T>
  Tensor<T> ones(const TensorShape &shape) {
    Tensor<T> tensor(shape);



    return tensor;
  }

}

#endif //TENSORSCRIPT_TENSOR_HPP
