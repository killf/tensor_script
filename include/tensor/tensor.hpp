#ifndef TENSORSCRIPT_TENSOR_HPP
#define TENSORSCRIPT_TENSOR_HPP

#include "tensor/device.hpp"

namespace ts {

  template<class T>
  class Tensor {

  private:
    Device device_;
    T *data_;
  };

}

#endif //TENSORSCRIPT_TENSOR_HPP
