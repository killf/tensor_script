#include "tensor/tensor.hpp"

using namespace ts;

int main(int argc, char **argv) {
  DeviceGuard deviceGuard("cuda");

  Tensor<float> a({3, 4});
  auto b = a;

  b = a.cuda();

  std::cout << a.str() << std::endl;
  std::cout << b.str() << std::endl;

  return 0;
}