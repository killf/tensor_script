
#include "exception.hpp"
#include "tensor/tensor.hpp"
#include "tensor/device.hpp"

int main(int argc, char **argv) {

  printf("%s\n",typeid(ts::CPU).name());

  return 0;
}