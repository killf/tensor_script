#ifndef TENSORSCRIPT_DEVICE_HPP
#define TENSORSCRIPT_DEVICE_HPP

#include <cstdint>
#include <string>
#include <sstream>
#include <unistd.h>

#include "exception.hpp"

namespace ts {

  enum DeviceType {
    CPU = 0,
    CUDA = 1
  };

  typedef int16_t DeviceIndex;

  struct Device {
  public:
    Device(DeviceType type, DeviceIndex index = -1) : type_(type), index_(index) {};

    Device(const std::string &device_string) {
      auto idx = device_string.find(':');
      std::string type_string;
      if (idx == std::string::npos) {
        index_ = -1;
        type_string = device_string;
      } else {
        std::string index_string = device_string.substr(index_ + 1);
        index_ = atoi(index_string.c_str());
        type_string = device_string.substr(0, index_);
      }

      if (type_string == "cpu" || type_string == "CPU") {
        type_ = CPU;
        if (index_ != -1 || index_ != 0)
          throw CustomError("RuntimeError: CPU device index must be -1 or zero, got %d.", index_);
      } else if (type_string == "cuda" || type_string == "CUDA") {
        type_ = CUDA;
        int count = 0;
        CHECK(cudaGetDeviceCount(&count));
        if (index_ != -1 & index_ >= count)
          throw CustomError("RuntimeError: CPU device index must be -1 or between 0 and %d, got %d.", count, index_);
      } else {
        throw CustomError("RuntimeError: Expected one of cpu, cuda device type at start of device string: %s.",
                          type_string.c_str());
      }
    }

    inline bool operator==(const Device &rhs) const {
      return type_ == rhs.type_ &&
             (index_ == rhs.index_ || (index_ == -1 && rhs.index_ == 0) || (index_ == 0 && rhs.index_ == -1));
    }

    inline bool operator!=(const Device &rhs) const {
      return !(*this == rhs);
    }

    inline void set_index(DeviceIndex index) {
      this->index_ = index;
    }

    inline DeviceType type() const noexcept { return type_; }

    inline DeviceIndex index() const noexcept { return index_; };

    inline bool has_index() const noexcept { return index_ != -1; };

    inline bool is_cuda() const noexcept { return type_ == CUDA; }

    inline bool is_cpu() const noexcept { return type_ == CPU; }

    std::string str() const {
      std::stringstream ss;
      if (type_ == CPU) ss << "cpu";
      else if (type_ == CUDA) ss << "cuda";
      else ss << "unknown";

      if (has_index()) ss << ":" << index_;
      return ss.str();
    }

  private:
    DeviceType type_;
    DeviceIndex index_;
  };
}

#endif //TENSORSCRIPT_DEVICE_HPP
