#ifndef TENSORSCRIPT_OPS_HPP
#define TENSORSCRIPT_OPS_HPP

#include <string>

#include "tensor.hpp"
#include "device.hpp"

namespace ts {

  struct OperatorDescriptor {
    std::string name;
    DeviceType type;
    void *forward;
    void *backward;

    OperatorDescriptor(std::string name,
                       DeviceType type,
                       void *forward,
                       void *backward = nullptr) : name(std::move(name)),
                                                   type(type),
                                                   forward(forward),
                                                   backward(backward) {}

    bool operator==(const OperatorDescriptor &rhs) const {
      return type == rhs.type && name == rhs.name;
    }

    bool operator!=(const OperatorDescriptor &rhs) const {
      return type != rhs.type || name != rhs.name;
    }

    bool has_backward() const { return backward != nullptr; }
  };

  // name, device, func

}

#endif //TENSORSCRIPT_OPS_HPP
