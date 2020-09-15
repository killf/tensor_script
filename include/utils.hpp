#ifndef TENSORSCRIPT_UTILS_HPP
#define TENSORSCRIPT_UTILS_HPP

#include <iostream>
#include <string>

namespace ts {

  inline void obj2str(std::ostream &ostream) {}

  template<typename T, typename ...ARGS>
  inline void obj2str(std::ostream &ostream, T value, ARGS...args) {
    ostream << value;
    obj2str(ostream, args...);
  }

  template<typename ...ARGS>
  std::string str(ARGS...args) {
    std::stringstream ss;
    obj2str(ss, args...);
    return ss.str();
  }
}

#endif //TENSORSCRIPT_UTILS_HPP
