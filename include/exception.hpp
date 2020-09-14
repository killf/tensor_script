#ifndef TENSORSCRIPT_EXCEPTION_HPP
#define TENSORSCRIPT_EXCEPTION_HPP

#include <exception>
#include <string>
#include <cstdarg>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(exp) \
if (exp != cudaSuccess) { \
    auto err = cudaGetLastError(); \
    throw ts::CustomError("[ERROR] [%s:%d] %s(%d).\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
}

namespace ts {

  class CustomError : public std::exception {
  public:
    explicit CustomError(std::string msg) : msg_(std::move(msg)) {}

    explicit CustomError(const char *__restrict format, ...) {
      static char buf[512];
      int max_len = sizeof(buf);

      va_list ap;
      va_start(ap, format);
      int len = vsnprintf(buf, max_len - 1, format, ap);
      va_end(ap);

      if (len < max_len) {
          msg_ = std::string(buf);
      } else {
        auto str = new char[len + 2];

        va_start(ap, format);
        vsprintf(str, format, ap);
        va_end(ap);

          msg_ = std::string(str);
        delete[] str;
      }
    }

    const char *what() const noexcept override {
      return msg_.c_str();
    }

  private:
    std::string msg_;
  };

}

#endif //TENSORSCRIPT_EXCEPTION_HPP
