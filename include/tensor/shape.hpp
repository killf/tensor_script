#ifndef TENSORSCRIPT_SHAPE_HPP
#define TENSORSCRIPT_SHAPE_HPP

#include <vector>
#include <initializer_list>
#include <string>
#include <sstream>

#include "exception.hpp"

namespace ts {

  class TensorShape {
  public:
    TensorShape() = default;

    TensorShape(const std::initializer_list<std::size_t> &list) : _dims(list) {
      for (auto i:_dims)_size *= i;
    }

    inline std::size_t size() const { return _size; }

    inline std::size_t ndim() const { return _dims.size(); }

    inline std::size_t operator[](std::size_t index) const { return _dims[index]; }

    std::size_t index(const std::initializer_list<std::size_t> &index) const {
      std::vector<size_t> index_v(index);
      check_index(index_v);

      std::size_t ind = 0;
      for (int i = 0; i < _dims.size(); i++) {
        std::size_t step = index_v[i];
        for (int j = i + 1; j < _dims.size(); j++) step *= _dims[j];
        ind += step;
      }

      return ind;
    }

    inline std::string str() const {
      std::stringstream ss;

      ss << "(";
      if (ndim() > 0) ss << _dims[0];
      for (int i = 1; i < ndim(); i++)ss << "," << _dims[i];
      ss << ")";

      return ss.str();
    }

  private:
    void check_index(const std::vector<std::size_t> &index) const {
      if (index.size() > ndim()) {
        throw CustomError("IndexError：too many indices for array");
      } else if (index.size() < ndim()) {
        throw CustomError("IndexError：暂不支持切片");
      }

      for (int i = 0; i < _dims.size(); i++) {
        if (index[i] > _dims[i]) {
          throw CustomError("IndexError：index %d is out of bounds for axis %d with size %d", index[i], i, _dims[i]);
        }
      }
    }

  private:
    std::vector<std::size_t> _dims;
    int _size = 1;
  };

  inline std::ostream &operator<<(std::ostream &stream, const TensorShape &obj) { return stream << obj.str(); }

}

#endif //TENSORSCRIPT_SHAPE_HPP
