#ifndef GREEN_IMPURITY_COMMON_DEFS_H
#define GREEN_IMPURITY_COMMON_DEFS_H

#include <complex>
#include <string>
#include <type_traits>

#include <Eigen/Dense>
#include "except.h"

#include <green/grids/transformer_t.h>
#include <green/ndarray/ndarray_math.h>
#include <green/symmetry/symmetry.h>

namespace green::impurity {

  enum class impurity_solver_type { ED, INCHWORM, GW };

  inline impurity_solver_type parse_impurity_solver_type(const std::string& s) {
    if (s == "ED") return impurity_solver_type::ED;
    if (s == "INCHWORM") return impurity_solver_type::INCHWORM;
    if (s == "GW") return impurity_solver_type::GW;
    throw incorr_impurity_solver_type("Unknown impurity_solver type '" + s +
                                      "'. Valid options are: ED, INCHWORM, GW.");
  }

  template <size_t N>
  using ztensor = ndarray::ndarray<std::complex<double>, N>;
  template <size_t N>
  using dtensor = ndarray::ndarray<double, N>;
  template <size_t N>
  using itensor    = ndarray::ndarray<int, N>;
  using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  template <typename prec>
  using MMatrixX   = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcd = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXd  = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  template <typename prec>
  using CMMatrixX   = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXd  = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(ndarray::ndarray<prec, 2>&& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const ndarray::ndarray<const prec, 2>& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(ndarray::ndarray<const prec, 2>&& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const ndarray::ndarray<prec, 2>& array) {
    return CMMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }
}

#endif // GREEN_IMPURITY_COMMON_DEFS_H