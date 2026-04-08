/*
 * Copyright (c) 2024 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GREEN_ED_BATH_FITTING_H
#define GREEN_ED_BATH_FITTING_H

#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>

#include <complex>
#include <lsqcpp/lsqcpp.hpp>

namespace green::impurity {

  template <size_t N>
  using ztensor = green::ndarray::ndarray<std::complex<double>, N>;
  template <size_t N>
  using dtensor = green::ndarray::ndarray<double, N>;
  template <size_t N>
  using itensor = green::ndarray::ndarray<int, N>;

  template <typename T, size_t D>
  std::array<size_t, D + 1> operator+(const std::array<size_t, D>& a, T b) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin());
    result[D] = size_t(b);
    return result;
  }

  template <typename T, size_t D>
  std::array<size_t, D + 1> operator+(T b, const std::array<size_t, D>& a) {
    std::array<size_t, D + 1> result;
    std::copy(a.begin(), a.end(), result.begin() + 1);
    result[0] = size_t(b);
    return result;
  }

  /**
   * @brief Implementation of the residual estimator for minimization of Hybridization function:
   *                                         V_σik V*_σik
   * res = \sum_{ω} || Δ_{σii}(ω) - \sum_ik ------------- || * f(ω)
   *                                          ω - ε_σk
   * with cost function f(ω) that can enforce importance of certain frequency regions
   *
   */
  struct hybridization_function_error {
    static constexpr bool ComputesJacobian = false;

    /// default constructor
    hybridization_function_error() : _freqs(0), _target_delta(0, 0, 0, 0), _bath_structure(0), _nw(0), _io(0), _is(0) {}

    /**
     * Construct estimator for given frequency grid and hybridization function
     * @param freqs - Matsubara frequencies where hybridization function is defined
     * @param delta - Hybridization function
     * @param bath_structure - structure of the bath for `io`-th orbital
     * @param io - number of current orbital to minimize
     * @param is - current spin channel to minimize
     * @param type - type of cost function to use: 1 - sum of residuals, 2 - max residual, 3 - max residual with no frequency cutoff
     */
    hybridization_function_error(const ztensor<1>& freqs, const ztensor<4>& delta, const itensor<1>& bath_structure, size_t io,
                                 size_t is, int type = 3) :
        _freqs(freqs), _target_delta(delta), _bath_structure(bath_structure), _nw(delta.shape()[0]), _io(io), _is(is),
        _type(type) {}

    /**
     * Calculate residual for a given bath parameters in xval vector and put it into a target function fval
     *
     * @param xval - bath parameters. for each orbital we have a group of bath parameters with the first `nk` values
     * for hybridization strength V_{ik} and second `nk` values for bath energies `ε_k`
     * @param fval - residuals for each orbital
     */
    template <typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1>& xval, Eigen::Matrix<Scalar, Outputs, 1>& fval) const {
      fval.resize(1);
      // size_t nk = (xval.size() / _nio) / 2;
      // size_t ik = xval.size() / _nio;
      double res(0);
      size_t nk = _bath_structure(_io);
      for (size_t iw = 0; iw < _nw; ++iw) {
        if (_type == 1) {
          type_1(res, xval, iw, nk);
        } else if (_type == 2) {
          type_2(res, xval, iw, nk);
        } else if (_type == 3) {
          type_3(res, xval, iw, nk);
        }
      }
      fval(0) = res;  // _target_delta.size();
    }

  private:
    template <typename Scalar, int Inputs>
    void type_1(double& res, const Eigen::Matrix<Scalar, Inputs, 1>& xval, size_t iw, size_t nk) const {
      if (std::abs(_freqs(iw).imag()) > 5.0) return;
      std::complex<double> hyb(0, 0);
      for (size_t i = 0; i < nk; ++i) hyb += (xval(i) * xval(i)) / (_freqs(iw) - xval(nk + i));
      res += std::max(0.0, std::abs(_target_delta(iw, _is, _io, _io) - hyb));
    }

    template <typename Scalar, int Inputs>
    void type_2(double& res, const Eigen::Matrix<Scalar, Inputs, 1>& xval, size_t iw, size_t nk) const {
      if (std::abs(_freqs(iw).imag()) > 5.0) return;
      std::complex<double> hyb(0, 0);
      for (size_t i = 0; i < nk; ++i) hyb += (xval(i) * xval(i)) / (_freqs(iw) - xval(nk + i));
      res = std::max(res, std::abs(_target_delta(iw, _is, _io, _io) - hyb));
    }

    template <typename Scalar, int Inputs>
    void type_3(double& res, const Eigen::Matrix<Scalar, Inputs, 1>& xval, size_t iw, size_t nk) const {
      std::complex<double> hyb(0, 0);
      for (size_t i = 0; i < nk; ++i) hyb += (xval(i) * xval(i)) / (_freqs(iw) - xval(nk + i));
      res = std::max(res, std::abs(_target_delta(iw, _is, _io, _io) - hyb));
    }

    ztensor<1> _freqs;
    ztensor<4> _target_delta;
    itensor<1> _bath_structure;
    size_t     _nw;
    size_t     _io;
    size_t     _is;
    int        _type;
  };
  /**
   * For a given bath patameters evaluate and return hybridization function
   *
   * @param freqs - Matsubara frequency grid
   * @param bath - bath parameters
   * @param ns - number of spins
   * @param nio - number of impurity orbitals
   */
  ztensor<4> compute_hyb_fun(const ztensor<1>& freqs, const dtensor<2>& bath, const itensor<1>& bath_structure, size_t ns,
                             size_t nio);

  /**
   * For a given Hybridization function defiend on Matsubara frequency grid find discrete approximation and
   * corresponding bath parameters using Gauss-Newton method
   *
   * @param freqs Matsubara frequency grid
   * @param hyb_fun Hybridization function on Matsubara frequencies to be minimized
   * @param initial_guess initial guess for bath parameters
   * @param bath_structure 1d array with numbers of bath sites for each orbitals
   * @return Discretized approximation of the Hybridization function and corresponding bath parameters
   */
  std::pair<ztensor<4>, dtensor<2>> minimize(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                             const dtensor<2>& initial_guess, const itensor<1>& bath_structure,
                                             int type = 3);

}  // namespace green::impurity

#endif  // GREEN_ED_BATH_FITTING_H
