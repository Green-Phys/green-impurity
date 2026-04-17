/*
 * Copyright (c) 2024 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GREEN_ED_BATH_FITTING_H
#define GREEN_ED_BATH_FITTING_H

#include <lsqcpp/lsqcpp.hpp>

#include "common_defs.h"

namespace green::impurity {

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
   * @brief Scalar cost function for minimization of Hybridization function (NORM_L1 or NORM_LINF).
   *
   * Computes a single scalar residual:
   *   NORM_L1:   sum of |residual(ω)| over active frequencies
   *   NORM_LINF: max |residual(ω)| over active frequencies
   *
   * Frequencies with |ω_imag| > freq_cutoff are excluded when freq_cutoff > 0.
   * Uses numerical Jacobian (finite differences).
   */
  struct hybridization_function_error_scalar {
    static constexpr bool ComputesJacobian = false;

    hybridization_function_error_scalar() : _freqs(0), _target_delta(0, 0, 0, 0), _bath_structure(0), _nw(0), _io(0), _is(0) {}

    hybridization_function_error_scalar(const ztensor<1>& freqs, const ztensor<4>& delta, const itensor<1>& bath_structure,
                                        size_t io, size_t is, bath_fitting_method method = NORM_LINF,
                                        double freq_cutoff = -1.0) :
        _freqs(freqs), _target_delta(delta), _bath_structure(bath_structure), _nw(delta.shape()[0]), _io(io), _is(is),
        _method(method), _freq_cutoff(freq_cutoff) {}

    template <typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1>& xval, Eigen::Matrix<Scalar, Outputs, 1>& fval) const {
      fval.resize(1);
      double res(0);
      size_t nk = _bath_structure(_io);
      for (size_t iw = 0; iw < _nw; ++iw) {
        if (_freq_cutoff > 0 && std::abs(_freqs(iw).imag()) > _freq_cutoff) continue;
        std::complex<double> hyb(0, 0);
        for (size_t i = 0; i < nk; ++i) hyb += (xval(i) * xval(i)) / (_freqs(iw) - xval(nk + i));
        double r = std::abs(_target_delta(iw, _is, _io, _io) - hyb);
        if (_method == NORM_L1) {
          res += r;
        } else {
          res = std::max(res, r);
        }
      }
      fval(0) = res;
    }

  private:
    ztensor<1>          _freqs;
    ztensor<4>          _target_delta;
    itensor<1>          _bath_structure;
    size_t              _nw;
    size_t              _io;
    size_t              _is;
    bath_fitting_method _method;
    double              _freq_cutoff;
  };

  /**
   * @brief Per-frequency least-squares residual with analytical Jacobian (NORM_L2_TRAPZ):
   *                                         V_σik V*_σik
   * res = \sum_{ω} w(ω) || Δ_{σii}(ω) - \sum_ik ------------- ||²
   *                                          ω - ε_σk
   * with trapezoidal frequency weighting w(ω).
   *
   * Frequencies with |ω_imag| > freq_cutoff are excluded when freq_cutoff > 0.
   * Returns 2*nw_active residuals (real and imaginary parts) with exact Jacobian,
   * enabling faster and more reliable Gauss-Newton convergence.
   */
  struct hybridization_function_error_jacobian {
    static constexpr bool ComputesJacobian = true;

    hybridization_function_error_jacobian() :
        _freqs(0), _target_delta(0, 0, 0, 0), _bath_structure(0), _nw(0), _io(0), _is(0) {}

    hybridization_function_error_jacobian(const ztensor<1>& freqs, const ztensor<4>& delta, const itensor<1>& bath_structure,
                                          size_t io, size_t is, double freq_cutoff = -1.0) :
        _freqs(freqs), _target_delta(delta), _bath_structure(bath_structure), _nw(delta.shape()[0]), _io(io), _is(is),
        _freq_cutoff(freq_cutoff) {}

    template <typename Scalar, int Inputs, typename JacType>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1>& xval, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& fval,
                    JacType& jacobian) const {
      size_t nk = _bath_structure(_io);

      // Count active frequencies
      size_t nw_active = 0;
      for (size_t iw = 0; iw < _nw; ++iw) {
        if (_freq_cutoff > 0 && std::abs(_freqs(iw).imag()) > _freq_cutoff) continue;
        nw_active++;
      }

      fval.resize(2 * nw_active);
      jacobian.resize(2 * nw_active, xval.size());
      jacobian.setZero();

      size_t res_idx = 0;
      for (size_t iw = 0; iw < _nw; ++iw) {
        if (_freq_cutoff > 0 && std::abs(_freqs(iw).imag()) > _freq_cutoff) continue;

        std::complex<double>                hyb(0, 0);
        std::vector<std::complex<double>> dhydb_dv(nk);
        std::vector<std::complex<double>> dhydb_de(nk);

        for (size_t i = 0; i < nk; ++i) {
          std::complex<double> denom = _freqs(iw) - xval(nk + i);
          hyb += (xval(i) * xval(i)) / denom;
          dhydb_dv[i] = 2.0 * xval(i) / denom;
          dhydb_de[i] = (xval(i) * xval(i)) / (denom * denom);
        }

        // Trapezoidal frequency weight
        double weight = 1.0;
        if (iw > 0 && iw < _nw - 1) {
          double dw_left  = std::abs(_freqs(iw).imag() - _freqs(iw - 1).imag());
          double dw_right = std::abs(_freqs(iw + 1).imag() - _freqs(iw).imag());
          weight = (dw_left + dw_right) / 2.0;
        } else if (iw == 0 && _nw > 1) {
          weight = std::abs(_freqs(1).imag() - _freqs(0).imag()) / 2.0;
        } else if (iw == _nw - 1 && _nw > 1) {
          weight = std::abs(_freqs(_nw - 1).imag() - _freqs(_nw - 2).imag()) / 2.0;
        }

        double sqrt_weight = std::sqrt(weight);

        std::complex<double> target = _target_delta(iw, _is, _io, _io);
        std::complex<double> diff   = target - hyb;

        fval(res_idx)     = sqrt_weight * diff.real();
        fval(res_idx + 1) = sqrt_weight * diff.imag();

        for (size_t i = 0; i < nk; ++i) {
          jacobian(res_idx, i)     = -sqrt_weight * dhydb_dv[i].real();
          jacobian(res_idx + 1, i) = -sqrt_weight * dhydb_dv[i].imag();
        }

        for (size_t i = 0; i < nk; ++i) {
          jacobian(res_idx, nk + i)     = -sqrt_weight * dhydb_de[i].real();
          jacobian(res_idx + 1, nk + i) = -sqrt_weight * dhydb_de[i].imag();
        }

        res_idx += 2;
      }
    }

    template <typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1>& xval, Eigen::Matrix<Scalar, Outputs, 1>& fval) const {
      fval.resize(1);
      fval(0) = 0.0;
    }

  private:
    ztensor<1> _freqs;
    ztensor<4> _target_delta;
    itensor<1> _bath_structure;
    size_t     _nw;
    size_t     _io;
    size_t     _is;
    double     _freq_cutoff;
  };

  /**
   * For a given bath parameters evaluate and return hybridization function
   */
  ztensor<4> compute_hyb_fun(const ztensor<1>& freqs, const dtensor<2>& bath, const itensor<1>& bath_structure, size_t ns,
                             size_t nio);

  /**
   * For a given Hybridization function defined on Matsubara frequency grid find discrete approximation and
   * corresponding bath parameters using Gauss-Newton method
   *
   * @param freqs Matsubara frequency grid
   * @param hyb_fun Hybridization function on Matsubara frequencies to be minimized
   * @param initial_guess initial guess for bath parameters
   * @param bath_structure 1d array with numbers of bath sites for each orbitals
   * @param freq_cutoff frequency cutoff; frequencies with |ω_imag| > cutoff are excluded. Negative disables cutoff.
   * @param method residual norm: NORM_L2_TRAPZ (default), NORM_L1, or NORM_LINF
   * @return Discretized approximation of the Hybridization function and corresponding bath parameters
   */
  std::pair<ztensor<4>, dtensor<2>> minimize(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                             const dtensor<2>& initial_guess, const itensor<1>& bath_structure,
                                             double freq_cutoff = -1.0,
                                             bath_fitting_method method = NORM_L2_TRAPZ);

}  // namespace green::impurity

#endif  // GREEN_ED_BATH_FITTING_H
