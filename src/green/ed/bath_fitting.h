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

namespace green::ed {

  template <size_t N>
  using ztensor = green::ndarray::ndarray<std::complex<double>, N>;
  template <size_t N>
  using dtensor = green::ndarray::ndarray<double, N>;

  /**
   * @brief Implementation of the residual estimator for minimization of Hybridization function:
   *                                       V_ik V*_ik
   * res = \sum_{ω} || Δ_{ii}(ω) - \sum_ik ----------- || * f(ω)
   *                                        ω - ε_k
   * with cost function f(ω) that can enforce importance of certain frequency regions
   *
   */
  struct hybridization_function_error {
    static constexpr bool ComputesJacobian = false;

    /// default constructor
    hybridization_function_error() : _freqs(0), _target_delta(0, 0, 0, 0), _nw(0), _ns(0), _nio(0) {}

    /**
     * Construct estimator for given frequency grid and hybridization function
     * @param freqs - Matsubara frequencies where hybridization function is defined
     * @param delta - Hybridization function
     */
    hybridization_function_error(const ztensor<1>& freqs, const ztensor<4>& delta) :
        _freqs(freqs), _target_delta(delta), _nw(delta.shape()[0]), _ns(delta.shape()[1]), _nio(delta.shape()[2]) {}

    /**
     * Calculate residual for a given bath parameters in xval vector and put it into a target function fval
     *
     * @param xval - bath parameters. for each orbital we have a group of bath parameters with the first `nk` values
     * for hybridization strength V_{ik} and second `nk` values for bath energies `ε_k`
     * @param fval - residuals for each orbital
     */
    template <typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1>& xval, Eigen::Matrix<Scalar, Outputs, 1>& fval) const {
      fval.resize(_nio);
      size_t nk = (xval.size() / _nio) / 2;
      size_t ik = xval.size() / _nio;
      double res(0);
      for (size_t io = 0; io < _nio; ++io) {
        for (size_t iw = 0; iw < _nw; ++iw) {
          for (size_t is = 0; is < _ns; ++is) {
            std::complex<double> hyb(0, 0);
            for (size_t i = 0; i < nk; ++i)
              hyb += (xval(ik * io + i) * xval(ik * io + i)) / (_freqs(iw) - xval(ik * io + nk + i));
            res += std::abs(_target_delta(iw, is, io, io) - hyb);
          }
        }
        fval(io) = res / _target_delta.size();
      }
    }

  private:
    ztensor<1> _freqs;
    ztensor<4> _target_delta;
    size_t     _nw;
    size_t     _ns;
    size_t     _nio;
  };
  /**
   * For a given bath patameters evaluate and return hybridization function
   *
   * @param freqs - Matsubara frequency grid
   * @param bath - bath parameters
   * @param ns - number of spins
   * @param nio - number of impurity orbitals
   */
  ztensor<4> compute_hyb_fun(const ztensor<1>& freqs, const dtensor<1>& bath, size_t ns, size_t nio) {
    ztensor<4> hyb(freqs.size(), ns, nio, nio);
    size_t     nk = (bath.size() / nio) / 2;
    size_t     ik = bath.size() / nio;
    for (size_t iw = 0; iw < freqs.size(); ++iw) {
      for (size_t is = 0; is < ns; ++is) {
        for (size_t io = 0; io < nio; ++io) {
          for (size_t i = 0; i < nk; ++i)
            hyb(iw, is, io, io) += (bath(ik * io + i) * bath(ik * io + i)) / (freqs(iw) - bath(ik * io + nk + i));
        }
      }
    }
    return hyb;
  }

  std::pair<ztensor<4>, dtensor<1>> minimize(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                             const dtensor<1>& initial_guess) {
    // There are DenseSVDSolver and DenseCholeskySolver available.
    lsqcpp::GaussNewtonX<double, hybridization_function_error, lsqcpp::ArmijoBacktracking> optimizer;
    // lsqcpp::LevenbergMarquardtX<double, HybridizationFunctionError/*, lsqcpp::ArmijoBacktracking*/> optimizer;
    // lsqcpp::GradientDescentX<double, HybridizationFunctionError, lsqcpp::ArmijoBacktracking> optimizer;

    optimizer.setObjective(hybridization_function_error(freqs, hyb_fun));

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaximumIterations(10000);
    // Set the minimum length of the gradient.
    // The optimizer stops minimizing if the gradient length falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 1e-9).
    optimizer.setMinimumGradientLength(1e-8);
    // Set the minimum length of the step.
    // The optimizer stops minimizing if the step length falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 1e-9).
    optimizer.setMinimumStepLength(1e-9);

    // Set the minimum least squares error.
    // The optimizer stops minimizing if the error falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 0).
    optimizer.setMinimumError(1e-14);

    // Set the parameters of the step refiner (Armijo Backtracking).
    optimizer.setRefinementParameters({0.8, 1e-4, 1e-10, 1.0, 0});

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbosity(0);

    // Set initial guess.
    Eigen::VectorXd initialGuess(initial_guess.size());
    std::copy(initial_guess.begin(), initial_guess.end(), initialGuess.data());

    // Start the optimization.
    auto result = optimizer.minimize(initialGuess);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false") << " Iterations: " << result.iterations
              << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval.transpose() << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;
    dtensor<1> res(result.xval.size());
    std::copy(result.xval.data(), result.xval.data() + result.xval.size(), res.begin());
    compute_hyb_fun(freqs, res, hyb_fun.shape()[1], hyb_fun.shape()[2]);
    return std::make_pair(compute_hyb_fun(freqs, res, hyb_fun.shape()[1], hyb_fun.shape()[2]), res);
  }

}  // namespace green::ed

#endif  // GREEN_ED_BATH_FITTING_H
