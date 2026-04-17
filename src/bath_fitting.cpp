#include <green/impurity/bath_fitting.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace green::impurity {

  /**
   * Improve initial guess by analyzing the hybridization function structure.
   * Used by the NORM_L2_TRAPZ method.
   */
  void improve_initial_guess(const ztensor<1>& freqs, const ztensor<4>& hyb_fun, dtensor<2>& guess,
                             const itensor<1>& bath_structure, size_t is, size_t io) {
    size_t nk = bath_structure(io);
    size_t shift = 0;
    for (size_t j = 0; j < io; ++j) shift += bath_structure(j) * 2;

    // Estimate V scale from hybridization magnitude in middle frequency range
    double v_mag = 0.0;
    size_t count = 0;
    for (size_t iw = freqs.size() / 4; iw < 3 * freqs.size() / 4 && iw < freqs.size(); ++iw) {
      if (std::abs(freqs(iw).imag()) <= 5.0) {
        v_mag += std::abs(hyb_fun(iw, is, io, io).imag());
        count++;
      }
    }
    if (count > 0) v_mag /= count;
    double v_scale = std::sqrt(std::max(v_mag / nk, 1e-3));
    v_scale = std::clamp(v_scale, 0.15, 2.0);

    // Set V parameters with scaling
    for (size_t i = 0; i < nk; ++i) {
      guess(is, shift + i) = v_scale * (0.3 + 0.5 * i / std::max(nk - 1.0, 1.0));
    }

    // Distribute energy parameters across typical range based on frequency extent
    for (size_t i = 0; i < nk; ++i) {
      double frac = (nk > 1) ? static_cast<double>(i) / (nk - 1) : 0.5;
      guess(is, shift + nk + i) = -2.0 + 4.0 * frac;
    }
  }

  ztensor<4> compute_hyb_fun(const ztensor<1>& freqs, const dtensor<2>& bath, const itensor<1>& bath_structure, size_t ns,
                              size_t nio) {
    ztensor<4> hyb(freqs.size(), ns, nio, nio);
    hyb.set_zero();
    for (size_t iw = 0; iw < freqs.size(); ++iw) {
      for (size_t is = 0; is < ns; ++is) {
        size_t shift = 0;
        for (size_t io = 0; io < nio; ++io) {
          size_t nk = bath_structure(io);
          size_t ik = bath_structure(io) * 2;
          for (size_t i = 0; i < nk; ++i)
            hyb(iw, is, io, io) += (bath(is, shift + i) * bath(is, shift + i)) / (freqs(iw) - bath(is, shift + nk + i));
          shift += ik;
        }
      }
    }
    return hyb;
  }

  /**
   * Scalar cost function approach (NORM_L1 or NORM_LINF).
   * Uses numerical Jacobian with a single scalar residual.
   */
  static std::pair<ztensor<4>, dtensor<2>> minimize_scalar(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                                            const dtensor<2>& initial_guess,
                                                            const itensor<1>& bath_structure,
                                                            double freq_cutoff, bath_fitting_method method) {
    size_t     ns = hyb_fun.shape()[1];
    dtensor<2> res(ns, std::reduce(bath_structure.begin(), bath_structure.end()) * 2);
    for (size_t is = 0; is < hyb_fun.shape()[1]; ++is) {
      std::cout << "spin " << is << std::endl;
      size_t shift = 0;
      for (size_t io = 0; io < hyb_fun.shape()[3]; ++io) {
        std::cout << "orbital " << io << std::endl;
        lsqcpp::GaussNewtonX<double, hybridization_function_error_scalar, lsqcpp::DoglegMethod> optimizer;
        optimizer.setMaximumIterations(40000);
        optimizer.setMinimumGradientLength(1e-8);
        optimizer.setMinimumStepLength(1e-9);
        optimizer.setMinimumError(1e-14);
        optimizer.setRefinementParameters({1.0, 3.0, 1e-6, 0.001, 100});
        optimizer.setVerbosity(0);
        size_t                              ik = bath_structure(io) * 2;
        hybridization_function_error_scalar function_error(freqs, hyb_fun, bath_structure, io, is, method, freq_cutoff);
        optimizer.setObjective(function_error);
        Eigen::VectorXd initialGuess(ik);
        for (size_t i = 0; i < ik; ++i) initialGuess(i) = initial_guess(is, i + shift);
        std::cout << "Initial guess: " << initialGuess.transpose() << std::endl;
        auto result = optimizer.minimize(initialGuess);
        std::cout << "Done! Converged: " << (result.converged ? "true" : "false") << " Iterations: " << result.iterations
                  << std::endl;
        std::cout << "Final fval: " << result.fval.transpose() << std::endl;
        std::cout << "Final xval: " << result.xval.transpose() << std::endl;
        std::copy(result.xval.data(), result.xval.data() + result.xval.size(), res(is).begin() + shift);
        std::transform(res(is).begin() + shift, res(is).begin() + shift + bath_structure(io), res(is).begin() + shift,
                       [](double x) { return std::abs(x); });
        shift += ik;
      }
    }
    return std::make_pair(compute_hyb_fun(freqs, res, bath_structure, hyb_fun.shape()[1], hyb_fun.shape()[2]), res);
  }

  /**
   * Per-frequency least-squares approach with analytical Jacobian (NORM_L2_TRAPZ).
   * Uses improved initial guess and trapezoidal frequency weighting.
   */
  static std::pair<ztensor<4>, dtensor<2>> minimize_jacobian(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                                              const dtensor<2>& initial_guess,
                                                              const itensor<1>& bath_structure, double freq_cutoff) {
    size_t     ns  = hyb_fun.shape()[1];
    size_t     nio = hyb_fun.shape()[3];
    dtensor<2> res(ns, std::reduce(bath_structure.begin(), bath_structure.end()) * 2);
    std::cout << "Bath fitting (NORM_L2_TRAPZ): " << ns << " spin(s), " << nio << " orbital(s)";
    if (freq_cutoff > 0)
      std::cout << ", freq_cutoff = " << freq_cutoff;
    else
      std::cout << ", no freq cutoff";
    std::cout << std::endl;
    for (size_t is = 0; is < ns; ++is) {
      std::cout << "\tSpin " << is << ":" << std::endl;
      size_t shift = 0;
      for (size_t io = 0; io < nio; ++io) {
        std::cout << "\t\tOrbital " << io << " (" << bath_structure(io) << " bath sites):" << std::endl;
        lsqcpp::GaussNewtonX<double, hybridization_function_error_jacobian, lsqcpp::DoglegMethod> optimizer;
        optimizer.setMaximumIterations(10000);
        optimizer.setMinimumGradientLength(1e-12);
        optimizer.setMinimumStepLength(1e-14);
        optimizer.setMinimumError(1e-14);
        optimizer.setRefinementParameters({1.0, 3.0, 1e-10, 0.00001, 100});
        optimizer.setVerbosity(0);
        size_t                                ik = bath_structure(io) * 2;
        hybridization_function_error_jacobian function_error(freqs, hyb_fun, bath_structure, io, is, freq_cutoff);
        optimizer.setObjective(function_error);

        // Improve initial guess from hybridization function structure
        dtensor<2> improved_guess = initial_guess;
        improve_initial_guess(freqs, hyb_fun, improved_guess, bath_structure, is, io);

        Eigen::VectorXd initialGuess(ik);
        for (size_t i = 0; i < ik; ++i) initialGuess(i) = improved_guess(is, i + shift);
        std::cout << "\t\t\tInitial guess: " << initialGuess.transpose() << std::endl;
        auto result = optimizer.minimize(initialGuess);

        double fval_abs_min = result.fval.cwiseAbs().minCoeff();
        double fval_abs_max = result.fval.cwiseAbs().maxCoeff();
        double fval_abs_avg = result.fval.cwiseAbs().mean();

        std::cout << "\t\t\tConvergence: " << (result.converged ? "yes" : "NO") << " (" << result.iterations << " iterations)" << std::endl;
        std::cout << "\t\t\t|Delta_fit - Delta| (min/max/avg): " << fval_abs_min << " / " << fval_abs_max << " / " << fval_abs_avg << std::endl;
        std::cout << "\t\t\tBath params: " << result.xval.transpose() << std::endl;
        std::copy(result.xval.data(), result.xval.data() + result.xval.size(), res(is).begin() + shift);
        std::transform(res(is).begin() + shift, res(is).begin() + shift + bath_structure(io), res(is).begin() + shift,
                       [](double x) { return std::abs(x); });
        shift += ik;
      }
    }
    return std::make_pair(compute_hyb_fun(freqs, res, bath_structure, ns, nio), res);
  }

  std::pair<ztensor<4>, dtensor<2>> minimize(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                              const dtensor<2>& initial_guess, const itensor<1>& bath_structure,
                                              double freq_cutoff, bath_fitting_method method) {
    switch (method) {
      case NORM_L1:
      case NORM_LINF:     return minimize_scalar(freqs, hyb_fun, initial_guess, bath_structure, freq_cutoff, method);
      case NORM_L2_TRAPZ: return minimize_jacobian(freqs, hyb_fun, initial_guess, bath_structure, freq_cutoff);
      default:            return minimize_jacobian(freqs, hyb_fun, initial_guess, bath_structure, freq_cutoff);
    }
  }

}  // namespace green::impurity
