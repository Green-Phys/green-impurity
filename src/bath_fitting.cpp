#include <green/impurity/bath_fitting.h>

#include <iostream>
#include <numeric>

namespace green::impurity {

  ztensor<4> compute_hyb_fun(const ztensor<1>& freqs, const dtensor<2>& bath, const itensor<1>& bath_structure, size_t ns,
                              size_t nio) {
    ztensor<4> hyb(freqs.size(), ns, nio, nio);
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

  std::pair<ztensor<4>, dtensor<2>> minimize(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                              const dtensor<2>& initial_guess, const itensor<1>& bath_structure,
                                              int type) {
    size_t     ns = hyb_fun.shape()[1];
    dtensor<2> res(ns, std::reduce(bath_structure.begin(), bath_structure.end()) * 2);
    for (size_t is = 0; is < hyb_fun.shape()[1]; ++is) {
      std::cout << "spin " << is << std::endl;
      size_t shift = 0;
      for (size_t io = 0; io < hyb_fun.shape()[3]; ++io) {
        std::cout << "orbital " << io << std::endl;
        lsqcpp::GaussNewtonX<double, hybridization_function_error, lsqcpp::DoglegMethod> optimizer;
        optimizer.setMaximumIterations(40000);
        optimizer.setMinimumGradientLength(1e-8);
        optimizer.setMinimumStepLength(1e-9);
        optimizer.setMinimumError(1e-14);
        optimizer.setRefinementParameters({1.0, 3.0, 1e-6, 0.001, 100});
        optimizer.setVerbosity(0);
        size_t                       ik = bath_structure(io) * 2;
        hybridization_function_error function_error(freqs, hyb_fun, bath_structure, io, is, type);
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

}  // namespace green::impurity
