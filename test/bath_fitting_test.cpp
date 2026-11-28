#include "green/impurity/bath_fitting.h"

#include <green/h5pp/archive.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <limits>

#include <random>

template <size_t N>
using ztensor = green::ndarray::ndarray<std::complex<double>, N>;
template <size_t N>
using dtensor = green::ndarray::ndarray<double, N>;
template <size_t N>
using itensor = green::ndarray::ndarray<int, N>;

bool compare_bath(const dtensor<1>& bath_1, const dtensor<1>& bath_2, const itensor<1>& bath_structure, double tol = 1e-4) {
  size_t shift = 0;
  for (size_t io = 0; io < bath_structure.shape()[0]; ++io) {
    std::vector<std::pair<double, double>> bath1(bath_structure(io));
    std::vector<std::pair<double, double>> bath2(bath_structure(io));
    for (size_t i = 0; i < bath_structure(io); ++i) {
      bath1[i] = {bath_1(shift + i), bath_1(shift + bath_structure(io) + i)};
      bath2[i] = {bath_2(shift + i), bath_2(shift + bath_structure(io) + i)};
    }
    for (auto& x : bath1) {
      for (size_t i = 0; i < bath2.size(); ++i) {
        if (std::abs(x.first - bath2[i].first) < tol && std::abs(x.second - bath2[i].second) < tol) {
          bath2.erase(bath2.begin() + i);
          break;
        }
      }
    }
    if (!bath2.empty()) {
      return false;
    }
    shift += bath_structure(io) * 2;
  }
  return true;
}

TEST_CASE("Bath Fitting") {
  const auto method = GENERATE(green::impurity::NORM_L2_TRAPZ, green::impurity::ESPRIT);
  CAPTURE(method);
  SECTION("Test Single Orbital") {
    std::cout << "test leastsq" << std::endl;
    double     beta = 10;
    size_t     nw   = 100;
    size_t     ns   = 1;
    size_t     nio  = 1;
    size_t     ik   = 4;
    size_t     nk   = ik / 2;
    ztensor<1> freqs(nw);
    ztensor<4> hyb_fun(nw, ns, nio, nio);
    dtensor<1> bath(ns * nio * ik);
    std::cout << "Bath:\n";
    for (size_t io = 0; io < nio; ++io) {
      for (size_t i = 0; i < nk; ++i) {
        bath(io * ik + i)      = 0.5;
        bath(io * ik + nk + i) = -0.7 + 1.0 * i;
      }
      for (size_t i = 0; i < nk; ++i) {
        std::cout << bath(io * ik + i) << " ";
      }
      std::cout << "\n";
      for (size_t i = 0; i < nk; ++i) {
        std::cout << bath(io * ik + nk + i) << " ";
      }
      std::cout << std::endl;
    }
    for (int iw = 0, in = -(nw / 2); iw < nw; ++iw, ++in) {
      freqs(iw) = std::complex<double>(0, (2 * in + 1) * M_PI / beta);
      for (size_t is = 0; is < ns; ++is) {
        for (size_t io = 0; io < nio; ++io) {
          for (size_t i = 0; i < nk; ++i)
            hyb_fun(iw, is, io, io) += (bath(io * ik + i) * bath(io * ik + i)) / (freqs(iw) - bath(io * ik + nk + i));
        }
      }
    }
    dtensor<2> initial_guess(1, 4);
    itensor<1> bath_structure(1);
    bath_structure(0)        = 2;
    initial_guess(0, 0)      = 0.4;
    initial_guess(0, 1)      = 0.3;
    initial_guess(0, 2)      = -0.8;
    initial_guess(0, 3)      = 0.3;

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure, -1.0, method);
    REQUIRE(compare_bath(bath, new_bath(0), bath_structure));
  }

  SECTION("Test Single Orbital Two Spins") {
    std::cout << "test leastsq" << std::endl;
    double     beta = 10;
    size_t     nw   = 100;
    size_t     ns   = 2;
    size_t     nio  = 1;
    size_t     ik   = 4;
    size_t     nk   = ik / 2;
    ztensor<1> freqs(nw);
    ztensor<4> hyb_fun(nw, ns, nio, nio);
    dtensor<2> bath(ns, nio * ik);
    std::cout << "Bath:\n";
    for (size_t is = 0; is < ns; ++is) {
      std::cout << "spin " << is << std::endl;
      for (size_t io = 0; io < nio; ++io) {
        for (size_t i = 0; i < nk; ++i) {
          bath(is, io * ik + i)      = 0.5;
          bath(is, io * ik + nk + i) = -0.7 + 1.0 * static_cast<double>(i) + 0.1 * (0.5 - static_cast<double>(is));
        }
        for (size_t i = 0; i < nk; ++i) {
          std::cout << bath(is, io * ik + i) << " ";
        }
        std::cout << "\n";
        for (size_t i = 0; i < nk; ++i) {
          std::cout << bath(is, io * ik + nk + i) << " ";
        }
        std::cout << std::endl;
      }
    }
    for (int iw = 0, in = -(nw / 2); iw < nw; ++iw, ++in) {
      freqs(iw) = std::complex<double>(0, (2 * in + 1) * M_PI / beta);
      for (size_t is = 0; is < ns; ++is) {
        for (size_t io = 0; io < nio; ++io) {
          for (size_t i = 0; i < nk; ++i)
            hyb_fun(iw, is, io, io) += (bath(is, io * ik + i) * bath(is, io * ik + i)) /
                                       (freqs(iw) - bath(is, io * ik + nk + i));
        }
      }
    }
    dtensor<2> initial_guess(2, 4);
    itensor<1> bath_structure(1);
    bath_structure(0)        = 2;
    initial_guess(0, 0)      = 0.4;
    initial_guess(0, 1)      = 0.3;
    initial_guess(0, 2)      = -0.8;
    initial_guess(0, 3)      = 0.3;
    initial_guess(1, 0)      = 0.4;
    initial_guess(1, 1)      = 0.3;
    initial_guess(1, 2)      = -0.8;
    initial_guess(1, 3)      = 0.3;

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure, -1.0, method);
    REQUIRE(compare_bath(bath(0), new_bath(0), bath_structure));
    REQUIRE(compare_bath(bath(1), new_bath(1), bath_structure));
  }

  SECTION("Test Two Orbital") {
    std::cout << "test leastsq" << std::endl;
    double beta = 10;
    size_t nw   = 400;
    size_t ns   = 1;
    size_t nio  = 2;
    size_t nb   = 10;
    // size_t     nk   = ik / 2;
    itensor<1> bath_structure(nio);
    bath_structure(0) = 2;
    bath_structure(1) = 3;
    ztensor<1> freqs(nw);
    ztensor<4> hyb_fun(nw, ns, nio, nio);
    dtensor<1> bath(nb);
    std::cout << "Bath:\n";
    size_t shift = 0;
    for (size_t io = 0; io < nio; ++io) {
      size_t ik = bath_structure(io) * 2;
      size_t nk = bath_structure(io);
      for (size_t i = 0; i < nk; ++i) {
        bath(shift + i)      = 0.5;
        bath(shift + nk + i) = -0.7 + 1.0 * i;
      }
      for (size_t i = 0; i < nk; ++i) {
        std::cout << bath(shift + i) << " ";
      }
      std::cout << "\n";
      for (size_t i = 0; i < nk; ++i) {
        std::cout << bath(shift + nk + i) << " ";
      }
      std::cout << std::endl;
      shift += ik;
    }
    for (int iw = 0, in = -(nw / 2); iw < nw; ++iw, ++in) {
      freqs(iw) = std::complex<double>(0, (2 * in + 1) * M_PI / beta);
      for (size_t is = 0; is < ns; ++is) {
        shift = 0;
        for (size_t io = 0; io < nio; ++io) {
          size_t ik = bath_structure(io) * 2;
          size_t nk = bath_structure(io);
          for (size_t i = 0; i < bath_structure(io); ++i)
            hyb_fun(iw, is, io, io) += (bath(shift + i) * bath(shift + i)) / (freqs(iw) - bath(shift + nk + i));
          shift += ik;
        }
      }
    }
    for (size_t iw = 0; iw < nw; ++iw) {
      hyb_fun(iw, 0, 0, 1) = {3.0, 4.0};
      hyb_fun(iw, 0, 1, 0) = {3.0, -4.0};
    }

    dtensor<1> initial_guess(10);
    initial_guess(0)         = 0.4;
    initial_guess(1)         = 0.3;
    initial_guess(2)         = -0.8;
    initial_guess(3)         = 0.3;
    initial_guess(4)         = 0.4;
    initial_guess(5)         = 0.3;
    initial_guess(6)         = 0.3;
    initial_guess(7)         = -0.8;
    initial_guess(8)         = 0.2;
    initial_guess(9)         = 0.8;

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess.reshape(1, initial_guess.shape()[0]), bath_structure, -1.0, method);
    REQUIRE(compare_bath(bath, new_bath(0), bath_structure));
    for (size_t iw = 0; iw < nw; ++iw) {
      REQUIRE(new_hyb(iw, 0, 0, 1) == std::complex<double>{});
      REQUIRE(new_hyb(iw, 0, 1, 0) == std::complex<double>{});
    }
  }

  SECTION("Test Two Orbital Two Spins") {
    std::cout << "test leastsq" << std::endl;
    double beta = 10;
    size_t nw   = 400;
    size_t ns   = 2;
    size_t nio  = 2;
    size_t nb   = 10;
    // size_t     nk   = ik / 2;
    itensor<1> bath_structure(nio);
    bath_structure(0) = 2;
    bath_structure(1) = 3;
    ztensor<1> freqs(nw);
    ztensor<4> hyb_fun(nw, ns, nio, nio);
    dtensor<2> bath(ns, nb);
    std::cout << "Bath:\n";
    for (size_t is = 0; is < ns; ++is) {
      size_t shift = 0;
      for (size_t io = 0; io < nio; ++io) {
        size_t ik = bath_structure(io) * 2;
        size_t nk = bath_structure(io);
        for (size_t i = 0; i < nk; ++i) {
          bath(is, shift + i)      = 0.5;
          bath(is, shift + nk + i) = -0.7 + 1.0 * static_cast<double>(i) + 0.1 * (0.5 - static_cast<double>(is));
        }
        for (size_t i = 0; i < nk; ++i) {
          std::cout << bath(is, shift + i) << " ";
        }
        std::cout << "\n";
        for (size_t i = 0; i < nk; ++i) {
          std::cout << bath(is, shift + nk + i) << " ";
        }
        std::cout << std::endl;
        shift += ik;
      }
    }
    for (int iw = 0, in = -(nw / 2); iw < nw; ++iw, ++in) {
      freqs(iw) = std::complex<double>(0, (2 * in + 1) * M_PI / beta);
      for (size_t is = 0; is < ns; ++is) {
        size_t shift = 0;
        for (size_t io = 0; io < nio; ++io) {
          size_t ik = bath_structure(io) * 2;
          size_t nk = bath_structure(io);
          for (size_t i = 0; i < bath_structure(io); ++i)
            hyb_fun(iw, is, io, io) +=
                (bath(is, shift + i) * bath(is, shift + i)) / (freqs(iw) - bath(is, shift + nk + i));
          shift += ik;
        }
      }
    }
    dtensor<2> initial_guess(2, 10);
    initial_guess(0, 0)         = 0.4;
    initial_guess(0, 1)         = 0.3;
    initial_guess(0, 2)         = -0.8;
    initial_guess(0, 3)         = 0.3;
    initial_guess(0, 4)         = 0.4;
    initial_guess(0, 5)         = 0.3;
    initial_guess(0, 6)         = 0.3;
    initial_guess(0, 7)         = -0.8;
    initial_guess(0, 8)         = 0.2;
    initial_guess(0, 9)         = 0.8;
    initial_guess(1, 0)         = 0.4;
    initial_guess(1, 1)         = 0.3;
    initial_guess(1, 2)         = -0.8;
    initial_guess(1, 3)         = 0.3;
    initial_guess(1, 4)         = 0.4;
    initial_guess(1, 5)         = 0.3;
    initial_guess(1, 6)         = 0.3;
    initial_guess(1, 7)         = -0.8;
    initial_guess(1, 8)         = 0.2;
    initial_guess(1, 9)         = 0.8;

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure, -1.0, method);
    REQUIRE(compare_bath(bath(0), new_bath(0), bath_structure));
    REQUIRE(compare_bath(bath(1), new_bath(1), bath_structure));
  }
}

bool test_noise_stability_for_bath_fitting(green::impurity::bath_fitting_method METHOD) {
  std::string input_file = std::string(TEST_PATH) + "/ed.0.input.h5";

  // Read Delta/data_in and freq from the HDF5 file
  ztensor<4> hyb_fun;
  dtensor<1> freq_real;
  {
    green::h5pp::archive ar(input_file, "r");
    ar["Delta/data_in"] >> hyb_fun;
    ar["freq"] >> freq_real;
    ar.close();
  }

  size_t nw  = hyb_fun.shape()[0];  // 108
  size_t ns  = hyb_fun.shape()[1];  // 2
  size_t nio = hyb_fun.shape()[2];  // 2

  // Build complex Matsubara frequencies from real frequency values
  ztensor<1> freqs(nw);
  for (size_t iw = 0; iw < nw; ++iw) {
    freqs(iw) = std::complex<double>(0.0, freq_real(iw));
  }

  // Set up bath structure: 4 bath sites per orbital
  size_t     nk = 4;
  itensor<1> bath_structure(nio);
  for (size_t io = 0; io < nio; ++io) {
    bath_structure(io) = nk;
  }

  // Initial guess: V's and epsilon's for each spin and orbital
  size_t     nb_per_spin = nio * nk * 2;
  dtensor<2> initial_guess(ns, nb_per_spin);
  initial_guess.set_zero();
  std::mt19937                          gen(42);
  std::uniform_real_distribution<double> dist_v(0.1, 1.0);
  std::uniform_real_distribution<double> dist_e(-2.0, 2.0);
  for (size_t is = 0; is < ns; ++is) {
    size_t shift = 0;
    for (size_t io = 0; io < nio; ++io) {
      for (size_t i = 0; i < nk; ++i) {
        initial_guess(is, shift + i)      = dist_v(gen);
        initial_guess(is, shift + nk + i) = dist_e(gen);
      }
      shift += nk * 2;
    }
  }

  // Run minimization on clean data
  auto [hyb_clean, bath_clean] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure, -1.0, METHOD);

  // Add 1e-10 noise to the hybridization function
  ztensor<4>                            hyb_noisy(hyb_fun.shape());
  std::normal_distribution<double>      noise_dist(0.0, 1e-10);
  std::mt19937                          noise_gen(123);
  for (size_t iw = 0; iw < nw; ++iw) {
    for (size_t is = 0; is < ns; ++is) {
      for (size_t i = 0; i < nio; ++i) {
        for (size_t j = 0; j < nio; ++j) {
          hyb_noisy(iw, is, i, j) = hyb_fun(iw, is, i, j) +
                                     std::complex<double>(noise_dist(noise_gen), noise_dist(noise_gen));
        }
      }
    }
  }

  // Run minimization on noisy data with the same initial guess
  auto [hyb_noisy_fit, bath_noisy] = green::impurity::minimize(freqs, hyb_noisy, initial_guess, bath_structure, -1.0, METHOD);

  // Compare the fitted hybridization functions (not raw bath parameters,
  // since bath sites with near-zero V have unconstrained epsilon).
  double tol     = 1e-8;
  double max_diff = 0.0;
  for (size_t iw = 0; iw < nw; ++iw) {
    for (size_t is = 0; is < ns; ++is) {
      for (size_t io = 0; io < nio; ++io) {
        double diff = std::abs(hyb_clean(iw, is, io, io) - hyb_noisy_fit(iw, is, io, io));
        max_diff    = std::max(max_diff, diff);
      }
    }
  }
  std::cout << "Max hybridization difference (clean vs noisy): " << max_diff << std::endl;
  return max_diff < tol;
}

TEST_CASE("New Bath Fitting Stability: Jacobian + Trapezoidal") {
  REQUIRE(test_noise_stability_for_bath_fitting(green::impurity::NORM_L2_TRAPZ));
}

TEST_CASE("Legacy Bath Fitting Stability: Scalar + norm", "[.][legacy bath fitting][sensitive to noise]") {
  REQUIRE_FALSE(test_noise_stability_for_bath_fitting(green::impurity::NORM_LINF));
}

TEST_CASE("ESPRIT bath fitting honors the frequency window and orbital site budgets") {
  using green::impurity::ESPRIT;
  using green::impurity::minimize;
  constexpr size_t nw = 100;
  ztensor<1> freqs(nw);
  ztensor<4> hybridization(nw, 1, 2, 2);
  hybridization.set_zero();
  itensor<1> sites(2);
  sites(0) = 4;
  sites(1) = 1;
  dtensor<2> unused_guess;
  constexpr double cutoff = 8.0;
  for (size_t i = 0; i < nw; ++i) {
    freqs(i) = {0, (2.0 * (static_cast<double>(i) - nw / 2) + 1) * M_PI / 10};
    const auto z = freqs(i);
    hybridization(i, 0, 0, 0) = 0.09 / (z + 0.4) + 0.16 / (z - 0.7);
    if (std::abs(z.imag()) > cutoff) hybridization(i, 0, 0, 0) += 100.0;
    hybridization(i, 0, 0, 1) = 0.2 / (z + 0.4);
    hybridization(i, 0, 1, 0) = hybridization(i, 0, 0, 1);
  }
  const auto [fit, bath] = minimize(freqs, hybridization, unused_guess, sites, cutoff, ESPRIT);
  REQUIRE(bath.shape()[1] == 10);
  REQUIRE(std::abs(bath(0, 0) - 0.3) < 1e-9);
  REQUIRE(std::abs(bath(0, 1) - 0.4) < 1e-9);
  for (size_t i : {2, 3, 6, 7, 8, 9}) REQUIRE(bath(0, i) == 0);
  for (size_t i = 0; i < nw; ++i) {
    const auto z = freqs(i);
    REQUIRE(std::abs(fit(i, 0, 0, 0) - (0.09 / (z + 0.4) + 0.16 / (z - 0.7))) < 1e-9);
    REQUIRE(fit(i, 0, 0, 1) == std::complex<double>{});
    REQUIRE(fit(i, 0, 1, 0) == std::complex<double>{});
    REQUIRE(fit(i, 0, 1, 1) == std::complex<double>{});
  }
  REQUIRE_THROWS_AS(minimize(freqs, hybridization, unused_guess, sites, 0.01, ESPRIT), std::invalid_argument);
  freqs(0) = freqs(1);
  REQUIRE_THROWS_AS(minimize(freqs, hybridization, unused_guess, sites, cutoff, ESPRIT), std::invalid_argument);
  freqs(0) = {0, -99 * M_PI / 10};
  hybridization(nw / 2, 0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_WITH(minimize(freqs, hybridization, unused_guess, sites, cutoff, ESPRIT),
                      "ESPRIT spin 0, orbital 0: nonfinite hybridization");
  sites(0) = -1;
  REQUIRE_THROWS_AS(minimize(freqs, hybridization, unused_guess, sites, cutoff, ESPRIT), std::invalid_argument);
}

TEST_CASE("ESPRIT reports a channel with no positive-residue representation") {
  using green::impurity::ESPRIT;
  using green::impurity::minimize;
  ztensor<1> freqs(4);
  ztensor<4> hybridization(4, 1, 1, 1);
  itensor<1> sites(1);
  sites(0) = 1;
  dtensor<2> unused_guess;
  for (size_t i = 0; i < freqs.size(); ++i) {
    freqs(i) = {0.0, 2.0 * i - 3.0};
    hybridization(i, 0, 0, 0) = -1.0 / freqs(i);
  }
  REQUIRE_THROWS_WITH(minimize(freqs, hybridization, unused_guess, sites, -1, ESPRIT),
                      "ESPRIT spin 0, orbital 0: no real poles with positive residues");
  sites(0) = 0;
  const auto [fit, bath] = minimize(freqs, hybridization, unused_guess, sites, -1, ESPRIT);
  REQUIRE(bath.size() == 0);
  for (const auto value : fit) REQUIRE(value == std::complex<double>{});
}

TEST_CASE("ESPRIT recovers an identifiable bath with deterministic noise") {
  using green::impurity::ESPRIT;
  using green::impurity::minimize;
  constexpr size_t nw = 100;
  ztensor<1> freqs(nw);
  ztensor<4> hybridization(nw, 1, 1, 1);
  ztensor<4> noisy(nw, 1, 1, 1);
  itensor<1> sites(1);
  sites(0) = 4;
  const double couplings[] = {0.19, 0.25, 0.30, 0.35};
  const double energies[] = {-2.3, -0.8, 0.55, 1.8};
  for (size_t i = 0; i < nw; ++i) {
    freqs(i) = {0, (2.0 * (static_cast<double>(i) - nw / 2) + 1) * M_PI / 10};
    std::complex<double> expected{};
    for (size_t k = 0; k < 4; ++k) expected += couplings[k] * couplings[k] / (freqs(i) - energies[k]);
    hybridization(i, 0, 0, 0) = expected;
    noisy(i, 0, 0, 0) = expected + 1e-10 * std::complex<double>(std::sin(i + 1.0), std::cos(2.0 * i + 1.0));
  }
  dtensor<2> unused_guess;
  const auto clean_fit = minimize(freqs, hybridization, unused_guess, sites, -1, ESPRIT).first;
  const auto noisy_fit = minimize(freqs, noisy, unused_guess, sites, -1, ESPRIT).first;
  for (size_t i = 0; i < nw; ++i) {
    REQUIRE(std::abs(clean_fit(i, 0, 0, 0) - hybridization(i, 0, 0, 0)) < 1e-10);
    REQUIRE(std::abs(noisy_fit(i, 0, 0, 0) - hybridization(i, 0, 0, 0)) < 1e-8);
  }
}

TEST_CASE("ESPRIT fits the sparse HDF5 bath without an extra active pole") {
  using green::impurity::ESPRIT;
  using green::impurity::minimize;
  green::h5pp::archive archive(std::string(TEST_PATH) + "/ed.0.input.h5", "r");
  ztensor<4> hybridization;
  dtensor<1> omega;
  archive["Delta/data_in"] >> hybridization;
  archive["freq"] >> omega;
  archive.close();
  ztensor<1> freqs(omega.size());
  auto noisy = hybridization.copy();
  const size_t ns = hybridization.shape()[1], nio = hybridization.shape()[2];
  for (size_t i = 0; i < omega.size(); ++i) {
    freqs(i) = {0, omega(i)};
    for (size_t spin = 0; spin < ns; ++spin) {
      for (size_t orbital = 0; orbital < nio; ++orbital) {
        noisy(i, spin, orbital, orbital) += 1e-10 * std::complex<double>(std::sin(i + spin), std::cos(i + orbital));
      }
    }
  }
  itensor<1> sites(nio);
  std::fill(sites.begin(), sites.end(), 4);
  dtensor<2> unused_guess;
  const auto [fit, bath] = minimize(freqs, hybridization, unused_guess, sites, -1, ESPRIT);
  const auto noisy_fit = minimize(freqs, noisy, unused_guess, sites, -1, ESPRIT).first;
  for (size_t spin = 0; spin < ns; ++spin) {
    for (size_t orbital = 0; orbital < nio; ++orbital) {
      CAPTURE(spin, orbital);
      for (size_t k = 0; k < 3; ++k) REQUIRE(bath(spin, 8 * orbital + k) > 0);
      REQUIRE(bath(spin, 8 * orbital + 3) == 0);
      double cost = 0;
      for (size_t i = 0; i < omega.size(); ++i) {
        const double width = omega(std::min(i + 1, omega.size() - 1)) - omega(i ? i - 1 : 0);
        cost += width * std::norm(fit(i, spin, orbital, orbital) - hybridization(i, spin, orbital, orbital)) / 4;
        REQUIRE(std::abs(fit(i, spin, orbital, orbital) - noisy_fit(i, spin, orbital, orbital)) < 1e-8);
      }
      REQUIRE(cost < 3.35e-6);
    }
  }
}
