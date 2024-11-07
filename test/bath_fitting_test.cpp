#include "green/impurity/bath_fitting.h"

#include <catch2/catch_test_macros.hpp>

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

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure);
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

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure);
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

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess.reshape(1, initial_guess.shape()[0]), bath_structure);
    REQUIRE(compare_bath(bath, new_bath(0), bath_structure));
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

    auto [new_hyb, new_bath] = green::impurity::minimize(freqs, hyb_fun, initial_guess, bath_structure);
    REQUIRE(compare_bath(bath(0), new_bath(0), bath_structure));
    REQUIRE(compare_bath(bath(1), new_bath(1), bath_structure));
  }
}
