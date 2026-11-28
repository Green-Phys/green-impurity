#include <green/impurity/bath_fitting.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace green::impurity {
  namespace {
    using Eigen::Index;
    using Eigen::MatrixXd;
    using Eigen::MatrixXcd;
    using Eigen::VectorXd;
    using Eigen::VectorXcd;

    struct Spectrum {
      VectorXd energy;
      VectorXd residue;
    };

    VectorXd stacked(const VectorXcd& values) {
      VectorXd result(2 * values.size());
      result << values.real(), values.imag();
      return result;
    }

    MatrixXd bath_kernel(const VectorXcd& z, const VectorXd& weight, const VectorXd& energy) {
      MatrixXd kernel(2 * z.size(), energy.size());
      for (Index i = 0; i < energy.size(); ++i) {
        kernel.col(i) = stacked(weight.array() / (z.array() - energy(i)));
      }
      return kernel;
    }

    // Lawson-Hanson active-set NNLS. QR on the passive columns avoids normal equations.
    VectorXd nonnegative_fit(const MatrixXd& kernel, const VectorXd& target) {
      VectorXd norms = kernel.colwise().norm();
      MatrixXd normalized = kernel * norms.cwiseInverse().asDiagonal();
      VectorXd x = VectorXd::Zero(kernel.cols());
      std::vector<Index> passive;
      std::vector<bool> available(kernel.cols(), true);
      const double tolerance = 1e-13 * target.norm();
      const Index limit = 10 * kernel.cols();
      Index iterations = 0;
      while (iterations < limit) {
        const VectorXd gradient = normalized.transpose() * (target - normalized * x);
        Index entering = -1;
        double largest = tolerance;
        for (Index i = 0; i < gradient.size(); ++i) {
          if (available[i] && gradient(i) > largest) {
            entering = i;
            largest = gradient(i);
          }
        }
        if (entering < 0) return x.cwiseQuotient(norms);
        passive.push_back(entering);
        available[entering] = false;
        while (iterations++ < limit) {
          MatrixXd submatrix(kernel.rows(), passive.size());
          for (size_t j = 0; j < passive.size(); ++j) submatrix.col(j) = normalized.col(passive[j]);
          const VectorXd solution = submatrix.colPivHouseholderQr().solve(target);
          if ((solution.array() > 0).all()) {
            for (size_t j = 0; j < passive.size(); ++j) x(passive[j]) = solution(j);
            break;
          }
          double fraction = 1.0;
          for (size_t j = 0; j < passive.size(); ++j) {
            const double current = x(passive[j]);
            if (solution(j) <= 0) {
              fraction = std::min(fraction, current > 0 ? current / (current - solution(j)) : 0.0);
            }
          }
          for (size_t j = 0; j < passive.size(); ++j) {
            const Index i = passive[j];
            x(i) += fraction * (solution(j) - x(i));
            if (x(i) <= tolerance) {
              x(i) = 0;
              available[i] = true;
            }
          }
          passive.erase(std::remove_if(passive.begin(), passive.end(), [&](Index i) { return available[i]; }), passive.end());
        }
      }
      throw std::runtime_error("nonnegative residue solve did not converge");
    }

    // Positive Hermitian Loewner realization of a finite Stieltjes bath.
    // It gives an accurate initial spectrum for exactly representable data.
    VectorXd loewner_poles(const VectorXcd& z, const VectorXcd& y, Index budget) {
      std::vector<Index> upper;
      for (Index i = 0; i < z.size(); ++i) {
        if (z(i).imag() > 0) upper.push_back(i);
      }
      const Index size = upper.size();
      if (size < 2) return {};
      MatrixXcd gram(size, size), shifted(size, size);
      for (Index i = 0; i < size; ++i) {
        const auto zi = z(upper[i]);
        const auto yi = y(upper[i]);
        for (Index j = 0; j < size; ++j) {
          const auto zj = z(upper[j]);
          const auto yj = y(upper[j]);
          gram(i, j) = (std::conj(yj) - yi) / (zi - std::conj(zj));
          shifted(i, j) = (std::conj(zj * yj) - zi * yi) / (zi - std::conj(zj));
        }
      }
      Eigen::SelfAdjointEigenSolver<MatrixXcd> spectrum(gram);
      if (spectrum.info() != Eigen::Success) return {};
      const auto eig = spectrum.eigenvalues();
      const double threshold = std::max(1e-11 * eig(size - 1), 0.0);
      Index rank = 0;
      while (rank < std::min(budget, size) && eig(size - 1 - rank) > threshold) ++rank;
      if (!rank) return {};
      MatrixXcd basis = spectrum.eigenvectors().rightCols(rank) * eig.tail(rank).cwiseSqrt().cwiseInverse().asDiagonal();
      const MatrixXcd pencil = basis.adjoint() * shifted * basis;
      Eigen::SelfAdjointEigenSolver<MatrixXcd> poles(pencil, Eigen::EigenvaluesOnly);
      if (poles.info() != Eigen::Success) return {};
      return poles.eigenvalues();
    }

    // MiniPole's infinite real-frequency map and shifted Hankel ESPRIT
    // (Zhang, Yu and Gull, Phys. Rev. B 110, 235131).
    void compress(const Spectrum& source, double scale, double shift, Index budget, const VectorXcd& z,
                  const VectorXd& weight, const VectorXd& target, Spectrum& best, double& best_error) {
      const Index max_rank = std::min(budget, source.energy.size());
      const Index moments_count = std::max<Index>(64, 3 * max_rank + 1);
      const Index order = 2 * (moments_count - 1) / 5;  // MiniPole's Lfactor = 0.4
      const std::complex<double> a(0, scale);
      VectorXcd moments = VectorXcd::Zero(moments_count);
      for (Index i = 0; i < source.energy.size(); ++i) {
        const std::complex<double> pole(source.energy(i), -shift);
        const auto root = (pole + a) / (pole - a);
        auto amplitude = source.residue(i) * (root - 1.0) * (root - 1.0) / (-2.0 * a);
        for (Index k = 0; k < moments_count; ++k) {
          moments(k) += amplitude;
          amplitude *= root;
        }
      }
      MatrixXcd hankel(moments_count - order, order + 1);
      for (Index i = 0; i < hankel.rows(); ++i) hankel.row(i) = moments.segment(i, order + 1).transpose();
      Eigen::JacobiSVD<MatrixXcd> svd(hankel, Eigen::ComputeThinV);
      if (svd.info() != Eigen::Success) return;
      for (Index rank = 1; rank <= max_rank && svd.singularValues()(rank - 1) > 1e-13 * svd.singularValues()(0); ++rank) {
        const MatrixXcd space = svd.matrixV().leftCols(rank).conjugate();
        const MatrixXcd rotation = space.topRows(order).colPivHouseholderQr().solve(space.bottomRows(order));
        Eigen::ComplexEigenSolver<MatrixXcd> eigen(rotation, false);
        if (eigen.info() != Eigen::Success) continue;
        std::vector<double> candidates;
        for (const auto root : eigen.eigenvalues()) {
          if (std::abs(root) < 1) {
            const double energy = (a * (root + 1.0) / (root - 1.0)).real();
            if (std::isfinite(energy)) candidates.push_back(energy);
          }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        if (candidates.empty()) continue;
        const VectorXd energy = Eigen::Map<const VectorXd>(candidates.data(), candidates.size());
        const MatrixXd kernel = bath_kernel(z, weight, energy);
        const VectorXd residue = nonnegative_fit(kernel, target);
        const double error = (kernel * residue - target).norm();
        if (error < best_error) {
          best = {energy, residue};
          best_error = error;
        }
      }
    }

    Spectrum fit_channel(const VectorXcd& z, const VectorXcd& y, const VectorXd& weight, Index budget) {
      const VectorXd target = stacked(weight.array() * y.array());
      if (target.isZero(0.0)) return {};
      Spectrum best;
      double error = target.norm();
      const VectorXd initial = loewner_poles(z, y, budget);
      if (initial.size()) {
        const double scale = std::max(z.imag().cwiseAbs().minCoeff(), initial.cwiseAbs().mean());
        compress({initial, nonnegative_fit(bath_kernel(z, weight, initial), target)}, scale, 0.1 * scale,
                 budget, z, weight, target, best, error);
      }
      if (error > 1e-8 * target.norm()) {
        // Resolve low-energy poles on a logarithmic grid and wider features on
        // a linear grid before compressing the positive spectrum with ESPRIT.
        const double width = std::max(35.0, initial.size() ? 2 * initial.cwiseAbs().maxCoeff() : 0.0);
        const double low = std::min(1e-5, 0.01 * z.imag().cwiseAbs().minCoeff());
        VectorXd grid(2301);
        grid.head(701) = VectorXd::LinSpaced(701, -width, width);
        grid.segment(701, 800) = VectorXd::LinSpaced(800, std::log(low), std::log(width)).array().exp();
        grid.tail(800) = -grid.segment(701, 800);
        const VectorXd residues = nonnegative_fit(bath_kernel(z, weight, grid), target);
        Spectrum source{VectorXd(residues.size()), VectorXd(residues.size())};
        Index active = 0;
        for (Index i = 0; i < residues.size(); ++i) {
          if (residues(i) <= 0) continue;
          source.energy(active) = grid(i);
          source.residue(active++) = residues(i);
        }
        source.energy.conservativeResize(active);
        source.residue.conservativeResize(active);
        if (active) {
          for (double scale : {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0}) {
            for (double ratio : {0.05, 0.2, 0.5, 1.0}) {
              compress(source, scale, ratio * scale, budget, z, weight, target, best, error);
            }
          }
        }
      }
      if (!best.energy.size()) throw std::runtime_error("no real poles with positive residues");
      return best;
    }
  }

  std::pair<ztensor<4>, dtensor<2>> fit_bath_esprit(const ztensor<1>& freqs, const ztensor<4>& hyb_fun,
                                                  const itensor<1>& bath_structure, double freq_cutoff) {
    const auto [nw, ns, rows, nio] = hyb_fun.shape();
    if (freqs.size() != nw || rows != nio || bath_structure.size() != nio || !std::isfinite(freq_cutoff) ||
        std::any_of(bath_structure.begin(), bath_structure.end(), [](int sites) { return sites < 0; })) {
      throw std::invalid_argument("ESPRIT: invalid frequency cutoff or bath dimensions");
    }

    std::vector<size_t> selected;
    for (size_t i = 0; i < nw; ++i) {
      const auto freq = freqs(i);
      if (freq.real() != 0 || !std::isfinite(freq.imag()) || freq.imag() == 0 ||
          (i && freq.imag() <= freqs(i - 1).imag())) {
        throw std::invalid_argument("ESPRIT: expected finite, strictly increasing fermionic Matsubara frequencies");
      }
      if (freq_cutoff <= 0 || std::abs(freq.imag()) <= freq_cutoff) selected.push_back(i);
    }
    if (selected.size() < 2) throw std::invalid_argument("ESPRIT: frequency window requires at least two samples");

    VectorXcd z(selected.size());
    for (Index i = 0; i < z.size(); ++i) z(i) = freqs(selected[i]);
    VectorXd weight(z.size());
    for (Index i = 0; i < z.size(); ++i) {
      weight(i) = std::sqrt((z(std::min(i + 1, z.size() - 1)).imag() - z(std::max<Index>(i - 1, 0)).imag()) / 2);
    }
    dtensor<2> bath(ns, 2 * std::accumulate(bath_structure.begin(), bath_structure.end(), size_t{0}));
    bath.set_zero();
    for (size_t spin = 0; spin < ns; ++spin) {
      size_t offset = 0;
      for (size_t orbital = 0; orbital < nio; ++orbital) {
        const Index budget = bath_structure(orbital);
        VectorXcd y(z.size());
        for (Index i = 0; i < z.size(); ++i) y(i) = hyb_fun(selected[i], spin, orbital, orbital);
        const std::string channel = "ESPRIT spin " + std::to_string(spin) + ", orbital " + std::to_string(orbital);
        Spectrum fit;
        try {
          if (!y.allFinite()) throw std::invalid_argument("nonfinite hybridization");
          if (budget) fit = fit_channel(z, y, weight, budget);
        } catch (const std::exception& error) {
          throw std::runtime_error(channel + ": " + error.what());
        }
        VectorXcd residual = y;
        Index active = 0;
        for (Index i = 0; i < fit.energy.size(); ++i) {
          if (fit.residue(i) <= 0) continue;
          bath(spin, offset + active) = std::sqrt(fit.residue(i));
          bath(spin, offset + budget + active++) = fit.energy(i);
          residual.array() -= fit.residue(i) / (z.array() - fit.energy(i));
        }
        std::cout << channel << ": " << active << "/" << budget
                  << " active poles, weighted RMS " << (weight.array() * residual.cwiseAbs().array()).matrix().norm() / weight.norm()
                  << ", max error " << residual.cwiseAbs().maxCoeff() << '\n';
        offset += 2 * budget;
      }
    }
    return {compute_hyb_fun(freqs, bath, bath_structure, ns, nio), bath};
  }
}
