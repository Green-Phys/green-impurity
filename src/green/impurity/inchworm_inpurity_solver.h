/*
 * Copyright (c) 2025 University of Michigan
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

#ifndef GREEN_INCHWORM_INPURITY_SOLVER_H
#define GREEN_INCHWORM_INPURITY_SOLVER_H

namespace green::impurity {

  class inchworm_inpurity_solver {
    template <typename prec>
    using MMatrixX = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    template <typename prec>
    using CMMatrixX = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  public:
    inchworm_inpurity_solver(const std::string& input_file, const std::string& impurity_solver_exec,
                             const std::string& impurity_solver_params, const std::string& root) :
        _input_file(input_file), _impurity_solver_exec(impurity_solver_exec), _impurity_solver_params(impurity_solver_params),
        _root(root) {
      h5pp::archive ar(input_file, "r");
      ar["nimp"] >> _nimp;
      ar["to_even_tau"] >> _uxl;
      ar.close();
    }

    auto solve(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
               const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction, const ztensor<4>& g_w) const {
      ztensor<3> sigma_inf_new(delta_1.shape());
      ztensor<4> sigma_new(delta_w.shape());
      size_t     nio = h_core.shape()[1];
      size_t     ns  = h_core.shape()[0];
      bool       rhf = (ns == 1);
      if (rhf) std::cout << "Detected ns = 1; Assuming this is RHF. Support for relativistic cases not implemented yet." << std::endl;
      // One-body term
      // Static:
      {
        std::ofstream hooping_file("imp_" + std::to_string(imp_n) + "_hopping.txt");
        for (size_t i = 0; i < nio; ++i) {
          for (size_t s1 = 0; s1 < ns; ++s1) {
            for (size_t j = 0; j < nio; ++j) {
              for (size_t s2 = 0; s2 < ns; ++s2) {
                auto h_imp = h_core(s1, i, j) + delta_1(s1, i, j);
                hooping_file << i * ns + s1 << " " << j * ns + s2 << " ";
                if (s1 == s2)
                  hooping_file << h_imp.real() << " " << h_imp.imag() << "\n";
                else
                  hooping_file << 0.0 << " " << 0.0 << "\n";
              }
            }
          }
        }
        hooping_file.close();
      }
      // Dynamic
      {
        auto&                           Tcn = ft.Tcn();
        CMMatrixX<double>               Ttc_even(_uxl.data(), _uxl.shape()[0], _uxl.shape()[1]);
        ztensor<4>                      delta_t(_uxl.shape()[0], ns, nio, nio);
        MMatrixX<std::complex<double>>  delta_t_m(delta_t.data(), _uxl.shape()[0], ns * nio * nio);
        CMMatrixX<std::complex<double>> delta_w_m(delta_w.data(), delta_w.shape()[0], ns * nio * nio);
        delta_t_m = Ttc_even * Tcn * delta_w_m * std::sqrt(2.0 / ft.sd().beta());
        std::ofstream delta_file("imp_" + std::to_string(imp_n) + "_delta.txt");
        for (size_t t = 0; t < delta_t.shape()[0]; ++t) {
          for (size_t i = 0; i < nio; ++i) {
            for (size_t s1 = 0; s1 < ns; ++s1) {
              for (size_t j = 0; j < nio; ++j) {
                for (size_t s2 = 0; s2 < ns; ++s2) {
                  delta_file << t << " " << i * ns + s1 << " " << j * ns + s2 << " ";
                  if (s1 == s2)
                    delta_file << delta_t(t, s1, i, j).real() << " " << delta_t(t, s1, i, j).imag() << "\n";
                  else
                    delta_file << 0.0 << " " << 0.0 << "\n";
                }
              }
            }
          }
        }
      }
      // Two-body term with spin symmetry
      // NOTE: Whether or not we break spin-symmetry in G, or even in x2c relativistic cases, U will still have spin-symmetry.
      {
        size_t ndim_increment   = (rhf) ? 1 : 4; // restricted vs unrestricted
        size_t spin_decrement   = (rhf) ? 0 : 2; // avoid double creation/annihilation in same spin-orbitals
        size_t non_zero         = 0;
        for (size_t I = 0; I < nio; ++I) {
          for (size_t J = 0; J < nio; ++J) {
            for (size_t K = 0; K < nio; ++K) {
              for (size_t L = 0; L < nio; ++L) {
                if (std::abs(interaction(I, J, K, L)) < 1e-10) continue; // forget all logic and skip if interaction is zero
                if (rhf) {
                  non_zero += ndim_increment;
                  continue;
                }
                // uhf case
                // if I == K or J == L, it warrants that the spin label of I and K will be opposite,
                // and by construction spin of I = spin of J and spin of K = spin of L,
                // so when (I==K) or (J==L) or both, we need to remove two combinations:
                // (up up up up) and (dn dn dn dn)
                non_zero += ndim_increment;
                if (I == K || J == L) non_zero -= spin_decrement;
              }
            }
          }
        }
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "NOTE: Will write interaction tensor in the Tensor notation!" << std::endl;
        std::cout << "i.,e., 2-body term = (1/2) * U(ijkl) cdag_i cdag_k c_k c_l" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
        std::ofstream U_file("imp_" + std::to_string(imp_n) + "_Uijkl.txt");
        U_file << non_zero << "\n";
        size_t idx = 0;
        for (size_t i = 0; i < nio * ns; ++i) {
          for (size_t j = 0; j < nio * ns; ++j) {
            for (size_t k = 0; k < nio * ns; ++k) {
              for (size_t l = 0; l < nio * ns; ++l) {
                size_t I = i / ns;
                size_t J = j / ns;
                size_t K = k / ns;
                size_t L = l / ns;
                // Internally, IJKL and ijkl will be treated as chemist notation indices, i.e.
                //    i = (I,s1); j = (J,s1); k = (K,s2); l = (L,s2)
                // When we store to Uijkl.txt, we will re-arrange the indices to match Tensor notation.
                // --------------------------------------------------------------
                // Logic for storing in Tensor notation
                // --------------------------------------------------------------
                // We have the tensor notation:    V(ijkl) cdag_i cdag_j c_k c_l
                // Let's rearrange by changing dummy index labels:
                //     V(iklj) cdag_i cdag_k c_l c_j
                // Compare with:             U(ijkl) cdag_i cdag_k c_l c_j
                // Therefore,   V(iklj) = U(ijkl)
                if (std::abs(interaction(I, J, K, L)) < 1e-10) continue;
                if (rhf) {
                  U_file << idx << "\t" << i << " " << k << " " << l << " " << j << " " << interaction(I, J, K, L) << " " << 0.0 << "\n";
                  ++idx;
                } else {
                  // Deal with spin only for UHF
                  size_t s1 = i % ns;
                  size_t s2 = j % ns;
                  size_t s3 = k % ns;
                  size_t s4 = l % ns;
                  if (i == k || j == l) continue;       // ignore double creation/annihilation in same spin-orbitals (different from I,J,K,L which are atomic orbitals)
                  if (s1 != s2 || s3 != s4) continue;   // ignore spin-flip terms
                  // what remains is a valid term in Uijkl
                  U_file << idx << "\t" << i << " " << k << " " << l << " " << j << " " << interaction(I, J, K, L) << " " << 0.0 << "\n";
                  ++idx;
                }
              }
            }
          }
        }
        U_file.close();
      }
      // Inchworm requires another SLURM job, so we will return ZERO here, and update the actual
      // result after inchworm calculation is done.

      // std::string run       = (_impurity_solver_exec + " " + _impurity_solver_params);
      // int         sysresult = std::system(run.c_str());
      // { std::cerr << "Impurity result file has not been found" << std::endl; }
      sigma_inf_new.set_zero();
      sigma_new.set_zero();
      return std::make_tuple(sigma_inf_new, sigma_new);
    }

  private:
    std::string _input_file;
    std::string _impurity_solver_exec;
    std::string _impurity_solver_params;
    std::string _root;
    size_t      _nimp;
    dtensor<2>  _uxl;
  };
}  // namespace green::impurity
#endif  // GREEN_INCHWORM_INPURITY_SOLVER_H
