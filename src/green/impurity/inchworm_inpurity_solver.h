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
      // One-body term
      // Static:
      {
        std::ofstream hooping_file("hopping.txt");
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
        std::ofstream delta_file("delta.txt");
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
      // Two-body term
      {
        // transform interaction into physics convention
        auto   interaction_phys = ndarray::transpose(interaction, "ijkl->ikjl");
        size_t non_zero         = 0;
        for (size_t i = 0; i < nio * ns; ++i) {
          for (size_t j = 0; j < nio * ns; ++j) {
            for (size_t k = 0; k < nio * ns; ++k) {
              for (size_t l = 0; l < nio * ns; ++l) {
                size_t I = i / ns;
                size_t J = j / ns;
                size_t K = k / ns;
                size_t L = l / ns;
                if (std::abs(interaction_phys(I, J, K, L)) > 1e-10) {
                  ++non_zero;
                }
              }
            }
          }
        }
        std::ofstream U_file("Uijkl.txt");
        U_file << non_zero << "\n";
        for (size_t i = 0; i < nio * ns; ++i) {
          for (size_t j = 0; j < nio * ns; ++j) {
            for (size_t k = 0; k < nio * ns; ++k) {
              for (size_t l = 0; l < nio * ns; ++l) {
                size_t I = i / ns;
                size_t J = j / ns;
                size_t K = k / ns;
                size_t L = l / ns;
                if (std::abs(interaction_phys(I, J, K, L)) > 1e-10)
                  U_file << i << " " << j << " " << k << " " << l << " " << interaction_phys(I, J, K, L) << " " << 0.0 << "\n";
              }
            }
          }
        }
        U_file.close();
      }
      std::string run       = (_impurity_solver_exec + " " + _impurity_solver_params);
      int         sysresult = std::system(run.c_str());
      { std::cerr << "Impurity result file has not been found" << std::endl; }
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
