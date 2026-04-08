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

#ifndef GREEN_IMPURITY_SOLVER_H
#define GREEN_IMPURITY_SOLVER_H

#include "common_defs.h"

namespace green::impurity {

  using green_dc_func = std::function<void(
        std::string, int imp_n, utils::shared_object<ztensor<5>>&, ztensor<4>&, utils::shared_object<ztensor<5>>&)>;

  class impurity_solver {
    using func = std::function<std::tuple<ztensor<3>, ztensor<4>>(
        size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff, const ztensor<3>& delta_1,
        const ztensor<4>& delta_w, const dtensor<4>& interaction, const ztensor<4>& g_w)>;

  public:
    impurity_solver(const green::params::params& p, const grids::transformer_t& ft, const bz_utils_t& bz_utils,
                    const green_dc_func& dc_func);

    std::tuple<ztensor<3>, ztensor<4>> solve(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                             const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
                                             const ztensor<4>& g) const;

  private:
    std::string                 _input_file;
    std::string                 _root;
    bool                        _spin_symm;
    const grids::transformer_t& _ft;
    const bz_utils_t&           _bz_utils;
    size_t                      _nimp;
    func                        _impurity_call;
    green_dc_func               _dc_solver;
    std::string                 _dc_data_prefix;

    std::tuple<ztensor<3>, ztensor<4>> solve_imp(size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                                                 const dtensor<4>& interaction, const ztensor<3>& sigma_inf,
                                                 const ztensor<4>& sigma_w, const ztensor<4>& g_w) const;

    std::tuple<ztensor<3>, ztensor<4>> extract_delta(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                                     const ztensor<3>& sigma_inf, const ztensor<4>& sigma_w,
                                                     const ztensor<4>& g_w) const;

    /**
     * Project impurity quantities to active space
     * @param mu chemical potential
     * @param ovlp overlap matrix
     * @param h_core kinetic Hamiltonian
     * @param sigma_inf impurity static self-energy
     * @param sigma impurity dynamic self-energy
     * @param g impurity Green's function
     * @param UU transformation to active space
     * @return tuple of projected (ovlp, h_core, sigma_inf, g, sigma)
     */
    std::tuple<ztensor<3>, ztensor<3>, ztensor<3>, ztensor<4>, ztensor<4>> project_to_as(
        double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
        const ztensor<4>& g, const ztensor<2>& UU) const;
  };

}  // namespace green::impurity

#endif  // GREEN_IMPURITY_SOLVER_H
