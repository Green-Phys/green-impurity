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

#include <functional>
#include <tuple>

#include <green/params/params.h>
#include <green/utils/mpi_shared.h>

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

    /**
     * Solve all impurity problems and return the total impurity self-energy correction.
     *
     * @param mu         chemical potential
     * @param ovlp       overlap matrix (local, orthogonal basis)
     * @param h_core     one-body Hamiltonian (local)
     * @param sigma_inf_weak  static self-energy from the weak-coupling solver (HF/GW)
     * @param sigma_inf_full  full static self-energy including impurity corrections from previous iterations
     * @param sigma      dynamic self-energy (tau, local)
     * @param g          Green's function (tau, local)
     * @return (sigma_inf_imp, sigma_tau_imp): impurity correction to static and dynamic self-energy,
     *         with double-counting already subtracted, ready to be added to the weak-coupling result
     */
    std::tuple<ztensor<3>, ztensor<4>> solve(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                             const ztensor<3>& sigma_inf_weak, const ztensor<3>& sigma_inf_full,
                                             const ztensor<4>& sigma, const ztensor<4>& g) const;

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

    /**
     * Extract the bath hybridization function Delta(iw) from the impurity and local Green's functions.
     *
     * Implements:  Delta(iw) = G_imp^{-1}(iw) - G_loc^{-1}(iw)
     * where        G_imp^{-1}(iw) = (iw + mu)*S - fock_act_loc - Sigma_w
     * and          fock_act_loc = h_core + sigma_inf_full  (Python's F_act_loc)
     *
     * Returns (delta_1, delta_w): static offset and frequency-dependent bath hybridization.
     */
    std::tuple<ztensor<3>, ztensor<4>> extract_delta(double mu, const ztensor<3>& ovlp,
                                                     const ztensor<3>& fock_act_loc,
                                                     const ztensor<4>& sigma_w, const ztensor<4>& g_w) const;

    /**
     * Project quantities from the orthogonal full space to the active space (AS).
     *
     * @param mu           chemical potential
     * @param ovlp         overlap
     * @param h_core       one-body Hamiltonian
     * @param sigma_inf    weak static self-energy to be projected
     * @param sigma        dynamic self-energy (tau)
     * @param g            Green's function (tau)
     * @param UU           active-space projection matrix (naso x nso)
     * @return tuple (ovlp_as, h_core_as, sigma_inf_as, g_as, sigma_as)
     */
    std::tuple<ztensor<3>, ztensor<3>, ztensor<3>, ztensor<4>, ztensor<4>> project_to_as(
        double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
        const ztensor<4>& g, const ztensor<2>& UU) const;
  };

}  // namespace green::impurity

#endif  // GREEN_IMPURITY_SOLVER_H
