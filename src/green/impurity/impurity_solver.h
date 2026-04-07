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

#ifndef GREEN_IMPURITY_SOLVER_H
#define GREEN_IMPURITY_SOLVER_H

#include "common_defs.h"
#include "ed_impurity_solver.h"
#include "inchworm_impurity_solver.h"
#include "gw_impurity_solver.h"

namespace green::impurity {

  using green_dc_func = std::function<void(
        std::string, int imp_n, utils::shared_object<ztensor<5>>&, ztensor<4>&, utils::shared_object<ztensor<5>>&)>;

  class impurity_solver {
    using func    = std::function<std::tuple<ztensor<3>, ztensor<4>>(
        size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& delta_1,
        const ztensor<4>& delta_w, const dtensor<4>& interaction, const ztensor<4>& g_w)>;

  public:
    impurity_solver(const green::params::params& p, const grids::transformer_t& ft, const bz_utils_t& bz_utils,
                    const green_dc_func& dc_func) :
        _input_file(p["seet_input"]), _root(p["seet_root_dir"]), _spin_symm(p["spin_symm"]), _ft(ft), _bz_utils(bz_utils),
        _dc_solver(dc_func), _dc_data_prefix(p["dc_data_prefix"])  {
      size_t        ns = 2;
      h5pp::archive ar(_input_file, "r");
      ar["nimp"] >> _nimp;
      ar.close();
      if(p["impurity_solver"].as<std::string>() == "ED") {
        std::shared_ptr<void> ed_solver(new ed_impurity_solver(p["seet_input"], p["bath_file"], p["impurity_solver_exec"],
                                                             p["impurity_solver_params"], p["seet_root_dir"]));
        _impurity_call = [ed_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                           const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction,
                                           const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
          return static_cast<ed_impurity_solver*>(ed_solver.get())
              ->solve(imp_n, _ft, mu, ovlp, h_core, delta_1, delta_w, interaction, g_w);
        };
      } else if (p["impurity_solver"].as<std::string>() == "INCHWORM") {
        if (p["itermax"].as<int>() > 1 && p["mixing_type"].as<std::string>() != "SIGMA_MIXING") {
          throw std::runtime_error("SEET + inchworm currently only supports itermax = 1 and SIGMA_MIXING mode");
        }
        std::shared_ptr<void> inchworm_solver(new inchworm_impurity_solver(p["seet_input"], p["impurity_solver_exec"],
                                                             p["impurity_solver_params"], p["seet_root_dir"]));
        _impurity_call = [inchworm_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                           const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction,
                                           const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
          return static_cast<inchworm_impurity_solver*>(inchworm_solver.get())
              ->solve(imp_n, _ft, mu, ovlp, h_core, delta_1, delta_w, interaction, g_w);
        };
      } else if (p["impurity_solver"].as<std::string>() == "GW") {
        std::shared_ptr<void> gw_solver(new gw_impurity_solver(p["seet_input"], p["bath_file"], p["impurity_solver_exec"],
                                                             p["impurity_solver_params"], _dc_data_prefix, p["seet_root_dir"]));
        _impurity_call = [gw_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                           const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction,
                                           const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
          return static_cast<gw_impurity_solver*>(gw_solver.get())
              ->solve(imp_n, _ft, mu, ovlp, h_core, delta_1, delta_w, interaction, g_w);
        };
      }
    }

    auto solve(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
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

    auto solve_imp(size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const dtensor<4>& interaction,
                   const ztensor<3>& sigma_inf, const ztensor<4>& sigma_w, const ztensor<4>& g_w) const;

    auto extract_delta(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf,
                       const ztensor<4>& sigma_w, const ztensor<4>& g_w) const -> std::tuple<ztensor<3>, ztensor<4>>;

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
    auto project_to_as(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf,
                       const ztensor<4>& sigma, const ztensor<4>& g,
                       const ztensor<2>& UU) const -> std::tuple<ztensor<3>, ztensor<3>, ztensor<3>, ztensor<4>, ztensor<4>>;
  };

  inline auto impurity_solver::solve_imp(size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                         const dtensor<4>& interaction, const ztensor<3>& sigma_inf, const ztensor<4>& sigma_w,
                                         const ztensor<4>& g_w) const {
    if (!std::filesystem::exists(_root)) {
      std::filesystem::create_directory(_root);
    }

    auto [delta_1, delta_w] = extract_delta(mu, ovlp, h_core, sigma_inf, sigma_w, g_w);
    return _impurity_call(imp_n, mu, ovlp, h_core, delta_1, delta_w, interaction, g_w);
  }

  inline auto impurity_solver::extract_delta(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                             const ztensor<3>& sigma_inf, const ztensor<4>& sigma_w,
                                             const ztensor<4>& g_w) const -> std::tuple<ztensor<3>, ztensor<4>> {
    size_t     nw   = g_w.shape()[0];
    size_t     ns   = g_w.shape()[1];
    size_t     naso = g_w.shape()[2];
    ztensor<3> delta_1(h_core.shape());
    ztensor<4> delta(g_w.shape());
    for (size_t iw = 0; iw < nw; ++iw) {
      for (size_t is = 0; is < ns; ++is) {
        auto g_inv_w_imp = matrix(ovlp(is)) * (_ft.wsample_fermi()(iw) * 1.0i + mu) - matrix(h_core(is)) - matrix(sigma_inf(is)) -
                           matrix(sigma_w(iw, is));
        auto g_inv_w_loc      = matrix(g_w(iw, is)).inverse().eval();
        matrix(delta(iw, is)) = g_inv_w_imp - g_inv_w_loc;
      }
    }
    if (_spin_symm) {
      for (size_t iw = 0; iw < nw; ++iw) {
        ztensor<2> tmp(naso, naso);
        for (size_t is = 0; is < ns; ++is) {
          tmp += delta(iw, is);
        }
        tmp /= ns;
        for (size_t is = 0; is < ns; ++is) delta(iw, is) << tmp;
      }
    }

    // extract constant shift in delta
    grids::MatrixXcd     A(3, 3);
    grids::MatrixXcd     B(3, 1);
    std::complex<double> iwn(0.0, -1. / _ft.wsample_fermi()(nw - 1));
    std::complex<double> iwn1(0.0, -1. / _ft.wsample_fermi()(nw - 2));
    std::complex<double> iwn2(0.0, -1. / _ft.wsample_fermi()(nw - 3));
    for (size_t is = 0; is < ns; ++is) {
      for (size_t io = 0; io < naso; ++io) {
        for (size_t jo = 0; jo < naso; ++jo) {
          A << 1.0, iwn, iwn * iwn, 1.0, iwn1, iwn1 * iwn1, 1.0, iwn2, iwn2 * iwn2;
          B << delta(nw - 1, is, io, jo), delta(nw - 2, is, io, jo), delta(nw - 3, is, io, jo);
          grids::MatrixXcd X  = A.colPivHouseholderQr().solve(B).eval();
          delta_1(is, io, jo) = X(0, 0).real();
        }
      }
    }
    for (size_t iw = 0; iw < nw; ++iw) delta(iw) -= delta_1;
    return std::make_tuple(delta_1, delta);
  }

  inline auto impurity_solver::solve(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf,
                                     const ztensor<4>& sigma, const ztensor<4>& g) const {
    size_t     nt = _ft.sd().repn_fermi().nts();
    size_t     ns = ovlp.shape()[0];
    ztensor<3> sigma_inf_loc_new(sigma_inf.shape());
    ztensor<4> sigma_w_loc_new(sigma.shape());
    utils::mpi_context mpi_ctx(MPI_COMM_SELF);
    for (int imp = 0; imp < _nimp; ++imp) {
      // project local quantities onto an active subspace
      dtensor<2> uu;
      dtensor<4> interaction;
      {
        h5pp::archive ar(_input_file, "r");
        ar[std::to_string(imp) + "/UU"] >> uu;
        ar[std::to_string(imp) + "/interaction"] >> interaction;
        ar.close();
      }
      auto [ovlp_as, h_core_as, sigma_inf_as, g_as, sigma_as] =
          project_to_as(mu, ovlp, h_core, sigma_inf, sigma, g, uu.astype<std::complex<double>>());
      size_t naso = h_core_as.shape()[2];
      ztensor<4> g_as_w(_ft.sd().repn_fermi().nw(), g_as.shape()[1], g_as.shape()[2], g_as.shape()[3]);
      ztensor<4> sigma_as_w(_ft.sd().repn_fermi().nw(), sigma_as.shape()[1], sigma_as.shape()[2], sigma_as.shape()[3]);
      _ft.tau_to_omega(g_as, g_as_w);
      _ft.tau_to_omega(sigma_as, sigma_as_w);
      auto [sigma_inf_new, sigma_w_new] = solve_imp(imp, mu, ovlp_as, h_core_as, interaction, sigma_inf_as, sigma_as_w, g_as_w);
      std::array<size_t, 5> shape_in{nt, ns, 1, naso, naso};
      std::array<size_t, 4> shape_in_inf{ns, 1, naso, naso};
      std::array<size_t, 4> shape_out{nt, ns, naso, naso};
      std::array<size_t, 3> shape_out_inf{ns, naso, naso};

      utils::shared_object<ztensor<5>> sigma_dc(shape_in, mpi_ctx);
      utils::shared_object<ztensor<5>> g_dc(shape_in, mpi_ctx);
      sigma_dc.fence();
      sigma_dc.object().set_zero();
      sigma_dc.fence();
      g_dc.fence();
      g_dc.object() << g_as.reshape(shape_in);
      g_dc.fence();

      // DEBUG save
      h5pp::archive debug_data("debug." + std::to_string(imp) + ".output.h5", "w");
      debug_data["dc/G_tau_in"] << g_dc.object();

      ztensor<4> sigma_inf_dc(shape_in_inf);
      _dc_solver(_dc_data_prefix, imp, g_dc, sigma_inf_dc, sigma_dc);

      // Transform impurity solver results from Omega to Tau
      _ft.omega_to_tau(sigma_w_new, sigma_as);
      debug_data["dc/Sigma1_out"] << sigma_inf_dc;
      debug_data["dc/Sigma_tau_out"] << sigma_dc.object();
      debug_data["impurity/Sigma1_raw"] << sigma_inf_new;
      debug_data["impurity/Sigma_tau_raw"] << sigma_as;
      debug_data.close();

      // Update impurity results by subtracting DC sigma
      sigma_inf_new -= sigma_inf_dc.reshape(shape_out_inf);
      sigma_as -= sigma_dc.object().reshape(shape_out);
      for (size_t is = 0; is < ns; ++is) {
        matrix(sigma_inf_loc_new(is)) += matrix(uu).transpose() * matrix(sigma_inf_new(is)) * matrix(uu);
      }
      for (size_t it = 0; it < nt; ++it) {
        for (size_t is = 0; is < ns; ++is) {
          matrix(sigma_w_loc_new(it, is)) += matrix(uu).transpose() * matrix(sigma_as(it, is)) * matrix(uu);
        }
      }
      std::cout << "Impurity " << imp << " finished" << std::endl;
    }
    return std::make_tuple(sigma_inf_loc_new, sigma_w_loc_new);
  }

  inline auto impurity_solver::project_to_as(
      double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
      const ztensor<4>& g, const ztensor<2>& UU) const -> std::tuple<ztensor<3>, ztensor<3>, ztensor<3>, ztensor<4>, ztensor<4>> {
    size_t     nt   = g.shape()[0];
    size_t     ns   = ovlp.shape()[0];
    size_t     naso = UU.shape()[0];
    ztensor<3> ovlp_as(ns, naso, naso);
    ztensor<3> h_core_as(ns, naso, naso);
    ztensor<3> sigma_inf_as(ns, naso, naso);
    ztensor<4> g_as(nt, ns, naso, naso);
    ztensor<4> sigma_as(sigma.shape()[0], ns, naso, naso);
    for (size_t is = 0; is < ns; ++is) {
      matrix(ovlp_as(is))      = matrix(UU) * matrix(ovlp(is)) * matrix(UU).transpose();
      matrix(h_core_as(is))    = matrix(UU) * matrix(h_core(is)) * matrix(UU).transpose();
      matrix(sigma_inf_as(is)) = matrix(UU) * matrix(sigma_inf(is)) * matrix(UU).transpose();
    }
    for (size_t it = 0; it < nt; ++it) {
      for (size_t is = 0; is < ns; ++is) {
        matrix(g_as(it, is))     = matrix(UU) * matrix(g(it, is)) * matrix(UU).transpose();
        matrix(sigma_as(it, is)) = matrix(UU) * matrix(sigma(it, is)) * matrix(UU).transpose();
      }
    }
    return std::make_tuple(ovlp_as, h_core_as, sigma_inf_as, g_as, sigma_as);
  }
}  // namespace green::impurity

#endif  // GREEN_IMPURITY_SOLVER_H
