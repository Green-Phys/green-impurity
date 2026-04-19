#include <green/impurity/impurity_solver.h>
#include <green/impurity/ed_impurity_solver.h>
#include <green/impurity/inchworm_impurity_solver.h>
#include <green/impurity/gw_impurity_solver.h>

#include <filesystem>

namespace green::impurity {

  impurity_solver::impurity_solver(const green::params::params& p, const grids::transformer_t& ft,
                                   const bz_utils_t& bz_utils, const green_dc_func& dc_func) :
      _input_file(p["seet_input"]), _root(p["seet_root_dir"]), _spin_symm(p["spin_symm"]), _ft(ft), _bz_utils(bz_utils),
      _dc_solver(dc_func), _dc_data_prefix(p["dc_data_prefix"]) {
    bath_fitting_method bf_method       = p["bath_fitting_method"];
    double              bf_freq_cutoff  = p["bath_fitting_freq_cutoff"];
    h5pp::archive ar(_input_file, "r");
    ar["nimp"] >> _nimp;
    ar.close();
    switch (parse_impurity_solver_type(p["impurity_solver"].as<std::string>())) {
      case impurity_solver_type::ED: {
        std::shared_ptr<void> ed_solver(new ed_impurity_solver(p["seet_input"], p["bath_file"], p["impurity_solver_exec"],
                                                               p["impurity_solver_params"], p["seet_root_dir"],
                                                               bf_method, bf_freq_cutoff));
        _impurity_call = [ed_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                                           const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction,
                                           const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
          return static_cast<ed_impurity_solver*>(ed_solver.get())
              ->solve(imp_n, _ft, mu, ovlp, hcore_eff, delta_1, delta_w, interaction, g_w);
        };
        break;
      }
      case impurity_solver_type::INCHWORM: {
        if (p["itermax"].as<int>() > 1 && p["mixing_type"].as<std::string>() != "SIGMA_MIXING") {
          throw std::runtime_error("SEET + inchworm currently only supports itermax = 1 and SIGMA_MIXING mode");
        }
        std::shared_ptr<void> inchworm_solver(new inchworm_impurity_solver(p["seet_input"], p["impurity_solver_exec"],
                                                                           p["impurity_solver_params"], p["seet_root_dir"]));
        _impurity_call = [inchworm_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                                                 const ztensor<3>& delta_1, const ztensor<4>& delta_w,
                                                 const dtensor<4>& interaction,
                                                 const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
          return static_cast<inchworm_impurity_solver*>(inchworm_solver.get())
              ->solve(imp_n, _ft, mu, ovlp, hcore_eff, delta_1, delta_w, interaction, g_w);
        };
        break;
      }
      case impurity_solver_type::GW: {
        std::shared_ptr<void> gw_solver(new gw_impurity_solver(p["seet_input"], p["bath_file"], p["impurity_solver_exec"],
                                                               p["impurity_solver_params"], _dc_data_prefix,
                                                               p["seet_root_dir"], bf_method, bf_freq_cutoff));
        _impurity_call = [gw_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                                           const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction,
                                           const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
          return static_cast<gw_impurity_solver*>(gw_solver.get())
              ->solve(imp_n, _ft, mu, ovlp, hcore_eff, delta_1, delta_w, interaction, g_w);
        };
        break;
      }
      // No default: parse_impurity_solver_type() throws incorr_impurity_solver_type
      // for any unrecognised value before this switch is reached.
    }
  }

  std::tuple<ztensor<3>, ztensor<4>> impurity_solver::extract_delta(double mu, const ztensor<3>& ovlp,
                                                                     const ztensor<3>& fock_act_loc,
                                                                     const ztensor<4>& sigma_w,
                                                                     const ztensor<4>& g_w) const {
    size_t     nw   = g_w.shape()[0];
    size_t     ns   = g_w.shape()[1];
    size_t     naso = g_w.shape()[2];
    ztensor<3> delta_1(fock_act_loc.shape());
    ztensor<4> delta(g_w.shape());
    for (size_t iw = 0; iw < nw; ++iw) {
      for (size_t is = 0; is < ns; ++is) {
        // G_imp^{-1}(iw) = (iw + mu)*S - F_act_loc - Sigma_w
        // F_act_loc = h_core + sigma_inf_full  (matches Python's F_act_loc = h_core + Sigma1_full)
        auto g_inv_imp        = matrix(ovlp(is)) * (_ft.wsample_fermi()(iw) * 1.0i + mu) -
                                matrix(fock_act_loc(is)) - matrix(sigma_w(iw, is));
        auto g_inv_loc        = matrix(g_w(iw, is)).inverse().eval();
        matrix(delta(iw, is)) = g_inv_imp - g_inv_loc;
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

    // Extract the static (iw -> inf) offset of delta by fitting the three largest Matsubara points
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

  std::tuple<ztensor<3>, ztensor<4>> impurity_solver::solve(double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                                             const ztensor<3>& sigma_inf_weak,
                                                             const ztensor<3>& sigma_inf_full,
                                                             const ztensor<4>& sigma, const ztensor<4>& g) const {
    if (!std::filesystem::exists(_root)) {
      std::filesystem::create_directories(_root);
    }
    size_t     nt  = _ft.sd().repn_fermi().nts();
    size_t     nw  = _ft.sd().repn_fermi().nw();
    size_t     ns  = ovlp.shape()[0];
    ztensor<3> sigma_inf_imp_loc(sigma_inf_weak.shape());
    ztensor<4> sigma_tau_imp_loc(sigma.shape());
    utils::mpi_context mpi_ctx(MPI_COMM_SELF);

    for (int imp = 0; imp < _nimp; ++imp) {
      dtensor<2> uu;
      dtensor<4> interaction;
      {
        h5pp::archive ar(_input_file, "r");
        ar[std::to_string(imp) + "/UU"] >> uu;
        ar[std::to_string(imp) + "/interaction"] >> interaction;
        ar.close();
      }
      auto uu_c = uu.astype<std::complex<double>>();

      // --- Project all quantities to the active space (AS) ---
      auto [ovlp_as, h_core_as, sigma_inf_weak_as, g_as, sigma_as] =
          project_to_as(mu, ovlp, h_core, sigma_inf_weak, sigma, g, uu_c);
      size_t naso = h_core_as.shape()[2];

      ztensor<3> sigma_inf_full_as(ns, naso, naso);
      for (size_t is = 0; is < ns; ++is)
        matrix(sigma_inf_full_as(is)) = matrix(uu_c) * matrix(sigma_inf_full(is)) * matrix(uu_c).transpose();

      // --- Fourier transform G and Sigma from tau to Matsubara ---
      ztensor<4> g_as_w(nw, ns, naso, naso);
      ztensor<4> sigma_as_w(nw, ns, naso, naso);
      _ft.tau_to_omega(g_as, g_as_w);
      _ft.tau_to_omega(sigma_as, sigma_as_w);

      // --- Compute double-counting (DC) self-energy ---
      std::array<size_t, 5> shape5{nt, ns, 1, naso, naso};
      std::array<size_t, 4> shape4_inf{ns, 1, naso, naso};
      utils::shared_object<ztensor<5>> sigma_dc(shape5, mpi_ctx);
      utils::shared_object<ztensor<5>> g_dc(shape5, mpi_ctx);
      sigma_dc.fence(); sigma_dc.object().set_zero(); sigma_dc.fence();
      g_dc.fence(); g_dc.object() << g_as.reshape(shape5); g_dc.fence();

      ztensor<4> sigma_inf_dc(shape4_inf);
      _dc_solver(_dc_data_prefix, imp, g_dc, sigma_inf_dc, sigma_dc);

      std::array<size_t, 3> shape3{ns, naso, naso};
      std::array<size_t, 4> shape4{nt, ns, naso, naso};
      ztensor<3> sigma_inf_dc_as(shape3);
      sigma_inf_dc_as << sigma_inf_dc.reshape(shape3);

      // --- Impurity Hamiltonian: H0_imp = h_core + Re(sigma_inf_weak - sigma_inf_dc) ---
      // The real part is taken because H0 must be Hermitian for ED; sigma_inf_weak and
      // sigma_inf_dc are Hermitian so their difference is real in a proper basis.
      ztensor<3> hcore_eff_as(h_core_as.shape());
      for (size_t is = 0; is < ns; ++is)
        matrix(hcore_eff_as(is)) = matrix(h_core_as(is)) +
            (matrix(sigma_inf_weak_as(is)) - matrix(sigma_inf_dc_as(is))).real();

      // --- Bath hybridization ---
      // F_act_loc = h_core + sigma_inf_full  (matches Python's F_act_loc)
      // G_imp^{-1}(iw) = (iw + mu)*S - F_act_loc - Sigma_w  =>  Delta = G_imp^{-1} - G_loc^{-1}
      ztensor<3> fock_act_loc_as(h_core_as.shape());
      for (size_t is = 0; is < ns; ++is)
        matrix(fock_act_loc_as(is)) = matrix(h_core_as(is)) + matrix(sigma_inf_full_as(is));

      auto [delta_1, delta_w] = extract_delta(mu, ovlp_as, fock_act_loc_as, sigma_as_w, g_as_w);

      // --- Solve the impurity problem ---
      auto [sigma_inf_imp_as, sigma_w_imp_as] =
          _impurity_call(imp, mu, ovlp_as, hcore_eff_as, delta_1, delta_w, interaction, g_as_w);

      // --- Subtract DC and back-project to the full space ---
      _ft.omega_to_tau(sigma_w_imp_as, sigma_as);
      sigma_inf_imp_as -= sigma_inf_dc_as;
      sigma_as -= sigma_dc.object().reshape(shape4);

      for (size_t is = 0; is < ns; ++is)
        matrix(sigma_inf_imp_loc(is)) += matrix(uu).transpose() * matrix(sigma_inf_imp_as(is)) * matrix(uu);
      for (size_t it = 0; it < nt; ++it)
        for (size_t is = 0; is < ns; ++is)
          matrix(sigma_tau_imp_loc(it, is)) += matrix(uu).transpose() * matrix(sigma_as(it, is)) * matrix(uu);

      std::cout << "Impurity " << imp << " finished" << std::endl;
    }
    return std::make_tuple(sigma_inf_imp_loc, sigma_tau_imp_loc);
  }

  std::tuple<ztensor<3>, ztensor<3>, ztensor<3>, ztensor<4>, ztensor<4>> impurity_solver::project_to_as(
      double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core, const ztensor<3>& sigma_inf, const ztensor<4>& sigma,
      const ztensor<4>& g, const ztensor<2>& UU) const {
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
