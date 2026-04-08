#include <fstream>
#include <cstdlib>
#include <string>
#include "green/impurity/gw_impurity_solver.h"

namespace green::impurity {

  void gw_impurity_solver::fit_and_parse_bath(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<4>& delta_w, const ztensor<3>& ovlp,
                dtensor<2>& Epsk, std::vector<dtensor<2>>& Vk) const {
    // bath fitting
    // minimize() fits delta_w(iw) ~ Σ_k V_k^2 / (iw - eps_k) using bare Matsubara frequencies iw.
    // This yields fitted bath energies eps_k = eps_k_phys - mu (shifted by -mu relative to physical values).
    // In prepare_gw_input(), mu is explicitly subtracted from the impurity diagonal of h0_eff, so the
    // GW solver (which uses frequency iw, not iw+mu) sees the bath propagator (iw - eps_k_fit)^{-1}
    // = (iw + mu - eps_k_phys)^{-1}, which is correct.
    auto [delta_out, bath_arr] = minimize(
      ft.sd().repn_fermi().wsample() * 1.0i, delta_w, _initial_bath[imp_n], _bath_structure[imp_n], 1
    );
    // Read results
    {
      std::ofstream ofile(_root + "/bath.dat", std::ios_base::out);
      for (auto b : bath_arr) {
        ofile << b << " ";
      }
      ofile << std::endl;
    }
    // Parse output of bath fitting
    size_t nio = ovlp.shape()[2];
    size_t ns  = ovlp.shape()[0];
    size_t nb  = std::reduce(_bath_structure[imp_n].begin(), _bath_structure[imp_n].end());
    Epsk = dtensor<2>(nb, ns);
    Vk.clear();
    for (size_t is = 0; is < ns; ++is) {
      for (size_t io = 0, ik = 0, shift = 0; io < nio; ++io) {
        size_t nk = _bath_structure[imp_n](io);
        for (size_t iik = 0; iik < nk; ++iik, ++ik) {
          Epsk(ik, is) = bath_arr(is, shift + nk + iik);
        }
        shift += 2 * nk;
      }
    }
    for (size_t io = 0; io < nio; ++io) {
      dtensor<2> Vk_(Epsk.shape());
      for (size_t io2 = 0, ik = 0, shift = 0; io2 < nio; io2++) {
        size_t nk = _bath_structure[imp_n](io2);
        for (size_t iik = 0; iik < nk; ++iik, ++ik) {
          for (size_t is = 0; is < ns; ++is) {
            if (io == io2) Vk_(ik, is) = bath_arr(is, shift + iik);
          }
        }
        shift += 2 * nk;
      }
      Vk.push_back(Vk_);
    }
  }

  void gw_impurity_solver::write_bath_debug(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                const ztensor<3>& delta_1, const ztensor<4>& delta_true, const ztensor<4>& g_w,
                const dtensor<2>& Epsk, const std::vector<dtensor<2>>& Vk) const {
    h5pp::archive debug_data(_root + "/gw_debug." + std::to_string(imp_n) + ".output.h5", "a");
    debug_data["bath/mu"] << mu;
    debug_data["bath/freq"] << ft.wsample_fermi();
    debug_data["bath/true/ovlp"] << ovlp;
    debug_data["bath/true/h_core"] << hcore_eff;
    debug_data["bath/true/g_w"] << g_w;
    debug_data["bath/true/delta_1"] << delta_1;
    debug_data["bath/true/delta_w"] << delta_true;
    debug_data["bath/fit/Epsk"] << Epsk;
    for (size_t io = 0; io < Vk.size(); ++io) {
      debug_data["bath/fit/Vk_" + std::to_string(io)] << Vk[io];
    }
    debug_data.close();
  }
  
  void gw_impurity_solver::prepare_gw_input(size_t imp_n, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff, const ztensor<3>& delta_1,
                const ztensor<4>& delta_w, const dtensor<2>& Epsk, const std::vector<dtensor<2>>& Vk, double mu,
                const dtensor<4>& interaction, const ztensor<4>& g_w, const grids::transformer_t& ft) const {
    // Step 2: Prepare hybrid system for external GW calculation and write input file
    size_t nio = ovlp.shape()[2];
    size_t ns  = ovlp.shape()[0];
    size_t nb  = Epsk.shape()[0];
    size_t nao_eff = nb + nio;
    ztensor<4> h0_eff(ns, 1, nao_eff, nao_eff);
    ztensor<4> Sk_eff(ns, 1, nao_eff, nao_eff);
    Sk_eff.set_zero();
    for (size_t is = 0; is < ns; ++is) {
      for (size_t i_io = 0; i_io < nio; ++i_io) {
        // Add hcore as top rows and columns
        for (size_t j_io = 0; j_io < nio; ++j_io) {
          h0_eff(is, 0, i_io, j_io) = hcore_eff(is, i_io, j_io) + delta_1(is, i_io, j_io);
        }
        h0_eff(is, 0, i_io, i_io) -= mu;
        // Vk
        for (size_t j_bo = 0; j_bo < nb; ++j_bo) {
          h0_eff(is, 0, i_io, nio + j_bo) = Vk[i_io](j_bo, is);
          h0_eff(is, 0, nio + j_bo, i_io) = Vk[i_io](j_bo, is);
        }
      }
      // eps-k
      for (size_t j_bo = 0; j_bo < nb; ++j_bo) {
        h0_eff(is, 0, nio + j_bo, nio + j_bo) = Epsk(j_bo, is);
      }
      // Sk-eff -- assuming embedding is in orthogonal basis
      for (size_t i = 0; i < nao_eff; ++i) {
        Sk_eff(is, 0, i, i) = std::complex<double>(1.0, 0.0);
      }
    }

    // Save to input file
    std::string gw_input_file = _root + "/gw." + std::to_string(imp_n) + ".input.h5";
    std::filesystem::copy_file(_dc_data_prefix + "." + std::to_string(imp_n) + "/dummy.h5", gw_input_file,
                               std::filesystem::copy_options::overwrite_existing);
    h5pp::archive data(gw_input_file, "a");
    std::array<size_t, 5> new_shape = {ns, 1, nao_eff, nao_eff, 2};
    data.set_attribute<std::string>("__green_version__", "0.3.2");  // Set version attribute -- hardwired for now
    data["HF/H-k"] << h0_eff.view<double>().reshape(new_shape);
    data["HF/H-k"].set_attribute<int>("__complex__", 1);
    data["HF/Fock-k"] << h0_eff.view<double>().reshape(new_shape);
    data["HF/Fock-k"].set_attribute<int>("__complex__", 1);
    data["HF/S-k"] << Sk_eff.view<double>().reshape(new_shape);
    data["HF/S-k"].set_attribute<int>("__complex__", 1);
    data["HF/Energy_nuc"] << 0.0;
    data["params/nel_cell"] << nio;
    data["params/ink"] << 1;
    // determine x2c level
    int nao_old, nso_old;
    data["params/nao"] >> nao_old;
    data["params/nso"] >> nso_old;
    bool x2c = (nao_old * 2 == nso_old);
    // update nao and nso
    data["params/nao"] << nao_eff;
    data["params/nso"] << (x2c ? 2 * nao_eff : nao_eff);
    data.close();

    // Sim file for starting Green's function
    std::string gw_sim_file = _root + "/gw." + std::to_string(imp_n) + ".sim.h5";
    h5pp::archive sim_data(gw_sim_file, "w");
    sim_data.set_attribute<std::string>("__grids_version__", ft.get_version());
    sim_data["iter"] << 1;
    sim_data["iter1/Energy_1b"] << 0.;
    sim_data["iter1/Energy_2b"] << 0.;
    sim_data["iter1/Energy_HF"] << 0.;
    sim_data["iter1/mu"] << 0.;
    sim_data.close();

    // Initialize self-energy to zero; the GW impurity solver will build it up self-consistently.
    size_t nt = ft.sd().repn_fermi().nts();
    ztensor<5> sigma_tau_embed(nt, ns, 1, nao_eff, nao_eff);
    ztensor<4> sigma_inf_embed(ns, 1, nao_eff, nao_eff);
    sigma_tau_embed.set_zero();
    sigma_inf_embed.set_zero();

    h5pp::archive sim_data_append(gw_sim_file, "a");
    sim_data_append["iter1/Sigma1"] << sigma_inf_embed;
    sim_data_append["iter1/Selfenergy/data"] << sigma_tau_embed;
    sim_data_append["iter1/Selfenergy/mesh"] << ft.tau_mesh().points();
    ztensor<5> g_tau_embed(nt, ns, 1, nao_eff, nao_eff);
    g_tau_embed.set_zero();
    sim_data_append["iter1/G_tau/data"] << g_tau_embed;
    sim_data_append["iter1/G_tau/mesh"] << ft.tau_mesh().points();
    sim_data_append.close();

    // Interaction Integrals -- we will simply embed the integrals from dc_int to GW output file
    std::string old_int_dir = _dc_data_prefix + "." + std::to_string(imp_n);
    std::string vq_file_dc = old_int_dir + "/VQ_0.h5";
    size_t chunk_size = 0;
    size_t naux = 0;
    {
      h5pp::archive meta_data(old_int_dir + "/meta.h5", "r");
      meta_data["chunk_size"] >> chunk_size;
      meta_data.close();
    }
    {
      h5pp::archive dc_input(old_int_dir + "/dummy.h5", "r");
      dc_input["params/NQ"] >> naux;
      dc_input.close();
    }
    ztensor<4> vq_old(chunk_size, naux, nio, nio);
    h5pp::archive int_data_dc(vq_file_dc, "r");
    int_data_dc["0"] >> reinterpret_cast<double*>(vq_old.data());
    int_data_dc.close();
    ztensor<4> vq_new(chunk_size, naux, nao_eff, nao_eff);
    vq_new.set_zero();
    for (size_t c = 0; c < chunk_size; ++c) {
      for (size_t q = 0; q < naux; ++q) {
        for (size_t i = 0; i < nio; ++i) {
          for (size_t j = 0; j < nio; ++j) {
            vq_new(c, q, i, j) = vq_old(c, q, i, j);
          }
        }
      }
    }
    std::string new_int_dir = _root + "/gw." + std::to_string(imp_n) + ".df_int";
    std::string new_int_file = new_int_dir +  "/VQ_0.h5";
    std::filesystem::create_directories(new_int_dir);
    std::filesystem::copy_file(old_int_dir + "/meta.h5", new_int_dir + "/meta.h5",
                               std::filesystem::copy_options::overwrite_existing);
    h5pp::archive int_data_new(new_int_file, "w");
    int_data_new["0"] << vq_new.view<double>();
    int_data_new.close();
  }
} 