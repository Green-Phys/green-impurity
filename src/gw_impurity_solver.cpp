#include <filesystem>
#include <fstream>
#include <numeric>
#include "green/impurity/bath_fitting.h"
#include "green/impurity/gw_impurity_solver.h"

namespace green::impurity {

  gw_impurity_solver::gw_impurity_solver(const std::string& input_file, const std::string& bath_file,
                                         const std::string& impurity_solver_exec, const std::string& impurity_solver_params,
                                         const std::string dc_data_prefix, const std::string& root,
                                         bath_fitting_method bf_method, double bf_freq_cutoff) :
      _input_file(input_file), _impurity_solver_exec(impurity_solver_exec), _impurity_solver_params(impurity_solver_params),
      _dc_data_prefix(dc_data_prefix), _root(root), _bf_method(bf_method), _bf_freq_cutoff(bf_freq_cutoff) {
    size_t        ns = 2;  // TODO: This is hardcoded - let it be for now
    size_t        nimp;
    h5pp::archive ar(input_file, "r");
    ar["nimp"] >> nimp;
    ar.close();
    if (!std::filesystem::exists(bath_file)) throw std::runtime_error("Bath structure file does not exist");
    std::ifstream ff(bath_file);
    for (size_t imp = 0; imp < nimp; ++imp) {
      std::vector<double> bath;
      std::vector<int>    bath_structure;
      size_t              nio, nbo;
      ff >> nio >> nbo;
      for (size_t io = 0; io < nio; ++io) {
        int nb_io;
        ff >> nb_io;
        bath_structure.push_back(nb_io);
      }
      for (size_t bo = 0; bo < nbo * 2; ++bo) {
        double b;
        ff >> b;
        bath.push_back(b);
      }
      dtensor<2> initial_bath(ns, bath.size());
      itensor<1> bath_struct(bath_structure.size());
      // first spin
      std::copy(bath.begin(), bath.end(), initial_bath(0).begin());
      // second spin
      std::copy(bath.begin(), bath.end(), initial_bath(1).begin());
      std::copy(bath_structure.begin(), bath_structure.end(), bath_struct.begin());
      _initial_bath.push_back(initial_bath);
      _bath_structure.push_back(bath_struct);
    }
  }

  std::tuple<ztensor<3>, ztensor<4>> gw_impurity_solver::solve(size_t imp_n, const grids::transformer_t& ft, double mu,
                                                                const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                                                                const ztensor<3>& delta_1, const ztensor<4>& delta_w,
                                                                const dtensor<4>& interaction, const ztensor<4>& g_w) const {
    ztensor<3> sigma_inf_new(delta_1.shape());
    ztensor<4> sigma_new(delta_w.shape());

    // Step 1: Bath fitting and parsing
    dtensor<2>              Epsk;
    std::vector<dtensor<2>> Vk;
    fit_and_parse_bath(imp_n, ft, mu, delta_w, ovlp, Epsk, Vk);
    // Step 2: Prepare hybrid system and write GW input
    prepare_gw_input(imp_n, ovlp, hcore_eff, delta_1, delta_w, Epsk, Vk, mu, interaction, g_w, ft);

    size_t      nio = ovlp.shape()[2];
    size_t      nb  = Epsk.shape()[0];
    size_t      ns  = Epsk.shape()[1];
    std::string run = (_impurity_solver_exec +
                       " --itermax 10 --const_density=false --scf_type=GW --E_thr=1e-5" +
                       " --mixing_weight=0.7 --restart=true" +
                       " --BETA=" + std::to_string(ft.sd().beta()) +
                       " --input_file=" + _root + "/gw." + std::to_string(imp_n) + ".input.h5" +
                       " --results_file=" + _root + "/gw." + std::to_string(imp_n) + ".sim.h5") +
                      " --dfintegral_file=" + _root + "/gw." + std::to_string(imp_n) + ".df_int" +
                      " --dfintegral_hf_file=" + _root + "/gw." + std::to_string(imp_n) + ".df_int" +
                      " " + _impurity_solver_params;
    std::cout << "\n\n\nWill run the following command!" << std::endl;
    std::cout << run << std::endl;
    std::cout << "\n\n\n----" << std::endl;
    int sysresult = launchCleanChild(run);
    if (sysresult != 0) {
      throw impurity_solver_exec_error("GW impurity child process failed with code " + std::to_string(sysresult));
    }
    if (std::filesystem::exists(_root + "/gw." + std::to_string(imp_n) + ".sim.h5")) {
      h5pp::archive ar(_root + "/gw." + std::to_string(imp_n) + ".sim.h5", "r");
      size_t        it_num;
      ar["iter"] >> it_num;
      // Sigma1 is written as complex by sc_loop — read as ztensor, not dtensor.
      ztensor<4> sigma_inf_imp_plus_bath;
      ar["iter" + std::to_string(it_num) + "/Sigma1"] >> sigma_inf_imp_plus_bath;
      for (size_t is = 0; is < ns; ++is) {
        for (size_t i = 0; i < nio; ++i) {
          for (size_t j = 0; j < nio; ++j) {
            sigma_inf_new(is, i, j) = sigma_inf_imp_plus_bath(is, 0, i, j);
          }
        }
      }
      // Selfenergy/data is stored in tau domain (nt points) by green's sc_loop.
      // Extract the impurity orbital block and transform tau -> omega before returning.
      ztensor<5> sigma_tau_imp_plus_bath;
      ar["iter" + std::to_string(it_num) + "/Selfenergy/data"] >> sigma_tau_imp_plus_bath;
      size_t     nt_read = sigma_tau_imp_plus_bath.shape()[0];
      ztensor<4> sigma_tau_imp(nt_read, ns, nio, nio);
      for (size_t it = 0; it < nt_read; ++it) {
        for (size_t is = 0; is < ns; ++is) {
          for (size_t i = 0; i < nio; ++i) {
            for (size_t j = 0; j < nio; ++j) {
              sigma_tau_imp(it, is, i, j) = sigma_tau_imp_plus_bath(it, is, 0, i, j);
            }
          }
        }
      }
      ft.tau_to_omega(sigma_tau_imp, sigma_new);
    } else {
      throw impurity_result_not_found("GW impurity result file not found: " + _root + "/gw." + std::to_string(imp_n) + ".sim.h5");
    }
    return std::make_tuple(sigma_inf_new, sigma_new);
  }

  void gw_impurity_solver::fit_and_parse_bath(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<4>& delta_w, const ztensor<3>& ovlp,
                dtensor<2>& Epsk, std::vector<dtensor<2>>& Vk) const {
    // bath fitting
    // minimize() fits delta_w(iw) ~ Σ_k V_k^2 / (iw - eps_k) using bare Matsubara frequencies iw.
    // This yields fitted bath energies eps_k = eps_k_phys - mu (shifted by -mu relative to physical values).
    // In prepare_gw_input(), mu is explicitly subtracted from the impurity diagonal of h0_eff, so the
    // GW solver (which uses frequency iw, not iw+mu) sees the bath propagator (iw - eps_k_fit)^{-1}
    // = (iw + mu - eps_k_phys)^{-1}, which is correct.
    auto [delta_out, bath_arr] = minimize(
      ft.sd().repn_fermi().wsample() * 1.0i, delta_w, _initial_bath[imp_n], _bath_structure[imp_n], _bf_freq_cutoff, _bf_method
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
    // C++11 [complex.numbers] §26.4 guarantees std::complex<T> is layout-compatible with T[2],
    // so reinterpret_cast<double*> is well-defined and is the correct way to read a flat
    // double dataset from HDF5 into a complex ndarray without a shape-rank mismatch.
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