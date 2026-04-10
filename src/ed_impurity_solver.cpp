#include <green/impurity/ed_impurity_solver.h>

#include <fstream>
#include <numeric>

namespace green::impurity {

  ed_impurity_solver::ed_impurity_solver(const std::string& input_file, const std::string& bath_file,
                                         const std::string& impurity_solver_exec, const std::string& impurity_solver_params,
                                         const std::string& root) :
      _input_file(input_file), _impurity_solver_exec(impurity_solver_exec), _impurity_solver_params(impurity_solver_params),
      _root(root) {
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
      // Store single-spin template; tiling to actual ns happens in solve()
      dtensor<2> initial_bath(1, bath.size());
      itensor<1> bath_struct(bath_structure.size());
      std::copy(bath.begin(), bath.end(), initial_bath(0).begin());
      std::copy(bath_structure.begin(), bath_structure.end(), bath_struct.begin());
      _initial_bath.push_back(initial_bath);
      _bath_structure.push_back(bath_struct);
    }
  }

  std::tuple<ztensor<3>, ztensor<4>> ed_impurity_solver::solve(size_t imp_n, const grids::transformer_t& ft, double mu,
                                                                const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
                                                                const ztensor<3>& delta_1, const ztensor<4>& delta_w,
                                                                const dtensor<4>& interaction, const ztensor<4>& g_w) const {
    ztensor<3> sigma_inf_new(delta_1.shape());
    ztensor<4> sigma_new(delta_w.shape());
    size_t     ns    = ovlp.shape()[0];
    size_t     nbath = _initial_bath[imp_n].shape()[1];
    dtensor<2> initial_bath_tiled(ns, nbath);
    for (size_t is = 0; is < ns; ++is)
      std::copy(_initial_bath[imp_n](0).begin(), _initial_bath[imp_n](0).end(), initial_bath_tiled(is).begin());
    auto [delta_out, bath_arr] =
        minimize(ft.sd().repn_fermi().wsample() * 1.0i, delta_w, initial_bath_tiled, _bath_structure[imp_n], 1);
    {
      std::ofstream ofile(_root + "/bath.dat", std::ios_base::out);
      for (auto b : bath_arr) {
        ofile << b << " ";
      }
      ofile << std::endl;
    }
    size_t                  nio = ovlp.shape()[2];
    size_t                  nb  = std::reduce(_bath_structure[imp_n].begin(), _bath_structure[imp_n].end());
    dtensor<2>              Epsk(nb, ns);
    std::vector<dtensor<2>> Vk;
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
    ztensor<4> g0_imp(delta_out.shape());
    for (size_t iw = 0; iw < delta_out.shape()[0]; ++iw) {
      for (size_t is = 0; is < ns; ++is) {
        auto g_inv_w_imp =
            matrix(ovlp(is)) * (ft.wsample_fermi()(iw) * 1.0i + mu) - matrix(hcore_eff(is)) - matrix(delta_1(is)) - matrix(delta_out(iw, is));
        auto xxx               = g_inv_w_imp.inverse().eval();
        matrix(g0_imp(iw, is)) = xxx;
      }
    }
    {
      h5pp::archive data(_root + "/ed." + std::to_string(imp_n) + ".input.h5", "w");
      data["freq"] << ft.wsample_fermi();
      data["G0_imp/data"] << g0_imp.view<double>();
      data["G_imp/data"] << g_w;
      data["Delta/data"] << delta_out;
      data["Delta/data_in"] << delta_w;
      data["Delta/static"] << delta_1;

      itensor<2> sectors(1, 2);
      sectors(0, 0) = 0;
      sectors(0, 1) = 0;
      auto hop_g    = data["sectors"];
      hop_g["values"] << sectors;
      auto bath   = data["Bath"];
      // Post process H0->H0_imp
      auto H0_imp = ndarray::transpose(hcore_eff + delta_1, "sij->ijs").astype<double>();

      bath["Epsk/values"] << Epsk;
      for (size_t io = 0; io < nio; ++io) {
        bath["Vk_" + std::to_string(io) + "/values"] << Vk[io];
        data["H0_" + std::to_string(io) + "/values"] << H0_imp(io);
      }
      dtensor<6> interaction_(2, 2, nio, nio, nio, nio);

      // transform interaction into physics convention
      auto interaction_phys = ndarray::transpose(interaction, "ijkl->ikjl");

      interaction_(0, 0) << interaction_phys;
      interaction_(0, 1) << interaction_phys;
      interaction_(1, 0) << interaction_phys;
      interaction_(1, 1) << interaction_phys;
      data["interaction/values"] << interaction_;
      data["mu"] << mu;
      if (nio > 1) {
        // Store the list of ordered orbital pairs for which G(io, jo) should be calculated
        // NOTE: Diagonal part of G is handled differently
        itensor<2> orbitals(nio * nio - nio, 2);
        for (size_t io = 0, iii = 0; io < nio; ++io) {
          for (size_t jo = 0; jo < nio; ++jo) {
            if (io != jo) {
              orbitals(iii, 0) = io;
              orbitals(iii, 1) = jo;
              ++iii;
            }
          }
        }
        data["GreensFunction_orbitals/values"] << orbitals;
      }
      data.close();
    }
    std::string run = (_impurity_solver_exec + " " + _impurity_solver_params + " --NSITES=" + std::to_string(nio + nb) +
                       " --NSPINS=" + std::to_string(2) + " --INPUT_FILE=" + _root + "/ed." + std::to_string(imp_n) +
                       ".input.h5" + " --OUTPUT_FILE=" + _root + "/ed." + std::to_string(imp_n) + ".result.h5" +
                       " --arpack.SECTOR=false --siam.NORBITALS=" + std::to_string(nio) +
                       " --spinstorage.ORBITAL_NUMBER=" + std::to_string(nio) +
                       " --lanc.BETA=" + std::to_string(ft.sd().beta()));
    int sysresult = std::system(run.c_str());
    if (std::filesystem::exists(_root + "/ed." + std::to_string(imp_n) + ".result.h5")) {
      h5pp::archive ar(_root + "/ed." + std::to_string(imp_n) + ".result.h5", "r");
      dtensor<3>    xxx;
      ar["results/Sigma_inf_ij"] >> xxx;
      sigma_inf_new.resize(xxx.shape());
      sigma_inf_new << xxx;
      ar["results/Sigma_ij"] >> sigma_new.view<double>();
    } else {
      std::cerr << "Impurity result file has not been found" << std::endl;
    }
    return std::make_tuple(sigma_inf_new, sigma_new);
  }

}  // namespace green::impurity
