#ifndef GREEN_GW_IMPURITY_SOLVER_H
#define GREEN_GW_IMPURITY_SOLVER_H

#include <cstdlib>
#include <green/params/params.h>
#include "common_defs.h"
#include "bath_fitting.h"

namespace green::impurity {


  class gw_impurity_solver {
    template <typename prec>
    using MMatrixX = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    template <typename prec>
    using CMMatrixX = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  public:
    gw_impurity_solver(const std::string& input_file, const std::string& bath_file, const std::string& impurity_solver_exec,
                       const std::string& impurity_solver_params, const std::string dc_data_prefix, const std::string& root) :
        _input_file(input_file), _impurity_solver_exec(impurity_solver_exec), _impurity_solver_params(impurity_solver_params),
        _dc_data_prefix(dc_data_prefix), _root(root) {
      size_t        ns = 2; // TODO: This is hardcoded - let it be for now
      size_t        nimp;
      h5pp::archive ar(input_file, "r");
      ar["nimp"] >> nimp;
      ar.close();
      if(!std::filesystem::exists(bath_file)) throw std::runtime_error("Bath structure file does not exist");
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

    auto solve(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff,
               const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction, const ztensor<4>& g_w) const {
      ztensor<3> sigma_inf_new(delta_1.shape());
      ztensor<4> sigma_new(delta_w.shape());

      // Step 1: Bath fitting and parsing
      dtensor<2> Epsk;
      std::vector<dtensor<2>> Vk;
      fit_and_parse_bath(imp_n, ft, mu, delta_w, ovlp, Epsk, Vk);
      // Step 2: Prepare hybrid system and write GW input (generic future extension)
      prepare_gw_input(imp_n, ovlp, hcore_eff, delta_1, delta_w, Epsk, Vk, mu, interaction, g_w, ft);

      size_t nio = ovlp.shape()[2];
      size_t nb = Epsk.shape()[0];
      size_t ns = Epsk.shape()[1];
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
      // int sysresult = std::system(run.c_str());
      int sysresult = launchCleanChild(run);
      if (sysresult != 0) {
        throw std::runtime_error("Impurity GW child process failed with code " + std::to_string(sysresult));
      }
      if (std::filesystem::exists(_root + "/gw." + std::to_string(imp_n) + ".sim.h5")) {
        h5pp::archive ar(_root + "/gw." + std::to_string(imp_n) + ".sim.h5", "r");
        size_t it_num;
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
          size_t nt_read = sigma_tau_imp_plus_bath.shape()[0];
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
        std::cerr << "Impurity result file has not been found" << std::endl;
      }
      return std::make_tuple(sigma_inf_new, sigma_new);
    }

  private:
    static bool has_env(const char* var) {
      return std::getenv(var) != nullptr;
    }

    static std::string build_launcher_cmd(const std::string& cmd) {
      // SLURM takes priority: use srun to create a proper new step.
      if (has_env("SLURM_JOB_ID")) {
        return "srun --ntasks=1 --overlap " + cmd;
      }

      // Fallback: use single-rank mpirun.
      return "mpirun -np 1 " + cmd;
    }

    void fit_and_parse_bath(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<4>& delta_w, const ztensor<3>& ovlp,
                dtensor<2>& Epsk, std::vector<dtensor<2>>& Vk) const;

    void prepare_gw_input(size_t imp_n, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff, const ztensor<3>& delta_1,
                const ztensor<4>& delta_w,
                          const dtensor<2>& Epsk, const std::vector<dtensor<2>>& Vk, double mu,
                          const dtensor<4>& interaction, const ztensor<4>& g_w, const grids::transformer_t& ft) const;

    int launchCleanChild(const std::string &cmd) const {
      // Must unset stale per-rank vars before calling launcher.
      unsetenv("PMI_FD");
      unsetenv("PMI_RANK");
      unsetenv("PMI_SIZE");
      unsetenv("PMI_PORT");
      unsetenv("PMIX_RANK");
      unsetenv("PMIX_SERVER_URI");
      unsetenv("PMIX_SERVER_URI2");
      unsetenv("OMPI_COMM_WORLD_RANK");
      unsetenv("OMPI_COMM_WORLD_SIZE");
      unsetenv("OMPI_COMM_WORLD_LOCAL_RANK");
      unsetenv("OMPI_COMM_WORLD_LOCAL_SIZE");
      unsetenv("OMPI_COMM_WORLD_NODE_RANK");
      unsetenv("OMPI_UNIVERSE_SIZE");

      std::string launch_cmd = build_launcher_cmd(cmd);
      return std::system(launch_cmd.c_str());
    }

    std::string             _input_file;
    std::string             _impurity_solver_exec;
    std::string             _impurity_solver_params;
    std::string             _dc_data_prefix;
    std::string             _root;
    size_t                  _nimp;
    std::vector<dtensor<2>> _initial_bath;
    std::vector<itensor<1>> _bath_structure;
  };
}  // namespace green::impurity
#endif  // GREEN_GW_IMPURITY_SOLVER_H