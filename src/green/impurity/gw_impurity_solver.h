#ifndef GREEN_GW_IMPURITY_SOLVER_H
#define GREEN_GW_IMPURITY_SOLVER_H

#include <array>
#include <cstdlib>
#include "common_defs.h"

namespace green::impurity {

  class gw_impurity_solver {
  public:
    gw_impurity_solver(const std::string& input_file, const std::string& bath_file, const std::string& impurity_solver_exec,
                       const std::string& impurity_solver_params, const std::string dc_data_prefix, const std::string& root,
                       bath_fitting_method bf_method = NORM_L2_TRAPZ, double bf_freq_cutoff = -1.0);

    std::tuple<ztensor<3>, ztensor<4>> solve(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<3>& ovlp,
                                             const ztensor<3>& hcore_eff, const ztensor<3>& delta_1, const ztensor<4>& delta_w,
                                             const dtensor<4>& interaction, const ztensor<4>& g_w) const;

  private:
    static bool has_env(const char* var) { return std::getenv(var) != nullptr; }

    static std::string build_launcher_cmd(const std::string& cmd) {
      if (has_env("SLURM_JOB_ID")) return "srun --ntasks=1 --overlap " + cmd;
      return cmd;
    }

    void fit_and_parse_bath(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<4>& delta_w,
                            const ztensor<3>& ovlp, dtensor<2>& Epsk, std::vector<dtensor<2>>& Vk) const;

    void prepare_gw_input(size_t imp_n, const ztensor<3>& ovlp, const ztensor<3>& hcore_eff, const ztensor<3>& delta_1,
                          const ztensor<4>& delta_w, const dtensor<2>& Epsk, const std::vector<dtensor<2>>& Vk, double mu,
                          const dtensor<4>& interaction, const ztensor<4>& g_w, const grids::transformer_t& ft) const;

    int launchCleanChild(const std::string& cmd) const {
      static constexpr std::array<const char*, 13> mpi_vars = {
          "PMI_FD", "PMI_RANK", "PMI_SIZE", "PMI_PORT",
          "PMIX_RANK", "PMIX_SERVER_URI", "PMIX_SERVER_URI2",
          "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",
          "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE",
          "OMPI_COMM_WORLD_NODE_RANK", "OMPI_UNIVERSE_SIZE"};
      std::string env_prefix = "env";
      for (const char* var : mpi_vars) {
        if (has_env(var)) env_prefix += std::string(" -u ") + var;
      }
      return std::system((env_prefix + " " + build_launcher_cmd(cmd)).c_str());
    }

    std::string             _input_file;
    std::string             _impurity_solver_exec;
    std::string             _impurity_solver_params;
    std::string             _dc_data_prefix;
    std::string             _root;
    bath_fitting_method     _bf_method;
    double                  _bf_freq_cutoff;
    std::vector<dtensor<2>> _initial_bath;
    std::vector<itensor<1>> _bath_structure;
  };

}  // namespace green::impurity
#endif  // GREEN_GW_IMPURITY_SOLVER_H
