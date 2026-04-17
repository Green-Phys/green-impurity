#ifndef GREEN_ED_IMPURITY_SOLVER_H
#define GREEN_ED_IMPURITY_SOLVER_H

#include "common_defs.h"

namespace green::impurity {
  class ed_impurity_solver {
  public:
    ed_impurity_solver(const std::string& input_file, const std::string& bath_file, const std::string& impurity_solver_exec,
                       const std::string& impurity_solver_params, const std::string& root,
                       bath_fitting_method bf_method = NORM_L2_TRAPZ, double bf_freq_cutoff = -1.0);

    std::tuple<ztensor<3>, ztensor<4>> solve(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<3>& ovlp,
                                             const ztensor<3>& hcore_eff, const ztensor<3>& delta_1, const ztensor<4>& delta_w,
                                             const dtensor<4>& interaction, const ztensor<4>& g_w) const;

  private:
    std::string             _input_file;
    std::string             _impurity_solver_exec;
    std::string             _impurity_solver_params;
    std::string             _root;
    bath_fitting_method     _bf_method;
    double                  _bf_freq_cutoff;
    std::vector<dtensor<2>> _initial_bath;
    std::vector<itensor<1>> _bath_structure;
  };

}  // namespace green::impurity

#endif  // GREEN_ED_IMPURITY_SOLVER_H
