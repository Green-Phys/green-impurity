/*
 * Copyright (c) 2025 University of Michigan
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

#ifndef GREEN_INCHWORM_IMPURITY_SOLVER_H
#define GREEN_INCHWORM_IMPURITY_SOLVER_H

#include "common_defs.h"

namespace green::impurity {

  class inchworm_impurity_solver {
  public:
    inchworm_impurity_solver(const std::string& input_file, const std::string& impurity_solver_exec,
                             const std::string& impurity_solver_params, const std::string& root);

    std::tuple<ztensor<3>, ztensor<4>> solve(size_t imp_n, const grids::transformer_t& ft, double mu, const ztensor<3>& ovlp,
                                             const ztensor<3>& hcore_eff, const ztensor<3>& delta_1, const ztensor<4>& delta_w,
                                             const dtensor<4>& interaction, const ztensor<4>& g_w) const;

  private:
    std::string _input_file;
    std::string _impurity_solver_exec;
    std::string _impurity_solver_params;
    std::string _root;
    size_t      _nimp;
    dtensor<2>  _uxl;
  };

}  // namespace green::impurity
#endif  // GREEN_INCHWORM_IMPURITY_SOLVER_H
