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

#define CATCH_CONFIG_RUNNER
#include "green/impurity/impurity_solver.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include <mpi.h>

namespace green::impurity {
  ztensor<4> compute_local_obj(const ztensor<5>& obj, const ztensor<3>& x_k, const grids::transformer_t& ft, const bz_utils_t& bz_utils, size_t ns, size_t nso) {
    ztensor<4> obj_loc(ft.sd().repn_fermi().nts(), ns, nso, nso);
    for (size_t it = 0; it < obj.shape()[0]; ++it) {
      for (size_t is = 0; is < obj.shape()[1]; ++is) {
        auto obj_full = bz_utils.ibz_to_full(obj(it, is));
        auto x_k_full = bz_utils.ibz_to_full(x_k);
        for (size_t ik = 0; ik < obj_full.shape()[0]; ++ik) {
          matrix(obj_loc(it, is)) += matrix(x_k_full(ik)) * matrix(obj_full(ik)) * matrix(x_k_full(ik)).adjoint();
        }
      }
    }
    obj_loc /= bz_utils.nk();
    return obj_loc;
  }

  ztensor<3> compute_local_obj(const ztensor<4>& obj, const ztensor<3>& x_k, const bz_utils_t& bz_utils, size_t ns, size_t nso, bool ibz=true) {
    ztensor<3> obj_loc(ns, nso, nso);
    for (size_t is = 0; is < obj.shape()[0]; ++is) {
      auto obj_full = ibz ? bz_utils.ibz_to_full(obj(is)) : obj(is).copy();
      auto x_k_full = bz_utils.ibz_to_full(x_k);
      for (size_t ik = 0; ik < obj_full.shape()[0]; ++ik) {
        matrix(obj_loc(is)) += matrix(x_k_full(ik)) * matrix(obj_full(ik)) * matrix(x_k_full(ik)).adjoint();
      }
    }
    obj_loc /= bz_utils.nk();
    return obj_loc;
  }
}

TEST_CASE("Impurity Solver") {
  std::string test_file   = TEST_PATH + "/data.h5"s;
  std::string bath_file   = TEST_PATH + "/bath.txt"s;
  std::string input_file   = TEST_PATH + "/transform.h5"s;
  std::string weak_input_file   = TEST_PATH + "/input.h5"s;
  std::string weak_sim_file   = TEST_PATH + "/data.h5"s;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  green::params::params p;
  green::symmetry::define_parameters(p);
  green::grids::define_parameters(p);
  p.define<bool>("spin_symm", "", false);
  p.define<std::string>("bath_file", "", bath_file);
  p.define<std::string>("impurity_solver_exec", "", "/bin/true");
  p.define<std::string>("impurity_solver_params", "", "");
  p.define<std::string>("dc_solver_exec", "", "/bin/true");
  p.define<std::string>("dc_solver_params", "", "");
  p.define<std::string>("dc_data_prefix", "", "");
  p.define<std::string>("seet_root_dir", "", TEST_PATH + ""s);
  p.define<std::string>("seet_input", "", input_file);

  p.parse("test --BETA 100 --grid_file " + grid_file + " --input_file " + weak_input_file );

  green::grids::transformer_t ft(p);
  green::impurity::bz_utils_t bz_utils(p);
  auto dummy_dc = [](std::string, int, green::utils::shared_object<green::impurity::ztensor<5>>&, green::impurity::ztensor<4>&, green::utils::shared_object<green::impurity::ztensor<5>>&) {
    return;
  };
  green::impurity::impurity_solver solver(p, ft, bz_utils, dummy_dc);
  green::impurity::ztensor<4> sigma1_k;
  green::impurity::ztensor<5> sigma_k;
  green::impurity::ztensor<5> g_k;
  green::impurity::ztensor<4> h_core_k;
  green::impurity::ztensor<4> ovlp_k;
  green::impurity::ztensor<3> x_k;
  green::impurity::ztensor<3> x_inv_k;
  double mu;
  {
    green::h5pp::archive ar(weak_sim_file, "r");
    ar["data/G_tau/data"] >> g_k;
    ar["data/Selfenergy/data"] >> sigma_k;
    ar["data/Sigma1"] >> sigma1_k;
    ar["data/mu"] >> mu;
    ar.close();
    ar.open(weak_input_file, "r");
    green::impurity::dtensor<5> xx;
    ar["HF/H-k"] >> xx;
    std::array<size_t, 4> shape{};
    std::copy(xx.shape().begin(), xx.shape().end()-1, shape.begin());
    h_core_k = xx.view<std::complex<double>>().reshape(shape).copy();
    ar["HF/S-k"] >> xx;
    ovlp_k = xx.view<std::complex<double>>().reshape(shape).copy();
    ar.close();
    ar.open(input_file, "r");
    ar["X_k"] >> x_k;
    ar["X_inv_k"] >> x_inv_k;
  }
  size_t ns = ovlp_k.shape()[0];
  size_t nso = ovlp_k.shape()[2];
  auto sigma = green::impurity::compute_local_obj(sigma_k, x_k, ft, bz_utils, ns, nso);
  auto g = green::impurity::compute_local_obj(g_k, x_inv_k, ft, bz_utils, ns, nso);
  auto ovlp = green::impurity::compute_local_obj(ovlp_k, x_k, bz_utils, ns, nso, false);
  auto h_core = green::impurity::compute_local_obj(h_core_k, x_k, bz_utils, ns, nso, false);
  auto sigma1 = green::impurity::compute_local_obj(sigma1_k, x_k, bz_utils, ns, nso);

  solver.solve(mu, ovlp, h_core, sigma1, sigma, g);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}