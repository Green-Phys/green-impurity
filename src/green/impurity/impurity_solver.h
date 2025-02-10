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

#include <green/grids/transformer_t.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>

#include <tuple>

#include "bath_fitting.h"

namespace green::impurity {

  template <size_t N>
  using ztensor = ndarray::ndarray<std::complex<double>, N>;
  template <size_t N>
  using dtensor = ndarray::ndarray<double, N>;
  template <size_t N>
  using itensor    = ndarray::ndarray<int, N>;
  using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  template <typename prec>
  using MMatrixX = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  template <typename prec>
  using CMMatrixX = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  template <typename prec, typename = std::enable_if_t<std::is_same_v<prec, std::remove_const_t<prec>>>>
  auto matrix(ndarray::ndarray<prec, 2>&& array) {
    return MMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const ndarray::ndarray<const prec, 2>& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(ndarray::ndarray<const prec, 2>&& array) {
    return CMMatrixX<prec>(const_cast<prec*>(array.data()), array.shape()[0], array.shape()[1]);
  }

  template <typename prec>
  auto matrix(const ndarray::ndarray<prec, 2>& array) {
    return CMMatrixX<prec>(array.data(), array.shape()[0], array.shape()[1]);
  }

  using green_dc_func = std::function<void(
        std::string, int imp_n, utils::shared_object<ztensor<5>>&, ztensor<4>&, utils::shared_object<ztensor<5>>&)>;

  class ed_impurity_solver {
  public:
    ed_impurity_solver(const std::string& input_file, const std::string& bath_file, const std::string& impurity_solver_exec,
                       const std::string& impurity_solver_params, const std::string& root) :
        _input_file(input_file), _impurity_solver_exec(impurity_solver_exec), _impurity_solver_params(impurity_solver_params),
        _root(root) {
      size_t        ns = 2;
      size_t        nimp;
      h5pp::archive ar(input_file, "r");
      ar["nimp"] >> nimp;
      ar.close();
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

    auto solve(size_t imp_n, const grids::transformer_t& _ft, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
               const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction, const ztensor<4>& g_w) const {
      ztensor<3> sigma_inf_new(delta_1.shape());
      ztensor<4> sigma_new(delta_w.shape());
      auto [delta_out, bath_arr] =
          minimize(_ft.sd().repn_fermi().wsample() * 1.0i, delta_w, _initial_bath[imp_n], _bath_structure[imp_n], 1);
      {
        std::ofstream ofile(_root + "/bath.dat", std::ios_base::out);
        for (auto b : bath_arr) {
          ofile << b << " ";
        }
        ofile << std::endl;
      }
      size_t                  nio = ovlp.shape()[2];
      size_t                  ns  = ovlp.shape()[0];
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
              matrix(ovlp(is)) * (_ft.wsample_fermi()(iw) * 1.0i + mu) - matrix(h_core(is)) - matrix(delta_out(iw, is));
          auto g_inv_w_loc       = matrix(g_w(iw, is)).inverse().eval();
          auto xxx               = g_inv_w_imp.inverse().eval();
          matrix(g0_imp(iw, is)) = xxx;
        }
      }
      {
        h5pp::archive data(_root + "/ed." + std::to_string(imp_n) + ".input.h5", "w");
        data["freq"] << _ft.wsample_fermi();
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
        auto H0_imp = ndarray::transpose(h_core + delta_1, "sij->ijs").astype<double>();

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
      std::string run       = (_impurity_solver_exec + " " + _impurity_solver_params + " --NSITES=" + std::to_string(nio + nb) +
                         " --NSPINS=" + std::to_string(2) + " --INPUT_FILE=" + _root + "/ed." + std::to_string(imp_n) +
                         ".input.h5" + " --OUTPUT_FILE=" + _root + "/ed." + std::to_string(imp_n) + ".result.h5" +
                         " --arpack.SECTOR=false"
                               " --siam.NORBITALS=" +
                         std::to_string(nio) + " --spinstorage.ORBITAL_NUMBER=" + std::to_string(nio) +
                         " --lanc.BETA=" + std::to_string(_ft.sd().beta()));
      int         sysresult = std::system(run.c_str());
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

  private:
    std::string             _input_file;
    std::string             _impurity_solver_exec;
    std::string             _impurity_solver_params;
    std::string             _root;
    size_t                  _nimp;
    std::vector<dtensor<2>> _initial_bath;
    std::vector<itensor<1>> _bath_structure;
  };

  class basic_dc_solver {
  public:
    basic_dc_solver(const std::string& input_file, const std::string& dc_solver_exec, const std::string& dc_solver_params,
                    const std::string& root) :
        _input_file(input_file), _dc_solver_exec(dc_solver_exec), _dc_solver_params(dc_solver_params), _root(root) {};

    auto solve(size_t imp_n, const grids::transformer_t& _ft, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
               const dtensor<4>& interaction, const ztensor<3>& sigma_inf, const ztensor<4>& sigma_w,
               const ztensor<4>& g_w, green_dc_func& green_solver) const {
      h5pp::archive fff(_root + "/dc." + std::to_string(imp_n) + ".input.h5", "w");
      size_t        ns      = ovlp.shape()[0];
      size_t        naso    = ovlp.shape()[1];

      ztensor<3>    ovlp_   = ovlp;
      ztensor<3>    h_core_ = h_core;
      ztensor<4>    g_tau_full(_ft.sd().repn_fermi().nts(), ns, naso, naso);
      dtensor<4>    g_tau(_ft.sd().repn_fermi().nts() - 2, ns, naso, naso);
      ztensor<3>    sigma_inf_new(sigma_inf.shape());
      ztensor<4>    sigma_new(_ft.sd().repn_fermi().nw(), ns, naso, naso);
      ztensor<4>    sigma_tau(_ft.sd().repn_fermi().nts(), ns, naso, naso);

      _ft.omega_to_tau(g_w, g_tau_full);
      for (size_t it = 1, it2 = 0; it < _ft.sd().repn_fermi().nts() - 1; ++it, ++it2) {
        g_tau(it2) << g_tau_full(it).astype<double>();
      }

      auto g_ftau = ndarray::transpose(g_tau, "tsij->sjit");

      fff["uchem"] << interaction;
      fff["ovlp"] << ovlp_(0).astype<double>();
      fff["hcore"] << h_core_(0).astype<double>();
      fff["h0"] << h_core_.astype<double>();

      for (size_t is = 0; is < ovlp.shape()[0]; ++is) {
        fff["fock/" + std::to_string(is + 1)] << (h_core + sigma_inf)(is).astype<double>();
        // fff["eigval/"+ std::to_string(is+1)] = np.diag(F[s].real)
        // fff["eigvec/"+ std::to_string(is+1)] = F[s].real
        fff["gf/ftau/" + std::to_string(is + 1)] << g_ftau(is);
        fff["rho/" + std::to_string(is + 1)] << -g_tau_full(_ft.sd().repn_fermi().nts() - 1, is).astype<double>();
      }

      fff["e_nuclear"] << 0.0;
      fff["enuc"] << 0.0;
      fff["etot_hf"] << 0.0;
      fff["filling"] << 4.;
      fff["mu"] << mu;
      fff["e0"] << 0.;
      fff["nelectron"] << 4.;
      fff["spin"] << 0.;
      fff["esigma1"] << 0.;
      fff["esigma2"] << 0.;
      fff["restricted"] << false;
      fff.close();

      std::string command = _dc_solver_exec + " " + _dc_solver_params + " --maxiter " + std::to_string(1) + " --unrestricted " +
                            " --mode GW" + " --repr ir" + " --beta " + std::to_string(_ft.sd().beta()) + " --hf-input " + _root +
                            "/dc." + std::to_string(imp_n) + ".input.h5" + " --output " + _root + "/dc." + std::to_string(imp_n) +
                            ".result.h5";
      std::cout << "DC Command " << command << std::endl;
      std::system(command.c_str());
      if (std::filesystem::exists(_root + "/dc." + std::to_string(imp_n) + ".result.h5")) {
        h5pp::archive ar(_root + "/dc." + std::to_string(imp_n) + ".result.h5");
        for (size_t is = 0; is < 2; ++is) {
          dtensor<2> sigma_1_;
          dtensor<3> sigma_t_;
          ar["output/sigma1/" + std::to_string(is + 1)] >> sigma_1_;
          ar["output/sigma2/" + std::to_string(is + 1)] >> sigma_t_;
          sigma_inf_new(is) += ndarray::transpose(sigma_1_, "ji->ij");
          dtensor<3> xxx = ndarray::transpose(sigma_t_, "jit->tij");
          for (size_t it = 1; it < sigma_tau.shape()[0] - 1; ++it) {
            sigma_tau(it, is) << xxx(it - 1);
          }
        }
        ar.close();
        ztensor<4> sigma_w_(_ft.sd().repn_fermi().nw(), ns, naso, naso);
        _ft.tau_to_omega(sigma_tau, sigma_new);
      } else {
        std::cerr << "Double counting result file has not been found" << std::endl;
      }
      return std::make_tuple(sigma_inf_new, sigma_new);
    }

  private:
    std::string _input_file;
    std::string _dc_solver_exec;
    std::string _dc_solver_params;
    std::string _root;
  };

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
      std::shared_ptr<void> ed_solver(new ed_impurity_solver(p["seet_input"], p["bath_file"], p["impurity_solver_exec"],
                                                             p["impurity_solver_params"], p["seet_root_dir"]));
      _impurity_call = [ed_solver, this](size_t imp_n, double mu, const ztensor<3>& ovlp, const ztensor<3>& h_core,
                                         const ztensor<3>& delta_1, const ztensor<4>& delta_w, const dtensor<4>& interaction,
                                         const ztensor<4>& g_w) -> std::tuple<ztensor<3>, ztensor<4>> {
        return static_cast<ed_impurity_solver*>(ed_solver.get())
            ->solve(imp_n, _ft, mu, ovlp, h_core, delta_1, delta_w, interaction, g_w);
      };
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
    size_t     naso = g_w.shape()[1];
    ztensor<3> delta_1(h_core.shape());
    ztensor<4> delta(g_w.shape());
    for (size_t iw = 0; iw < nw; ++iw) {
      for (size_t is = 0; is < ns; ++is) {
        auto g_inv_w_imp = matrix(ovlp(is)) * (_ft.wsample_fermi()(iw) * 1.0i + mu) - matrix(h_core(is)) - matrix(sigma_inf(is)) -
                           matrix(sigma_w(iw, is));
        auto g_inv_w_loc      = matrix(g_w(iw, is)).inverse().eval();
        matrix(delta(iw, is)) = g_inv_w_imp - g_inv_w_loc;
        auto xxx              = g_inv_w_imp.inverse().eval();
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
      g_dc.fence();
      g_dc.object() << g_as.reshape(shape_in);
      g_dc.fence();

      ztensor<4> sigma_inf_dc(shape_in_inf);
      _dc_solver(_dc_data_prefix, imp, g_dc, sigma_inf_dc, sigma_dc);
      sigma_inf_new -= sigma_inf_dc.reshape(shape_out_inf);
      _ft.omega_to_tau(sigma_w_new, sigma_as);
      sigma_as -= sigma_dc.object().reshape(shape_out);
      for (size_t is = 0; is < ns; ++is) {
        matrix(sigma_inf_loc_new(is)) += matrix(uu).transpose() * matrix(sigma_inf_new(is)) * matrix(uu);
      }
      for (size_t it = 0; it < nt; ++it) {
        for (size_t is = 0; is < ns; ++is) {
          matrix(sigma_w_loc_new(it, is)) += matrix(uu).transpose() * matrix(sigma_as(it, is)) * matrix(uu);
        }
      }
      std::cout << "Imputity " << imp << " finished" << std::endl;
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
