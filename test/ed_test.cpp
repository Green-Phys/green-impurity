/*
 * Fake ED impurity solver used by the impurity_solver test suite.
 *
 * Mimics the command-line contract of the real ED solver: reads --INPUT_FILE=<path>
 * and --OUTPUT_FILE=<path>, determines the required output shapes from the input
 * file, and writes zero-valued results/Sigma_inf_ij and results/Sigma_ij so that
 * ed_impurity_solver::solve() can read them back and return normally.
 *
 * Using a real helper binary (instead of pre-seeding the output file from the
 * test body) exercises the full solver code path: command construction, child
 * process launch, and post-run read.
 */

#include <green/h5pp/archive.h>
#include <green/impurity/common_defs.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::string input_file;
  std::string output_file;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto        eq = arg.find('=');
    if (eq == std::string::npos) continue;
    std::string key = arg.substr(0, eq);
    std::string val = arg.substr(eq + 1);
    if (key == "--INPUT_FILE") input_file = val;
    else if (key == "--OUTPUT_FILE") output_file = val;
  }
  if (input_file.empty() || output_file.empty()) {
    std::cerr << "ed_test: required args --INPUT_FILE=<path> --OUTPUT_FILE=<path>" << std::endl;
    return 1;
  }

  // Derive (ns, naso, nw) from the input file written by ed_impurity_solver.
  green::impurity::ztensor<3> delta_static;
  green::impurity::ztensor<4> delta_in;
  {
    green::h5pp::archive ar(input_file, "r");
    ar["Delta/static"] >> delta_static;
    ar["Delta/data_in"] >> delta_in;
    ar.close();
  }
  size_t ns   = delta_static.shape()[0];
  size_t naso = delta_static.shape()[1];
  size_t nw   = delta_in.shape()[0];

  // Zero-valued results, written as results/<name>/{data,shape} with a flat
  // row-major data buffer, matching the real ED solver's output layout.
  // Sigma_inf_ij is real (ns, naso, naso); Sigma_ij is the real view of the
  // complex self-energy (nw, ns, naso, naso), i.e. last dim doubled.
  std::vector<double> sigma_inf(ns * naso * naso, 0.0);
  std::vector<double> sigma_ij(nw * ns * naso * naso * 2, 0.0);

  green::h5pp::archive out(output_file, "w");
  out["results/Sigma_inf_ij/data"]  << sigma_inf;
  out["results/Sigma_inf_ij/shape"] << std::vector<size_t>{ns, naso, naso};
  out["results/Sigma_ij/data"]      << sigma_ij;
  out["results/Sigma_ij/shape"]     << std::vector<size_t>{nw, ns, naso, 2 * naso};
  out.close();
  return 0;
}
