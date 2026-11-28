# green-impurity
Wrapper for the impurity solver.

In addition to running impurity solver it also perform bath fitting routine if needed.

# Acknowledgements

This work is supported by National Science Foundation under the award OAC-2310582

### Bath fitting

ED and GW use the existing `NORM_L2_TRAPZ` bath fitter by default. Use
`--bath_fitting_method ESPRIT` to extract real bath energies with MiniPole's
real-frequency mapping and fit nonnegative residues. Both approaches fit
the diagonal hybridization independently for each spin and orbital.

ESPRIT uses the per-orbital site counts in `bath_file` as pole budgets and sets
unused couplings to zero. It reads the existing file format and ignores the
initial bath parameter values. The optional `--bath_fitting_freq_cutoff`
limits the fitted Matsubara window as for the other methods. ESPRIT reports
active pole counts and weighted RMS and maximum errors on that window.
