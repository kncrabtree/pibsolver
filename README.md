# pibsolver
Python scripts for solving Schrodinger equation for a particle in an arbitrary 1D potential using particle in a box basis functions and Simpson integration.

Each version of the script comes in two varieties: the normal one is intended to be used in a command-line or IDE environment like Spyder where the user can modify the script and run it.
The 'oo' version creates a PIBSolver class that contains its calculation variables internally, and calculations can be initiated by calling the calc function.
See the jupyter_example.ipynb file for an example of how to use the object-oriented version.

## PIBSolver v1

This version of the script is designed to calculate energy levels in units of the harmonic frequency hbar*omega, both of which are defined as 1.
When non-harmonic potentials, the eigenvalues should be considered relative energies.

## PIBSolver v2

This version of the script uses the bond length, dissociation energy, and force constant of a diatomic molecule from the NIST Chemistry Webbook to solve for its vibrational energy levels.
It is intended for use only with the harmonic and morse potentials, and the units of the eigenvalues are wavenumbers.

## PIBSolver v3

This script solves for the energy levels of the inversion mode of ammonia.
It requires as inputs the equilibrium N-H bond length and a text file containing the electronic potential energy as a function of the angle of the H atoms with respect to planarity.
Units are wavenumbers.
