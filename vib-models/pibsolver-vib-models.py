#!/usr/bin/python3

#
# pibsolver.py
#
# Solves the Schrodinger equation for a one-dimensional particle
# in an arbitrary potential using a finite basis of particle-
# in-a-box basis functions.
#
# Requires SciPy, NumPy, and matplotlib
#
# Adapted from EIGENSTATE1D (Fortran) by Nancy Makri
#
# This script generates a text file containing the eigenvalues
# and eigenvectors, a pdf graph of the energy spectrum, and
# a pdf graph showing the potential, eigenvalues, and
# eigenfunctions.
#
# Author: Kyle N. Crabtree
#

from scipy import linalg
import scipy.integrate as spi
import scipy.constants as spc
import numpy
from matplotlib import pyplot as plt
import datetime
import tabulate
import pandas as pd


class PIBSolver:
    def __init__(self):
        # --------------------------- VARIABLES  -
        #        ---------------------------

        self.re = 1.481
        self.mass = (32.*16.)/(32.+16.)

        # minimum x value for partice in a box calculation
        self.xmin = -0.7+self.re

        # maximum x value for particle in a box calculation
        # there is no limit, but if xmax<xmin, the values will be swapped
        # if xmax = xmin, then xmax will be set to xmin + 1
        self.xmax = 0.7+self.re

        # number of grid points at which to calculate integral
        # must be an odd number. If an even number is given, 1 will be added.
        # minimum number of grid points is 3
        self.ngrid = 1001

        # number of PIB wavefunctions to include in basis set
        #minimum is 1
        self.nbasis = 250

        # if you want to control the plot ranges, you can do so here
#        make_plots = True
        self.plotymin = 0
        self.plotymax = 0
        self.plotxmin = 0
        self.plotxmax = 0
        
        self.plotevery = 1

        # force constant (N/m)
        self.fk = 829.9849

        # morse potential dissociation energy (cm-1)
        self.de = 51500.

        self.V = self.harmonic

        self.evalues = []
        self.evectors = []

        self.print_vars()
        
        print(f'Perform a calculation with pb.calc(), or change any of the variables in the table above with pb.variable=x.\n\nExamples:\n\npb.fk = 500.0\n\npb.xmin = pb.re - 0.4\n\npb.mass = (12.0*16.0)/(12.0+16.0)\n\npb.V = pb.morse\n\npb.V = pb.harmonic\n\nTo re-print the table, use pb.print_vars().')

    # --------------------------
    #  POTENTIAL DEFINITIONS  -
    # --------------------------

    # definitions of potential functions

    # purely harmonic potential
    def harmonic(self, x):
        return self.prefactor*(x-self.re)*(x-self.re)

    # Morse potential
    def morse(self, x):
        return self.de*(1.-numpy.exp(-self.alpha*(x-self.re)))**2.

    def print_vars(self):
        # verify that inputs are sane
        if self.xmax == self.xmin:
            self.xmax = self.xmin+1
        elif self.xmax < self.xmin:
            self.xmin, self.xmax = self.xmax, self.xmin

        self.ngrid = max(self.ngrid, 3)

        if self.ngrid % 2 == 0:
            self.ngrid += 1

        self.nbasis = max(1, self.nbasis)

        # calculate some useful values
        self.L = self.xmax - self.xmin
        self.tl = numpy.sqrt(2./self.L)
        self.pixl = numpy.pi/self.L
        self.omega = numpy.sqrt(self.fk/self.mass)
        self.alpha = numpy.sqrt(self.fk/2.0/(self.de*spc.h*spc.c*100.0))*1e-10

        self.prefactor = 0.5 * self.fk * 1e-20/(spc.h*spc.c*100.0)

        td = [["re", self.re],
              ["xmin", self.xmin, "xmax", self.xmax],
              ["ngrid", self.ngrid, "nbasis", self.nbasis],
              ["mass", self.mass, "fk", self.fk],
              ["de", self.de, "V", self.V.__name__],
              ["plotxmin", self.plotxmin, "plotxmax", self.plotxmax],
              ["plotymin", self.plotymin, "plotymax", self.plotymax],
              ["plotevery",self.plotevery]]

        print(tabulate.tabulate(td))

    def pib(self, x, n):
        return self.tl*numpy.sin(n*x*self.pixl)

# ------------------------
#   BEGIN CALCULATION   -
# ------------------------
    def calc(self, make_plots=True):
        print("Beginning calculation with these parameters:")

        self.print_vars()
        self.evalues = []
        self.evectors = []

        # get current time
        starttime = datetime.datetime.now()

        # create grid
        x = numpy.linspace(self.xmin, self.xmax, self.ngrid)

        # create Hamiltonian matrix; fill with zeros
        H = numpy.zeros((self.nbasis, self.nbasis))

        # split Hamiltonian into kinetic energy and potential energy terms
        #H = T + V
        # V is defined above, and will be evaluated numerically

        # Compute kinetic energy
        # The Kinetic enery operator is T = -hbar/2m d^2/dx^2
        # in the PIB basis, T|phi(i)> = hbar/2m n^2pi^2/L^2 |phi(i)>
        # Therefore <phi(k)|T|phi(i)> = hbar/2m n^2pi^2/L^2 delta_{ki}
        # Kinetic energy is diagonal
        idx = numpy.arange(self.nbasis)+1
        kepf = spc.hbar*spc.hbar*numpy.pi**2. / \
            (2.*self.mass*spc.physical_constants['atomic mass constant'][0]*(
                self.L*1e-10)*(self.L*1e-10)*spc.h*spc.c*100)
        H = kepf*numpy.eye(self.nbasis)*idx*idx
        na = numpy.newaxis

        # now, add in potential energy matrix elements <phi(j)|V|phi(i)>
        # that is, multiply the two functions by the potential at each grid point and integrate
        H += spi.simps(self.pib(x[na,na,:]-self.xmin,idx[na,:,na])*self.V(x[na,na,:])*self.pib(x[na,na,:]-self.xmin,idx[:,na,na]), x, axis=2)


        # Solve for eigenvalues and eigenvectors
        self.evalues, self.evectors = linalg.eigh(H)

        # get ending time
        endtime = datetime.datetime.now()

        print("Calculation completed in "+str(endtime-starttime))
        print("First 10 Eigenvalues (to see all, use pb.evalues):")

        pl = []
        rowlen = 5
        if len(self.evalues) > 100:
            rowlen = 10
        while rowlen*(len(pl)+1) < len(self.evalues):
            pl.append(self.evalues[rowlen*(len(pl)):rowlen*(len(pl)+1)])

        pl.append(self.evalues[rowlen*(len(pl)):])

        #print(tabulate.tabulate(pl, floatfmt=".3f"))

        if(make_plots):
            self.plot()
            
        df = pd.DataFrame()
        return df.assign(E=self.evalues[0:10])

    def plot(self):
        if len(self.evalues) < 1:
            print("Cannot generate plots because there are no calculated eigenvalues.")
            return

        self.fig, self.axes = plt.subplots(1,2,figsize=(12,4),dpi=300)

        
        # Make graph of eigenvalue spectrum
        title = "EV Spectrum, Min={:.4f}, Max={:.4f}, Grid={:d}, Basis={:d}".format(
            self.xmin, self.xmax, self.ngrid, self.nbasis)

        self.axes[0].plot(self.evalues, 'ro')
        self.axes[0].set_xlabel('v')
        self.axes[0].set_ylabel(r'E (cm$^{-1}$)')
        self.axes[0].set_title(title)

        # Make graph with potential and eigenfunctions
        title = "Wfns, Min={:.4f}, Max={:.4f}, Grid={:d}, Basis={:d}".format(
            self.xmin, self.xmax, self.ngrid, self.nbasis)
        x = numpy.linspace(self.xmin, self.xmax, self.ngrid)
        self.axes[1].plot(x, self.V(x))

        pxmin = self.plotxmin
        pxmax = self.plotxmax
        pymin = self.plotymin
        pymax = self.plotymax

        if pxmin == 0:
            pxmin = self.xmin
        if pxmax == 0:
            pxmax = self.xmax
        if(pxmin == pxmax):
            pxmin, pxmax = self.xmin, self.xmax
        if(pxmin > pxmax):
            pxmin, pxmax = pxmax, pxmin
        if pymax == 0:
            pymax = 1.25*self.de
        if(pymin > pymax):
            pymin, pymax = pymax, pymin

        if self.nbasis > 2:
            sf = (self.evalues[2]-self.evalues[0])/2.25
        else:
            sf = 1.

        self.axes[1].hlines(self.evalues,self.xmin,self.xmax,color='black')
        idx = numpy.arange(self.nbasis)
        na = numpy.newaxis
        self.axes[1].plot( x[:,na],(sf*numpy.sum(self.evectors[idx[na,:],idx[:,na],na]*self.pib(x[na,:]-self.xmin,idx[:,na]+1),axis=1) + self.evalues[idx,na]).T)

        self.axes[1].vlines([self.xmin,self.xmax],pymin,pymax,color='black')
        self.axes[1].set_xlim(pxmin,pxmax)
        self.axes[1].set_ylim(pymin, pymax)
        self.axes[1].set_title(title)
        self.axes[1].set_xlabel(r'$R$ (Angstrom)')
        self.axes[1].set_ylabel(r'V (cm$^{-1}$)')
        
        self.fig.tight_layout()

        
print('PIBSolver initialized. Create a new object with "pb = PIBSolver()"')