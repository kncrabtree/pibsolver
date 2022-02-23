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
import scipy.interpolate as spp
import numpy
from matplotlib import pyplot as plt
import datetime
import tabulate
import os


class PIBSolver:
    def __init__(self):
        #---------------------------
        #        VARIABLES         -
        #---------------------------
        
        self.re = 0.998
        self.hisotope = 1.007825
        
        self.make_plots=True
        
        
        #number of grid points at which to calculate integral
        #must be an odd number. If an even number is given, 1 will be added.
        #minimum number of grid points is 3
        self.ngrid = 151
        
        #number of PIB wavefunctions to include in basis set
        #minimum is 1
        self.nbasis = 20
        
        #if you want to control the plot ranges, you can do so here
#        make_plots = True
        self.plotymin = 0
        self.plotymax = 3500
        self.plotxmin = 0
        self.plotxmax = 0
        
        self.potfile = "nh3_scf_aug-cc-pvtz.txt"
        
        self.evalues=[]
        self.evectors=[]
        
        self.print_vars()
        
        files = os.listdir()
        
        print(f'Perform a calculation with pb.calc(), or change any of the variables in the table above with pb.variable=x.\n\nExamples:\n\npb.potfile="nh3_scf_aug-cc-pvtz.txt" \npb.re = 0.998\n\nTo re-print the table, use pb.print_vars().\n\nDetected potential files:\n')
        
        for f in files:
            if 'nh3_' in f and '.txt' in f:
                print(f)
    
    
    def print_vars(self):
        #potential function used for this calculation
        #load a 1D potential and use cubic spline interpolation
        xl = []
        yl = []
        try:
            with open(self.potfile,"r") as pf:
                for line in pf:
                    if line.startswith("Angle"):
                        continue
                    d = line.split()
                    x = numpy.sin(float(d[0])*numpy.pi/180.)*self.re
                    y = float(d[1])*349.75
                    xl.append(x)
                    yl.append(y)
        except FileNotFoundError:
            print("The potential file \""+self.potfile+"\" was not found. Change "
                  +"the value of the potfile variable to a valid filename to continue.")
            return
        
        angles = numpy.asarray(xl)
        pot = numpy.asarray(yl)
        self.V = spp.CubicSpline(angles,pot,extrapolate=True)
        
        #minimum x value for partice in a box calculation
        self.xmin = xl[0]-0.1
        
        #maximum x value for particle in a box calculation
        #there is no limit, but if xmax<xmin, the values will be swapped
        #if xmax = xmin, then xmax will be set to xmin + 1
        self.xmax = xl[-1]+0.1
        
        self.ngrid = max(self.ngrid,3)
        
        if self.ngrid%2 == 0:
        	self.ngrid+=1
        
        self.nbasis = max(1,self.nbasis)
        
        #calculate some useful values
        self.L = self.xmax - self.xmin
        self.tl = numpy.sqrt(2./self.L)
        self.pixl = numpy.pi/self.L
        
        td = [["re",self.re,"potfile",self.potfile],
              ["ngrid",self.ngrid,"nbasis",self.nbasis],
              ["plotxmin",self.plotxmin,"plotxmax",self.plotxmax],
              ["plotymin",self.plotymin,"plotymax",self.plotymax],
              ["hisotope",self.hisotope]]
        
        print(tabulate.tabulate(td))
        
        
    def pib(self,x,n):
        	return self.tl*numpy.sin(n*x*self.pixl)

#------------------------
#   BEGIN CALCULATION   -
#------------------------
    def calc(self,make_plots=True):
        print("Beginning calculation with these parameters:")
        
        self.print_vars()
        self.evalues=[]
        self.evectors=[]
        self.mass = 3.*self.hisotope*14.003074/(14.003074+3*self.hisotope)
        
        #get current time
        starttime = datetime.datetime.now()
        
        #create grid
        x = numpy.linspace(self.xmin,self.xmax,self.ngrid)
        
        #create Hamiltonian matrix; fill with zeros
        H = numpy.zeros((self.nbasis,self.nbasis))
        
        
        #split Hamiltonian into kinetic energy and potential energy terms
        #H = T + V
        #V is defined above, and will be evaluated numerically
        
        #Compute kinetic energy
        #The Kinetic enery operator is T = -hbar/2m d^2/dx^2
        #in the PIB basis, T|phi(i)> = hbar/2m n^2pi^2/L^2 |phi(i)>
        #Therefore <phi(k)|T|phi(i)> = hbar/2m n^2pi^2/L^2 delta_{ki}
        #Kinetic energy is diagonal
        kepf = spc.hbar*spc.hbar*numpy.pi**2./(2.*self.mass*spc.physical_constants['atomic mass constant'][0]*(self.L*1e-10)*(self.L*1e-10)*spc.h*spc.c*100)
        for i in range(0,self.nbasis):
        	H[i,i] += kepf*(i+1.)*(i+1.)
        
        #now, add in potential energy matrix elements <phi(j)|V|phi(i)>
        #that is, multiply the two functions by the potential at each grid point and integrate
        for i in range(0,self.nbasis):
        	for j in range(0,self.nbasis):
        		if j >= i:
        			y = self.pib(x-self.xmin,i+1.)*self.V(x)*self.pib(x-self.xmin,j+1.)
        			H[i,j] += spi.simps(y,x)
        		else:
        			H[i,j] += H[j,i]
        
        
        #Solve for eigenvalues and eigenvectors
        self.evalues, self.evectors = linalg.eigh(H)
        
        #get ending time
        endtime = datetime.datetime.now()
        
        print("Calculation completed in "+str(endtime-starttime))
        print("Eigenvalues:")
        
        pl = []
        rowlen = 5
        if len(self.evalues)>100:
            rowlen=10
        while rowlen*(len(pl)+1) < len(self.evalues):
            pl.append(self.evalues[rowlen*(len(pl)):rowlen*(len(pl)+1)])
        
        pl.append(self.evalues[rowlen*(len(pl)):])
        
        print(tabulate.tabulate(pl,floatfmt=".3f"))

        x = numpy.linspace(self.xmin,self.xmax,self.ngrid)
        self.vplot = self.V(x)
        angles = numpy.arcsin(x/self.re)*180/numpy.pi
        self.angles = angles
        
        if(self.make_plots):
            self.plot()
        
    def plot(self):
        if len(self.evalues)<1:
            print("Cannot generate plots because there are no calculated eigenvalues.")
            return
        
        plt.close('all')
    
        
        self.evplot = plt.figure(1)
        #Make graph of eigenvalue spectrum
        title = "EV Spectrum, Min={:.4f}, Max={:.4f}, Grid={:d}, Basis={:d}".format(self.xmin,self.xmax,self.ngrid,self.nbasis)
        
        plt.plot(self.evalues,'ro')
        plt.xlabel('v')
        plt.ylabel(r'E (cm$^{-1}$)')
        plt.title(title)
        plt.show()
        
        #Make graph with potential and eigenfunctions
        self.potplot = plt.figure(2)
        title = "Wfns, Min={:.4f}, Max={:.4f}, Grid={:d}, Basis={:d}".format(self.xmin,self.xmax,self.ngrid,self.nbasis)
        x = numpy.linspace(self.xmin,self.xmax,self.ngrid)
        angles = self.angles

        plt.plot(self.angles,self.vplot)
        
        pxmin = self.plotxmin
        pxmax = self.plotxmax
        pymin = self.plotymin
        pymax = self.plotymax
        
        if pxmin == 0:
        	pxmin = angles[0]
        if pxmax == 0:
        	pxmax = angles[-1]
        if(pxmin == pxmax):
        	pxmin, pxmax = angles[0], angles[-1]
        if(pxmin > pxmax):
        	pxmin, pxmax = pxmax,pxmin
        if pymax == 0:
        	pymax = max(self.V(pxmin),self.V(pxmax))
        if(pymin > pymax):
        	pymin, pymax = pymax,pymin
        
        if self.nbasis>2:
            sf = (self.evalues[2]-self.evalues[0])/5.0
        else:
            sf = 1.        
        
        for i in range(0,self.nbasis):
        	plt.plot([angles[0],angles[-1]],[self.evalues[i],self.evalues[i]],'k-')
        for i in range(0,self.nbasis):
        	ef = numpy.zeros(self.ngrid)
        	ef += self.evalues[i]
        	for j in range(0,self.nbasis):
       			ef += self.evectors[j,i]*self.pib(x-self.xmin,j+1)*sf
        	plt.plot(angles,ef)
        
        plt.plot([angles[0],angles[0]],[pymin,pymax],'k-')
        plt.plot([angles[-1],angles[-1]],[pymin,pymax],'k-')
        plt.axis([pxmin,pxmax,pymin,pymax])
        plt.title(title)
        plt.xlabel(r'$\theta$ (degree)')
        plt.ylabel(r'V (cm$^{-1}$)')
        plt.show()


print('PIBSolver initialized. Create a new object with "pb = PIBSolver()"')