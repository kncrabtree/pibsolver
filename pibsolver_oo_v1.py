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
from scipy import integrate
import numpy
from matplotlib import pyplot as plt
import datetime
import tabulate


class PIBSolver:
    def __init__(self):
        #---------------------------
        #        VARIABLES         -
        #---------------------------
        
        #minimum x value for partice in a box calculation
        self.xmin = -5.0
        
        #maximum x value for particle in a box calculation
        #there is no limit, but if xmax<xmin, the values will be swapped
        #if xmax = xmin, then xmax will be set to xmin + 1
        self.xmax = 5.0
        
        #number of grid points at which to calculate integral
        #must be an odd number. If an even number is given, 1 will be added.
        #minimum number of grid points is 3
        self.ngrid = 101
        
        #number of PIB wavefunctions to include in basis set
        #minimum is 1
        self.nbasis = 20
        
        #if you want to control the plot ranges, you can do so here
#        make_plots = True
        self.plotymin = 0
        self.plotymax = 0
        self.plotxmin = 0
        self.plotxmax = 0
        
        #the units of energy are hbar*omega, so we'll let hbar equal 1 and calculate omega from mass and fk
        #While mass and hbar are used in the Hamiltonian calculation for all potentials, the units of hbar*omega only apply to the harmonic oscillator
        #For other potentials, let fk=1 and hbar=1
        self.mass = 1.0
        self.hbar = 1.0
        self.fk = 1.0
        
        
        #morse potential dissociation energy
        self.de = 10
        
        
        self.V = self.harmonic
        
        self.evalues=[]
        self.evectors=[]
        
        self.print_vars()

    #--------------------------
    #  POTENTIAL DEFINITIONS  -
    #--------------------------
    
    #definitions of potential functions
    #particle in a box: no potential. Note: you need to manually set plotymin and plotymax above for proper graphing
    def box(self,x):
    	return 0
    
    #purely harmonic potential
    def harmonic(self,x):
    	return 0.5*self.fk*(x**2.)
    
    #anharmonic potential
    def anharmonic(self,x):
    	return 0.5*(x**2.) + 0.05*(x**4.)
    
    #Morse potential (Note, try a box of -0.4, 5 to begin)
    def morse(self,x):
    	return self.de*(1.-numpy.exp(-self.alpha*x))**2.
    
    #double well potential (minima at x = +/-3)
    def doublewell(self,x):
    	return x**4.-18.*x**2. + 81.
    
    def print_vars(self):
        #verify that inputs are sane
        if self.xmax==self.xmin:
        	self.xmax = self.xmin+1
        elif self.xmax<self.xmin:
        	self.xmin,self.xmax = self.xmax,self.xmin
        
        self.ngrid = max(self.ngrid,3)
        
        if self.ngrid%2 == 0:
        	self.ngrid+=1
        
        self.nbasis = max(1,self.nbasis)
        
        #calculate some useful values
        self.L = self.xmax - self.xmin
        self.tl = numpy.sqrt(2./self.L)
        self.pixl = numpy.pi/self.L
        self.omega = numpy.sqrt(self.fk/self.mass)
        self.alpha = numpy.sqrt(self.fk/2.0/self.de)
        
        td = [["xmin",self.xmin,"xmax",self.xmax],
              ["ngrid",self.ngrid,"nbasis",self.nbasis],
              ["mass",self.mass,"fk",self.fk],
              ["de",self.de,"V",self.V.__name__],
              ["plotxmin",self.plotxmin,"plotxmax",self.plotxmax],
              ["plotymin",self.plotymin,"plotymax",self.plotymax]]
        
        print(tabulate.tabulate(td,tablefmt='github'))
        
        
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
        for i in range(0,self.nbasis):
        	H[i,i] += self.hbar*(i+1.)**2.*numpy.pi**2./(2.*self.mass*self.L**2.)
        
        #now, add in potential energy matrix elements <phi(j)|V|phi(i)>
        #that is, multiply the two functions by the potential at each grid point and integrate
        for i in range(0,self.nbasis):
        	for j in range(0,self.nbasis):
        		if j >= i:
        			y = self.pib(x-self.xmin,i+1.)*self.V(x)*self.pib(x-self.xmin,j+1.)
        			H[i,j] += integrate.simps(y,x)
        		else:
        			H[i,j] += H[j,i]
        
        
        #Solve for eigenvalues and eigenvectors
        self.evalues, self.evectors = linalg.eigh(H)
        self.evalues = self.evalues*(self.hbar*self.omega)
        
        #get ending time
        endtime = datetime.datetime.now()
        
        print("Calculation completed in "+str(endtime-starttime))
        print("Eigenvalues:")
        
        pl = []
        rowlen = 5
        while rowlen*(len(pl)+1) < len(self.evalues):
            pl.append(self.evalues[rowlen*(len(pl)):rowlen*(len(pl)+1)])
        
        pl.append(self.evalues[rowlen*(len(pl)):])
        
        print(tabulate.tabulate(pl,floatfmt=".4f"))

        if(make_plots):
            self.plot()
        
    def plot(self):
        if len(self.evalues)<1:
            print("Cannot generate plots because there are no calculated eigenvalues.")
            return
        
        plt.close('all')
    
        
        plt.figure(1)
        #Make graph of eigenvalue spectrum
        title = "EV Spectrum, Min={0}, Max={1}, Grid={2}, Basis={3}".format(self.xmin,self.xmax,self.ngrid,self.nbasis)
        
        plt.plot(self.evalues,'ro')
        plt.xlabel('v')
        plt.ylabel(r'E ($\hbar\omega$)')
        plt.title(title)
        plt.show()
        
        #Make graph with potential and eigenfunctions
        plt.figure(2)
        title = "Wfns, Min={0}, Max={1}, Grid={2}, Basis={3}".format(self.xmin,self.xmax,self.ngrid,self.nbasis)
        x = numpy.linspace(self.xmin,self.xmax,self.ngrid)
        vplot = numpy.zeros(x.size)
        for i in range(0,x.size):
            vplot[i] = self.V(x[i])
        plt.plot(x,vplot)
        
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
        	pxmin, pxmax = pxmax,pxmin
        if pymax == 0:
        	pymax = max(self.V(pxmin),self.V(pxmax))
        if(pymin > pymax):
        	pymin, pymax = pymax,pymin
        
        if self.nbasis>2:
            sf = (self.evalues[2]-self.evalues[0])/2.0
        else:
            sf = 1.        
        
        for i in range(0,self.nbasis):
        	plt.plot([self.xmin,self.xmax],[self.evalues[i],self.evalues[i]],'k-')
        for i in range(0,self.nbasis):
        	ef = numpy.zeros(self.ngrid)
        	ef += self.evalues[i]
        	for j in range(0,self.nbasis):
       			ef += self.evectors[j,i]*self.pib(x-self.xmin,j+1)*sf
        	plt.plot(x,ef)
        
        plt.plot([self.xmin,self.xmin],[pymin,pymax],'k-')
        plt.plot([self.xmax,self.xmax],[pymin,pymax],'k-')
        plt.axis([pxmin,pxmax,pymin,pymax])
        plt.title(title)
        plt.xlabel('$x-x_0$ $(\sqrt{\hbar/m\omega})$')
        plt.ylabel(r'V ($\hbar\omega$)')
        plt.show()


