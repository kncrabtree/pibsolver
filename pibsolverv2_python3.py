#!/usr/bin/python

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
import math
import numpy
import matplotlib
matplotlib.use('PDF',warn=False)
from matplotlib import pyplot as plt
import datetime


#---------------------------
#        VARIABLES         -
#---------------------------

#minimum x value for partice in a box calculation
xmin = -1.0

#maximum x value for particle in a box calculation
#there is no limit, but if xmax<xmin, the values will be swapped
#if xmax = xmin, then xmax will be set to xmin + 1
xmax = 15.0

#number of grid points at which to calculate integral
#must be an odd number. If an even number is given, 1 will be added.
#minimum number of grid points is 3
ngrid = 501

#number of PIB wavefunctions to include in basis set
#minimum is 1
nbasis = 200

#if you want to control the plot ranges, you can do so here
make_plots = True
plotymin = 0
plotymax = 0
plotxmin = 0
plotxmax = 0

#the units of energy are hbar*omega, so we'll let hbar equal 1 and calculate omega from mass and fk
#While mass and hbar are used in the Hamiltonian calculation for all potentials, the units of hbar*omega only apply to the harmonic oscillator
#For other potentials, let fk=1 and hbar=1
mass = 1.13864e-26
nubar = 2169.756
hbar = 1.0546e-34
fk = 1901.995
omega = math.sqrt(fk/mass)

#morse potential parameters
de = 90546.0
alpha = math.sqrt(fk/2/de)

#output file for eigenvalues
outfile = "eigenvalues.txt"

#output PDF for energy spectrum
spectrumfile = "energy.pdf"

#output PDF for potential and eigenfunctions
potentialfile = "potential.pdf"


#--------------------------
#  POTENTIAL DEFINITIONS  -
#--------------------------

#definitions of potential functions
#particle in a box: no potential. Note: you need to manually set plotymin and plotymax above for proper graphing
def box(x):
	return 0

#purely harmonic potential
def harmonic(x):
	return 0.5*fk*(x**2.)

#anharmonic potential
def anharmonic(x):
	return 0.5*(x**2.) + 0.05*(x**4.)

#Morse potential (Note, try a box of -0.4, 5 to begin)
def morse(x):
	return de*(1.-math.exp(-alpha*x))**2.

#double well potential (minima at x = +/-3)
def doublewell(x):
	return x**4.-18.*x**2. + 81.

#potential function used for this calculation
def V(x):
	return morse(x)

#------------------------
#   BEGIN CALCULATION   -
#------------------------

#function to compute normalized PIB wavefunction
def pib(x,n,L):
	return math.sqrt(2./L)*math.sin(n*math.pi*x/L)

#verify that inputs are sane
if xmax==xmin:
	xmax = xmin+1
elif xmax<xmin:
	xmin,xmax = xmax,xmin

L = xmax - xmin

ngrid = max(ngrid,3)

if not ngrid%2:
	ngrid+=1

nbasis = max(1,nbasis)

#get current time
starttime = datetime.datetime.now()

#create grid
x = numpy.linspace(xmin,xmax,ngrid)

#create Hamiltonian matrix; fill with zeros
H = numpy.zeros((nbasis,nbasis))


#split Hamiltonian into kinetic energy and potential energy terms
#H = T + V
#V is defined above, and will be evaluated numerically

#Compute kinetic energy
#The Kinetic enery operator is T = -hbar/2m d^2/dx^2
#in the PIB basis, T|phi(i)> = hbar/2m n^2pi^2/L^2 |phi(i)>
#Therefore <phi(k)|T|phi(i)> = hbar/2m n^2pi^2/L^2 delta_{ki}
#Kinetic energy is diagonal
for i in range(0,nbasis):
	H[i,i] += hbar*(i+1.)**2.*math.pi**2./(2.*mass*L**2.)

#now, add in potential energy matrix elements <phi(j)|V|phi(i)>
#that is, multiply the two functions by the potential at each grid point and integrate
for i in range(0,nbasis):
	for j in range(0,nbasis):
		if j >= i:
			y = numpy.zeros(ngrid)
			for k in range(0,ngrid):
				p = x[k]
				y[k] += pib(p-xmin,i+1.,L)*V(p)*pib(p-xmin,j+1.,L)
			H[i,j] += integrate.simps(y,x)
		else:
			H[i,j] += H[j,i]


#Solve for eigenvalues and eigenvectors
evalues, evectors = linalg.eigh(H)
evalues = evalues*nubar

#get ending time
endtime = datetime.datetime.now()

#------------------------
#    GENERATE OUTPUT    -
#------------------------

print(evalues)

with open(outfile,'w') as f:
	f.write(str('#pibsolver.py output\n'))
	f.write('#start time ' + starttime.isoformat(' ') + '\n')
	f.write('#end time ' + endtime.isoformat(' ') + '\n')
	f.write(str('#elapsed time ' + str(endtime-starttime) + '\n'))
	f.write(str('#xmin {0}\n').format(xmin))
	f.write(str('#xmax {0}\n').format(xmax))
	f.write(str('#grid size {0}\n').format(ngrid))
	f.write(str('#basis size {0}\n\n').format(nbasis))
	f.write(str('#eigenvalues\n'))
	f.write(str('{0:.5e}').format(evalues[0]))
	for i in range(1,nbasis):
		f.write(str('\t{0:.5e}').format(evalues[i]))
	f.write('\n\n#eigenvectors\n')
	for j in range(0,nbasis):
		f.write(str('{0:.5e}').format(evectors[j,0]))
		for i in range(1,nbasis):
			f.write(str('\t{0:.5e}').format(evectors[j,i]))
		f.write('\n')

if make_plots == True:
    #if this is run in interactive mode, make sure plot from previous run is closed
    plt.figure(1)
    plt.close()
    plt.figure(2)
    plt.close()

    
    plt.figure(1)
    #Make graph of eigenvalue spectrum
    title = "EV Spectrum, Min={0}, Max={1}, Grid={2}, Basis={3}".format(xmin,xmax,ngrid,nbasis)
    
    plt.plot(evalues,'ro')
    plt.xlabel('v')
    plt.ylabel(r'E (cm$^{-1}$)')
    plt.title(title)
    plt.savefig(spectrumfile)
    plt.show()
    
    #Make graph with potential and eigenfunctions
    plt.figure(2)
    title = "Wfns, Min={0}, Max={1}, Grid={2}, Basis={3}".format(xmin,xmax,ngrid,nbasis)
    vplot = numpy.zeros(x.size)
    for i in range(0,x.size):
        vplot[i] = V(x[i])*nubar
    plt.plot(x,vplot)
    
    if plotxmin == 0:
    	plotxmin = xmin
    if plotxmax == 0:
    	plotxmax = xmax
    if(plotxmin == plotxmax):
    	plotxmin, plotmax = xmin, xmax
    if(plotxmin > plotxmax):
    	plotxmin, plotxmax = plotxmax,plotxmin
    if plotymax == 0:
    #	plotymax = max(V(plotxmin),V(plotxmax))
          plotymax = evalues[10]
    if(plotymin > plotymax):
    	plotymin, plotymax = plotymax,plotymin
    
    sf = evalues[1]-evalues[0]
    for i in range(0,nbasis):
    	plt.plot([xmin,xmax],[evalues[i],evalues[i]],'k-')
    for i in range(0,nbasis):
    	ef = numpy.zeros(ngrid)
    	ef += evalues[i]
    	for j in range(0,nbasis):
    		for k in range(0,ngrid):
    			ef[k] += evectors[j,i]*pib(x[k]-xmin,j+1,L)*sf
    	plt.plot(x,ef)
    
    plt.plot([xmin,xmin],[plotymin,plotymax],'k-')
    plt.plot([xmax,xmax],[plotymin,plotymax],'k-')
    plt.axis([plotxmin,plotxmax,plotymin,plotymax])
    plt.title(title)
    plt.xlabel('$x-x_0$ $(\sqrt{\hbar/m\omega})$')
    plt.ylabel(r'V (cm$^{-1}$)')
    plt.savefig(potentialfile)
    plt.show()




