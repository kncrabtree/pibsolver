import psi4
import numpy as np
from matplotlib import pyplot as plt
from contextlib import redirect_stdout
import os

psi4.set_memory('1024 MB')

def calc_coordinates(angle_degrees, nhlength):
    N = [0.0,0.0,0.0]
    
    angle = angle_degrees/180.0*np.pi
    cos_example = np.cos(angle)
    
    h1x = np.cos(angle)*nhlength
    h1y = 0.0
    h1z = np.sin(angle)*nhlength
    
    h2x = np.cos(2.0*np.pi/3.0)*np.cos(angle)*nhlength
    h2y = np.sin(2.0*np.pi/3.0)*np.cos(angle)*nhlength
    h2z = h1z
    
    h3x = h2x
    h3y = -h2y
    h3z = h1z
    
    return f"""
N 0.0 0.0 0.0
H {h1x} {h1y} {h1z}
H {h2x} {h2y} {h2z}
H {h3x} {h3y} {h3z}
"""

def calc_all(methods,basis_sets,optimize_path=True):
    #ngle_array = np.zeros(2*len(angles)-1)
    
    numangles = 41
    maxangle = 40.
    angles = np.linspace(-np.abs(maxangle),0,numangles)
    
    total = len(methods)*len(basis_sets)
    count = 0
    
    res_angles, res_egys, res_bonds, res_barriers, res_calcs = [], [], [], [], []
    
    pecs = []
    out = []
    
    for method in methods:
        for basis in basis_sets:
            count = count+1
            calc = method+"/"+basis
            print('\n-------------------------------------')
            print('Starting '+calc+' calculation ('+str(count)+'/'+str(total)+')')
            print('-------------------------------------')
            print('Performing initial geometry optimization.')
            nh, angle, egy = optimize(method,basis)
            print('Calculating potential energy curve.')
            barrier, x, y, potfile = calc_single(method,basis,nh,angles,egy)
            
            res_calcs.append(calc)
            res_angles.append(angle)
            res_egys.append(egy)
            res_bonds.append(nh)
            res_barriers.append(barrier)
            pecs.append(y)
            out.append((nh,barrier,potfile))
            
    print("\n\nSummary\n")
    print("--------------------------------------------------------------------------")
    print("Calculation       |Energy (au)|N-H Bond (A)|Angle (deg)|Barrier (kcal/mol)")
    print("--------------------------------------------------------------------------")
    for c, e, b, a, ba in zip(res_calcs, res_egys, res_bonds, res_angles, res_barriers):
        print("{:18s} {:<11.4f} {:<12.4f} {:<11.3f} {:<18.4f}".format(c,e/psi4.constants.hartree2kcalmol,b,a,ba))
        
    fig,axes = plt.subplots(1,len(methods),figsize=(6*len(methods),4),dpi=150)
    i=0
    l = len(basis_sets)
    for ax in axes:
        ax.plot(x,np.asarray(pecs[i*l:i*l+l]).T,label=np.asarray(res_calcs[i*l:i*l+l]))
        ax.set_xlabel("Angle (degree)")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.legend()
        i+=1
    
    return out

def calc_single(method,basis,r,angles,egyend=0.0,optimize_path=True):
    calc = method+"/"+basis
    angle_array = np.zeros(2*len(angles)-1)
    egy_array = np.zeros(2*len(angles)-1)
    outfile = method+'_'+basis+'_output.dat'
    outfile = outfile.replace('*','s')
    outfile = outfile.replace('+','p')
    psi4.core.set_output_file(outfile,False)

    idx = 0
    for angle in angles:
        geo = calc_coordinates(angle,r)
        nh3 = psi4.geometry(geo)
        egy = 0.
        if optimize_path:
            psi4.set_options({'optking__FROZEN_BEND':'2 1 3 3 1 4 4 1 2'})
            with open(os.devnull,'w') as f:
                with redirect_stdout(f):
                    egy = psi4.optimize(calc,molecule=nh3)*psi4.constants.hartree2kcalmol - egyend
        else:
            egy = psi4.energy(calc,molecule=nh3)*psi4.constants.hartree2kcalmol - egyend
        printProgressBar(idx+1,len(angles))
        angle_array[idx] = angle
        angle_array[-(idx+1)] = -angle
        egy_array[idx] = egy
        egy_array[-(idx+1)] = egy
        idx+=1

    barrier = egy_array[idx-1]
    print('Barrier height: {:6.4f} kcal/mol'.format(barrier))

    #plt.plot(angle_array,egy_array,label=calc.upper())

    filename = 'nh3_'+method+'_'+basis
    if optimize_path:
        filename += '_opt'
    filename+='.txt'
    filename = filename.replace('*','s')
    print('Detailed potential calculation information written to '+outfile)
    print('Potential energy data written to '+filename)
    print('-------------------------------------')
    with open(filename,'w') as f:
        f.write('Angle (deg)\tEnergy (kcal/mol)')
        for angle,egy in zip(angle_array,egy_array):
            f.write('\n'+str(angle)+'\t'+str(egy))
            
    return barrier, angle_array, egy_array, filename
    
    
def optimize(method,basis):
    calc = method+"/"+basis
    psi4.set_options({'optking__FROZEN_BEND':''})
    outfile = method+'_'+basis+'_opt_output.dat'
    outfile = outfile.replace('*','s')
    outfile = outfile.replace('+','p')
    psi4.core.set_output_file(outfile,False)
    
    nh3_opt = psi4.geometry("""
        X
        N 1 1.0
        H 2 1.0 1 110.0
        H 2 1.0 1 110.0 3 120.0
        H 2 1.0 1 110.0 3 -120.0                  
        """)
    egyend = psi4.optimize(calc,molecule=nh3_opt)
    dm = nh3_opt.distance_matrix().to_array()
    nhbond = dm[0,1]*psi4.constants.bohr2angstroms

    #find optimized umbrella angle.
    #Get geometry matrix, and translate N atom to origin        
    gm = nh3_opt.geometry().to_array().transpose()        
    for i in range(0,3):
        d = gm[i,0]
        for j in range(0,4):
            gm[i,j] -= d

    #rotate H1 into the xz plane by rotating about z axis
    theta = 0.0
    if abs(gm[0,1]) > 1e-10:
        theta = np.arctan(-gm[1,1]/gm[0,1])
    else:
        theta=np.pi/2.0

    Rtheta = np.zeros((3,3))
    Rtheta[0,0] = np.cos(theta)
    Rtheta[0,1] = -np.sin(theta)
    Rtheta[1,0] = np.sin(theta)
    Rtheta[1,1] = np.cos(theta)
    Rtheta[2,2] = 1.0
    gmrot = np.dot(Rtheta,gm)

    #rotate H1 onto z axis by rotating about y axis
    theta2 = 0.0
    if abs(gmrot[0,1]) > 1e-10:
        theta2 = np.arctan(gmrot[2,1]/gmrot[0,1])
    else:
        theta2 = np.pi/2.0

    Rtheta2 = np.zeros((3,3))
    Rtheta2[0,0] = np.cos(theta2)
    Rtheta2[0,2] = np.sin(theta2)
    Rtheta2[1,1] = 1.0
    Rtheta2[2,0] = -np.sin(theta2)
    Rtheta2[2,2] = np.cos(theta2)
    gmrot2 = np.dot(Rtheta2,gmrot)

    #rotate about x axis until z(H2) = z(H3)
    theta3 = 0.0
    if abs(gmrot2[2,3]-gmrot2[2,2]) > 1e-10:
        if abs(gmrot2[1,2]-gmrot2[1,3]) > 1e-10:
            theta3 = np.arctan((gmrot2[2,3]-gmrot2[2,2])/(gmrot2[1,2]-gmrot2[1,3]))
        else:
            theta3 = np.pi/2.0

    Rtheta3 = np.zeros((3,3))
    Rtheta3[0,0] = 1.0
    Rtheta3[1,1] = np.cos(theta3)
    Rtheta3[1,2] = -np.sin(theta3)
    Rtheta3[2,1] = np.sin(theta3)
    Rtheta3[2,2] = np.cos(theta3)

    gmrot3 = np.dot(Rtheta3,gmrot2)

    #umbrella angle is the rotation angle about the y axis required for
    #Z(H1) = Z(H2) = Z(H3)
    umbangle = (np.arctan((gmrot3[2,2]-gmrot3[2,1])/(gmrot3[0,2]-gmrot3[0,1])))/np.pi*180.0
    egyend *= psi4.constants.hartree2kcalmol
    
    print('Detailed geometry optimization information written to '+outfile)
    
    return nhbond,umbangle,egyend


# Print iterations progress
def printProgressBar (count, total, decimals = 0, length = 50, fill = '*'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (count / float(total)))
    filledLength = int(length * count // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r|{bar}| {percent}%', end = '\r')
    # Print New Line on Complete
    if count == total: 
        print()