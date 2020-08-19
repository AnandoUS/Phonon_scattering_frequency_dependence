import numpy as np
import math
import matplotlib.pyplot as plt

######################        Constants       ###################
pi=np.pi
a=9.108*10**-10						#Lattice parameter
T=300							#Temperature
k=1.38 * 10**-23					#boltzman constant
M=98.07/(6.022*10**23)					#average atomic mass
C=((3**0.5)*pi*a*T*k)/(2*M)				#Constant coefficient to the frequency term of the scattering rate expression by Klemens.

################## Reading ShengBTE output files #################
qpts=np.loadtxt('BTE.qpoints')				#Contains q-points within the IBZ wedge of the system. And in the third column, it contains degeneracy of each q-point within the full BZ.
freq=np.loadtxt('BTE.omega')				#Frequency of all the points in the IBZ wedge given in the BTE.qpoints file 
rlv=np.loadtxt('BTE.ReciprocalLatticeVectors')		#File containing reciprocal lattice vectors
v=np.loadtxt('BTE.v')					#Speed of sound at each mode at all the q-points within the IBZ in km/s are given in the file BTE.v
t=np.loadtxt('BTE.w_anharmonic')			#File containing scattering rates


n=int(qpts[-1,0])					#number of points within the IBZ

n_T = 0							#Initializing Total number of modes 
for i in range(n):
        n_T = n_T + qpts[i,2]				#Calculating Total number of modes 


n_f=qpts[:,2]/n_T					#The fraction of the total number of q-points contributed by each symmetrically equivalent q-point in the BZ is given in the array n_f

freq=freq/(2*pi)					#Conversion of frequencies from rad/ps to THz
freq=freq*10**12					#Conversion of frequencies from THz to Hz
a=32							#Number of atoms in the primitive unit cell
m=3*a							#Number of modes

d=np.zeros(shape=(n))

for i in range(n):
	rlv1=qpts[i,3]*rlv[0,:]
	rlv2=qpts[i,4]*rlv[1,:]
	rlv3=qpts[i,5]*rlv[2,:]
	x1=rlv1[0]+rlv2[0]+rlv3[0]
	y1=rlv1[1]+rlv2[1]+rlv3[1]
	z1=rlv1[2]+rlv2[2]+rlv3[2]
	d[i]=x1**2+y1**2+z1**2

d=d**0.5
d=d*10**9

v_phase=np.zeros(shape=(n,m))				#Initializing phase velocity array
for i in range(n):
	v_phase[i,:]=freq[i,:]/d[i]			#Calculating phase velocity array

v_phase=np.delete(v_phase, 0, 0)

v=v*1000						#Converting to m/s
v2=v**2							#Squaring v_x, v_y, v_z at each mode in the IBZ

V2=v2[:,0]+v2[:,1]+v2[:,2]				
v2_mode=V2.reshape((m, n)).T				#Reshaping the 1D V2 array to the dimensions of cv_mode

v2_mode=np.delete(v2_mode, 0, 0)
v_mode=v2_mode**0.5					#group velocity for each mode in the IBZ

t=t[:,1]
t=t*10**12
t_mode=t.reshape((m, n)).T
t_mode=np.delete(t_mode, 0, 0)				#Scattering rate various modes

freq1=np.delete(freq, 0, 0)

quantity=(t_mode*v_mode*v_phase**2)/(freq1**2)
grun=quantity/C
grun=grun**0.5						#Back-calculated gruneisen parameter

print(np.std(grun), np.mean(grun))

plt.scatter(4.135665538536*freq1*10**-12, grun,c='r',linewidth=3)
plt.xlabel('Freqency (meV)', fontsize=24)
plt.ylabel('Gruneisen', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
