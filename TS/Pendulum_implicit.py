"""
This example solves for the motion of a pendulum
http://www.scholarpedia.org/article/Differential-algebraic_equations

The constraint x*x + y*y = 1 is differentiated with respect to time to obtain the 
constraint x*u + y*v = 0, where u and v are the horizontal and vertical velocities.
This decreases the index of the DAE and enables solution.

"""

import sys, petsc4py
petsc4py.init(sys.argv)

import numpy as np
import matplotlib.pyplot as plt 

from petsc4py import PETSc
from mpi4py import MPI


def F_semi(ts, t, u, udot, result, *args):
    #print('In F..', flush=True)
    g = 9.81

    result[0] = udot[0]
    result[1] = udot[1]
    result[2] = udot[2] + u[4]*u[0] 
    result[3] = udot[3] + u[4]*u[1]
    result[4] = u[0]*u[2] + u[1]*u[3]

    #print('error: ', u[0]*u[0] + u[1]*u[1] - 1.0)
    #print("  F res = ", result.getArray())
    return 0

def G(ts, t, u, result, *args):
    #print('In G..', flush=True)
    g = 9.81
    result[0] = u[2]
    result[1] = u[3]
    result[2] = 0.0
    result[3] = -g
    result[4] = 0.0
    return 0

def FJac_semi(ts,t,u,udot, shift, jac, precond, *args):
    # see https://petsc.org/release/docs/manual/ts/ for a description of the input arguments
    # sigma * dF/dudot + dF/du
    #print('In FJac..', flush=True)
   
    #jac = PETSc.Mat().createAIJ([5, 5])
    jac.setUp()
    rows = [0,1,2,2,2,3,3,3,4,4,4,4,4]
    cols = [0,1,2,0,4,3,1,4,0,1,2,3,4]
    vals = [shift,shift, shift, u[4], u[0], shift, u[4], u[1], u[2], u[3], u[0], u[1], 0.0]
    #rows = [0,1,2,3,4]
    #cols = [0,1,2,3,4]
    #vals = [shift]*len(rows)

    for i in range(len(rows)):
        jac.setValue(rows[i],cols[i], vals[i])
    jac.assemble()
    precond = jac.copy()
    
    ai, aj, av = jac.getValuesCSR()
    #print('  fjac = ', sp.csr_matrix((av, aj, ai)).todense())
    
    return 0 

# Number of state variables (pos=2,vel=2,lagrange=1) = 5
n = 5
u = PETSc.Vec().createSeq(n)
f = u.duplicate()
g = u.duplicate()

# Create a matrix to store the jacobian
df = PETSc.Mat().createAIJ([n,n])

# Create the PETSCtime integration context
ts = PETSc.TS().create()
ts.setFromOptions()

# Optionally, set the linear sovler and preconditioner options (here we set LU direct solve)
ts.getSNES().getKSP().setType(PETSc.KSP.Type.PREONLY) 
ts.getSNES().getKSP().getPC().setType(PETSc.PC.Type.LU)

# List of solvers: https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.TS.Type-class.html
ts.setType(ts.Type.ARKIMEX)
#ts.setARKIMEXType(ts.ARKIMEXType.ARKIMEX4) # see https://fossies.org/linux/petsc/src/binding/petsc4py/src/PETSc/TS.pyx

# Tell PETSc what function we want to integrate
ts.setIFunction(F_semi,f)
ts.setIJacobian(FJac_semi,df, df)
ts.setRHSFunction(G,g)

ts.setExactFinalTime(ts.ExactFinalTime.STEPOVER)

# Set the initial value to 1.0
u.setValues([0],[1])

# Set initial stepsize
ts.setTimeStep(0.01) 

# Specify how often we want to evaluate solution
num_intervals = 100
obs_times = np.linspace(0,5,num_intervals+1)
results = np.zeros((num_intervals+1,5))
results[0,:] = u.getArray()

for i,next_time in enumerate(obs_times[1:]):

    # Run the solver to compute the solution at the end of the interval
    ts.setMaxTime(next_time)
    ts.solve(u)

    # Save the current value
    results[i+1,:] = u.getArray()
    obs_times[i+1] = ts.getSolveTime()
    

plt.plot(obs_times, results[:,0], '.-', label='$x$-position')
plt.plot(obs_times, results[:,1], '.-', label='$y$-position')
plt.legend()
plt.title('Pendulum DAE Solution')
plt.xlabel('Time')
plt.ylabel('State Values')
plt.show()