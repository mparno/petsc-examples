import sys, petsc4py
petsc4py.init(sys.argv)

import numpy as np
import matplotlib.pyplot as plt 

from petsc4py import PETSc
from mpi4py import MPI
import math

def RHS_func(ts, t, U, F, *args):
    F[0] = U[1]
    F[1] = -10.0*U[0]


n = 2
u = PETSc.Vec().createSeq(n)
f = u.duplicate()

ts = PETSc.TS().create()

# List of solvers: https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.TS.Type-class.html
ts.setType(ts.Type.RK)

# Tell PETSc what function we want to integrate
ts.setRHSFunction(RHS_func, f)

ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)
# Set the initial value to 1.0
u.setValues([0,1],[-1.0,0.0])

# Set initial stepsize
ts.setTimeStep(0.01) 
#ts.setMaxSteps(10000)
#ts.setTolerances(1e-12)

ts.setFromOptions()

# Set the fineal time
num_intervals = 100
obs_times = np.linspace(0,5,num_intervals+1)
results = np.zeros((num_intervals+1,2))
results[0,:] = u.getArray()

for i,next_time in enumerate(obs_times[1:]):

    # Run the solver to compute the solution at the end of the interval
    ts.setMaxTime(next_time)
    ts.solve(u)

    # Save the current value
    results[i+1,:] = u.getArray()
    obs_times[i+1] = ts.getSolveTime()
    
plt.plot(obs_times, results)
plt.xlabel('Time')
plt.ylabel('State Values')
plt.show()