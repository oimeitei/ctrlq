from ctrlq import cvqe
from mole.molecule import h2
import numpy

# Adaptive pulse optimization starting from a single pulse for H_2 molecule

cHam = h2(dist=0.75)
eval1, evec1 = numpy.linalg.eigh(cHam)

mypulse = cvqe.pulse(shape='square',duration=10.,nwindow=1, nqubit = 2,
                     amp_bound=0.04,freq_bound=1.5)

myham = cvqe.transmon(mham = cHam,nqubit=2)
ctrl = cvqe.control(mypulse,myham,nstep=N,solver='trotter')

ctrl.adapt(exactE = eval1[0])
