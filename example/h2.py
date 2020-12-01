from ctrlq import cvqe
from mole.molecule import h2
import numpy

# Pulse optimization with a given initial pulse shape for H_2 molecule

cHam = h2(dist=0.75)
eval1, evec1 = numpy.linalg.eigh(cHam)

mypulse = cvqe.pulse(duration=10.0)

mypulse.amp=[[-0.05202901 ,  -0.11437925],
             [-0.00909615 ,   0.02574960]]

mypulse.tseq = [[1.754454],[6.094212]]

mypulse.freq=[29.407306   , 30.992068]

myham = cvqe.transmon(mham = cHam)

ctrl = cvqe.control(mypulse,myham,nstep=500,solver='trotter')
energy, leak = ctrl.optimize(normalize=True)

print('Error in ctrl-VQE energy : {:>.4e}'.format(energy-eval1[0]))
