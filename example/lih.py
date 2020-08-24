
from mole.molecule import lih
from ctrlq import cvqe
import numpy

# Compute the molecular energy of LiH given a pulse shape to drive
# the system.

cHam = lih(dist=1.5)
eval1, evec1 = numpy.linalg.eigh(cHam)

mypulse = cvqe.pulse(nqubit=4,nwindow=4,duration=10.0)

mypulse.amp = [[-0.047885212285980194, 0.0816915563933863,
                0.04478874748719369, 0.053024415201600616],
               [0.04693238199992897, -0.014343617746861409,
                0.014466456097236458, 0.04588045037289695],
               [0.015378982750353792, 0.029319260450179824,
                0.05770494310656549, 0.05381656189259357],
               [0.04092776682417609, 0.0035745581432664875,
                0.020068455303247412, -0.09377694196404446]]

mypulse.freq = [30.656116165387573, 29.76584572761093,
                30.63380879351657, 30.536383370218147]

mypulse.tseq = [[4.730478800533831, 1.668414130825434,
                 8.899408242154344],
                [3.6063765713877984, 6.989490624995938,
                 3.578495638820728],
                [4.516029730621258, 0.9284106013571314,
                 1.898071619343531],
                [3.3066899135692562, 8.284477712954509,
                 7.89897337509418]]

myham = cvqe.transmon(nqubit=4,mham = cHam)

ctrl = cvqe.control(mypulse,myham,nstep=3000,solver='trotter')
energy, leak = ctrl.energy(normalize=True)

print('Error in ctrl-VQE energy : {:>.4e}'.format(energy-eval1[0]))
