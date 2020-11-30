

import unittest
import numpy, scipy
from mole.molecule import h2
from ctrlq import cvqe


class expecth2(unittest.TestCase):

    def test_sq2_normalize_trotter(self):

        cHam = h2()
        
        mypulse = cvqe.pulse(shape='square', nqubit=2, nwindow=2,
                             duration=10.0)
        mypulse.amp=[[-0.05202901, -0.11437925],
                     [-0.00909615 ,   0.02574960]]
        mypulse.tseq = [[1.754454], [6.094212]]
        mypulse.freq=[29.407306, 30.992068]

        myham = cvqe.transmon(mham = cHam)

        ctrl = cvqe.control( mypulse, myham, nstep=5000,
                             solver='trotter', iprint=0)

        e1, l1 = ctrl.energy(normalize = True)

        self.assertAlmostEqual(e1, -1.121832748743835, 8)
        self.assertAlmostEqual(l1*100.,  0.638738650065718, 8)


    def test_sq3_normalize_trotter(self):

        cHam = h2()
        
        mypulse = cvqe.pulse(shape='square', nqubit=2, nwindow=3,
                             duration=10.0)
        mypulse.amp= [[0.1249952852252908, 0.09417523192820781,
                       -0.00764585921332081],
                      [0.06126291341779569, 0.03336212116885992,
                       0.03697398031839694]]
        mypulse.tseq = [[6.159809773315513, 2.3377316846035265],
                        [9.997895477848182, 9.352478104119168]]
        mypulse.freq=  [30.582426020499156, 30.928482921964044]

        myham = cvqe.transmon(mham = cHam)

        ctrl = cvqe.control( mypulse, myham, nstep=5000,
                             solver='trotter', iprint=0)

        e1, l1 = ctrl.energy(normalize = True)

        self.assertAlmostEqual(e1, -1.041003304941890, 8)
        self.assertAlmostEqual(l1*100., 0.067308598896398, 8)

    def test_sq4_normalize_trotter(self):

        cHam = h2()
        
        mypulse = cvqe.pulse(shape='square', nqubit=2, nwindow=4,
                             duration=10.0)
        mypulse.amp= [[-0.023397814557632512, -0.024404811767025192,
                       0.017514431203435765, -0.007597731625624313],
                      [0.08612679102439388, 0.0903360596717867,
                       -0.04077765915285132, 0.08333376553136651]]
        mypulse.tseq = [[6.51101596013727, 2.231101283609367,
                         1.6067317587499053],
                        [8.984285889444015, 9.365389790987054,
                         4.79755729956294]]
        mypulse.freq= [30.489396709842353, 31.34287045678159]
        
        myham = cvqe.transmon(mham = cHam)

        ctrl = cvqe.control( mypulse, myham, nstep=5000,
                             solver='trotter', iprint=0)

        e1, l1 = ctrl.energy(normalize = True)

        self.assertAlmostEqual(e1, -1.100170840004205, 8)
        self.assertAlmostEqual(l1*100., 0.232940585238139, 8)    
        
        
if __name__ == '__main__':
    unittest.main()
