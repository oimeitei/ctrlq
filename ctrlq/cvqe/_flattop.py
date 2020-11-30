#   Copyright 2020 Oinam Romesh Meitei
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from math import exp
import numpy, sys, functools

def flattop(x, a, n, m):
    return exp(-m * (x/a)**n)

def gaussquare(ini_state, pobj, hobj, solver, nstep, normalize,
               twindow=False, leak=False):
    from ctrlq.lib.solve import pulsec
    from ctrlq.cvqe.evolve import evolve
    import matplotlib.pyplot as plt

    # Gaussian square (flattop) pulse, expt.
    # convert pobj to amp at every time step
    
    mean = pobj.duration/2.0
    amp = []
    for i in range(pobj.nqubit):
        tmp_a = []
        tlist = numpy.linspace(0., pobj.duration, nstep)
        for j in tlist:
            tmp_a.append(pobj.amp[i][0] * flattop(j-mean, mean, 8.0, 2.0))
        
        tmp_b = 1000.* (1/(numpy.pi * 2.0)) * numpy.array(tmp_a)
        plt.plot(tlist, tmp_b)
        plt.show()
        amp.append(tmp_a)
            
    stseq = numpy.array(pobj.tseq) / pobj.tscale
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp, stseq, sfreq, pobj.duration,
                   pobj.nqubit, pobj.nwindow)

    out = evolve(ini_state, pobjc, hobj, solver=solver, nstep=nstep,
                 twindow=twindow)

    state_ = []
    for i in hobj.states:
        state_.append(out[i])
    state_ = numpy.array(state_)

    nrm = numpy.linalg.norm(state_)
    leak_ = 1. - nrm
    
    if normalize:
        nrm = numpy.linalg.norm(state_)
        staten = state_ /nrm
    else:
        staten = state_

    energy = functools.reduce(numpy.dot,
                              (staten.conj().T, hobj.mham, staten))
    
    energy = energy.real[0][0]
    if leak:
        return (energy, leak_)
    
    return energy
        

def gaussquare_obj(self, list1, pobj, hobj, solver, nstep,
                   normalize, misc=False, grad= True, supdate=True):

    from scipy.optimize import approx_fprime
    
    cout = 0
    for i in range(pobj.nqubit):
        pobj.amp[i][0] = list1[cout]
        cout += 1

    # No tseq here

    for i in range(pobj.nqubit):
        pobj.freq[i] = list1[cout]
        cout += 1

    ini_state = hobj.initial_state

    E_, L_ = gaussquare(ini_state, pobj, hobj, solver, nstep,
                         normalize, leak= True)

    if supdate:
        self.energy_ = E_
        self.leak = L_

    if not grad:
        return E_
    
    G_ = approx_fprime(list1, gaussquare_gfunc, 1e-8, ini_state,
                       pobj, hobj, solver, nstep, normalize)
    
    return (E_, G_)

def gaussquare_gfunc(list1, ini_state, pobj, hobj, solver,
                     nstep, normalize):

    cout = 0
    for i in range(pobj.nqubit):
        pobj.amp[i][0] = list1[cout]
        cout += 1

    # No tseq here

    for i in range(pobj.nqubit):
        pobj.freq[i] = list1[cout]
        cout += 1

    E_ = gaussquare(ini_state, pobj, hobj, solver, nstep,
                    normalize)
    return E_
