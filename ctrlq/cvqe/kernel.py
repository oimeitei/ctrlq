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

import numpy, functools, scipy, sys
from ctrlq.cvqe.pulse import pulse
from ctrlq.cvqe.evolve import evolve
from math import exp

class control:

    """ Control class: kernel for ctrl-VQE

    Parameters
    ----------
    pulse : cvqe.pulse
        Pulse
    ham : cvqe.ham
        Transmon Hamiltonian
    solver : str
        Solver for time evolution. Supported, 'ode', 'trotter'. Defaults to 'ode'.
    nstep : int
        Time step. Defaults to 500. 

    """
    
    def __init__(self, pulse_, ham, solver='ode', nstep=500, iprint=5):
        
        self.pulse = pulse_
        self.ham = ham
        self.solver = solver
        self.nstep = nstep
        self.energy_ = 0.0
        self.leak = 0.0
        self.itera_ = 0
        self.max_grad = 0.0
        self.rms_grad = 0.0
        self.prevE = 0.0
        self.square_amp = [[]]
        self.square_gamp = [[]]
        self.print_normalized = False
        self.elist_ = []
        self.llist_ = []
        self.glist_ = []
        self.gradient_ = []
        self.iprint = iprint
        if self.iprint > 1:
            print(flush=True)
            print('              **------------------**',flush=True)
            print('              *--------------------*',flush=True)
            print('              |        CtrlQ       |',flush=True)
            print('              *--------------------*',flush=True)
            print('              **------------------**',flush=True)
            print(flush=True)
            print('        Variational pulse shaping: ctrl-VQE',flush=True)
            print(flush=True)
        

    from ._grad import gradfunc, numgradfreq_2_,numgradamp_3_,\
        numgradamp_5_,numgradw_2_
    from ._anasquare import anatwin
    from ._anagaussian import anagaus,numgradamp_gaus,numgradmean_gaus,\
        numgradsigma_gaus
    from ._flattop import gaussquare_obj
    from ._adapt import adapt
    from ._opt import optimize
    from ._energy import energy, HF
    from ._optparam import objfunc_param, print_param, opt_param, gfunc_param
    from ._optpulsefreq import opt_wpulse, objfunc_wpulse
    
    def print_step(self, xk):

        self.elist_.append(self.energy_)
        self.llist_.append(self.leak)
        self.glist_.append(self.gradient_)
        
        if self.iprint > 1:
            print(flush=True)
            print('  Iter ',self.itera_,flush=True)
            cout_ = 0
            for i in xk:
                print(' {:>20.16f}  '.format(i),end='')
                cout_ += 1
                if cout_ == 5:
                    print(flush=True)
                    cout_ = 0
            print(flush=True)
        self.itera_ += 1

    def efunc(self, ini_state, pobj, hobj, solver, nstep, normalize,
              supdate=False,twindow=True,cobj=False,tmp=0,exactV=[]):
              
        from ctrlq.lib.solve import pulsec

        if not cobj:
            stseq = numpy.array(pobj.tseq) / pobj.tscale
            sfreq = numpy.array(pobj.freq) / pobj.fscale
            
            pobjc = pulsec(pobj.amp, stseq, sfreq, pobj.duration,
                           pobj.nqubit, pobj.nwindow)
            out = evolve(ini_state, pobjc, hobj, solver=solver, nstep=nstep,twindow=twindow)
        else:
            out = evolve(ini_state, pobj, hobj, solver=solver, nstep=nstep,twindow=twindow)

        state_ = []
        for i in hobj.states:
            state_.append(out[i])
        state_ = numpy.array(state_)

        if exactV:
            ovlp = numpy.dot(state_.conj().T, exactV)
            print(' Overlap           : {:>8.4f}     %'.format(ovlp *100.))

        if normalize:
            nrm = numpy.linalg.norm(state_)
            if supdate:
                self.leak = 1.0 - nrm
                
            staten = state_ /nrm
        else:
            if supdate:
                nrm = numpy.linalg.norm(state_)
                self.leak = 1.0 - nrm
                
            staten = state_
            
        energy = functools.reduce(numpy.dot,
                                  (staten.conj().T, hobj.mham, staten))        
        energy = energy.real[0][0]
        
        if supdate:
            self.energy_ = energy
            
        return energy

    def gfunc(self, list1, ini_state, pobj, hobj, solver, nstep, normalize):
        
        cout = 0   
        for i in range(pobj.nqubit):
            for j in range(pobj.nwindow):
        
                pobj.amp[i][j] = list1[cout]
                cout += 1
        
        #for i in range(pobj.nqubit):
        #    for j in range(pobj.nwindow -1 ):
        #
        #        pobj.tseq[i][j] = list1[cout]
        #        cout += 1
        
        for i in range(pobj.nqubit):
            pobj.freq[i] = list1[cout]
            cout += 1
        E_ = self.efunc(ini_state, pobj, hobj, solver, nstep, normalize)
        
        return E_
        
    def objfunc(self, list1, pobj, hobj, solver, nstep, normalize,
                grad_check, misc = False):
        
        from scipy.optimize import approx_fprime
        cout = 0   
        for i in range(pobj.nqubit):
            for j in range(pobj.nwindow):
        
                pobj.amp[i][j] = list1[cout]
                cout += 1
                
        #for i in range(pobj.nqubit):
        #    for j in range(pobj.nwindow -1 ):
        #
        #        pobj.tseq[i][j] = list1[cout]
        #        cout += 1
        
        for i in range(pobj.nqubit):
            pobj.freq[i] = list1[cout]
            cout += 1
        
        ini_state = hobj.initial_state

        E_ = self.efunc(ini_state, pobj, hobj, solver, nstep, normalize, supdate=True)

        G_ = approx_fprime(list1, self.gfunc, 1e-8, ini_state, pobj, hobj, solver, nstep, normalize)

        self.gradient_ = G_
        
        return (E_, G_)
        
