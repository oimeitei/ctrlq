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

import numpy, functools, scipy
from scipy.optimize import approx_fprime
import sys
from ctrlq.lib.solve import pulsec
from ctrlq.cvqe.pulse import pulse
from ctrlq.cvqe.evolve import evolve


                            

class control:

    """ Control class: kernel for ctrl-VQE

    Parameters
    ----------
    pulse : ctrq.cvqe.pulse
        Pulse
    ham : ctrq.cvqe.ham
        Transmon Hamiltonian
    solver : str
        Solver for time evolution. Supported, 'ode', 'trotter'. Defaults to 'ode'.
    nstep : int
        Time step. Defaults to 500. 

    """

    def __init__(self, pulse_, ham, solver='ode', nstep=500):
        
        self.pulse = pulse_
        self.ham = ham
        self.solver = solver
        self.nstep = nstep
        self.energy_ = 0.0
        self.leak = 0.0
        self.itera_ = 0


    def print_step(self, xk):
        self.itera_ += 1

        print(flush=True)
        print('---------------',flush=True)
        print('Iteration: ',self.itera_,flush=True)
        print('---------------',flush=True)
        print(flush=True)
        print(' Pulse parameters: Amplitude, time window (if any), frequency ')

        cout_ = 0
        for i in xk:
            print('{:>20.16f}  '.format(i),end='')
            cout_ += 1
            if cout_ == 5:
                print(flush=True)
                cout_ = 0
        print(flush=True)
        print(flush=True)
        print('Energy    :  {:>20.16f} Ha'.format(self.energy_),flush=True)
        print('Leakage   :  {:>20.16f} %'.format(self.leak*100.),flush=True)
        print(flush=True)

    def efunc(self, ini_state, pobj, hobj, solver, nstep, normalize, supdate=False):
        stseq = numpy.array(pobj.tseq) / pobj.tscale
        sfreq = numpy.array(pobj.freq) / pobj.fscale

        pobjc = pulsec(pobj.amp, stseq, sfreq, pobj.duration,
                       pobj.nqubit, pobj.nwindow)
        out = evolve(ini_state, pobjc, hobj, solver=solver, nstep=nstep)

        state_ = []
        for i in hobj.states:
            state_.append(out[i])
        state_ = numpy.array(state_)

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

    def gfunc1(self, list1, ini_state, pobj, hobj, solver, nstep, normalize):
        
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
        
    def objfunc(self, list1, pobj, hobj, solver, nstep, normalize, grad_check, misc = False):
        
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

        G_ = approx_fprime(list1, self.gfunc1, 1e-8, ini_state, pobj, hobj, solver, nstep, normalize)
        
        return (E_, G_)
        

    def optimize(self, method = 'l-bfgs-b', maxiter=100, maxls = 20,
                 gtol = 1.0e-12, ftol = 1.0e-12, exactE = 0.0,
                 normalize = False, echeck=False):
        """Variational pulse optimization
        Perform ctrl-VQE pulse optimization using the expectation 
        value of the molecular hamiltonian supplied in ham class.

        Parameters
        ----------
        method : str
             Numerical optimization method. Only supports 'l-bfgs-b'.
        normalize : bool
             Whether to normalize the expectation value. Defaults to False.
        maxiter : int
             Maximum number of iteration for optimization.
        maxls : int
             Set the maximum number of line searches in each optimization step.

        """
        
        print(flush=True)
        print('********************************',flush=True)
        print('***********  CtrlQ  ************',flush=True)
        print('********************************',flush=True)
        print(flush=True)
        print('Variational pulse shaping: ctrl-VQE',flush=True)
        
        if echeck:
            print(flush=True)
            print(' Checking for the sign of correlation energy ',flush=True)
            print(' For total molecular electronic energy, restart with echeck=False')
            print(flush=True)
            tmp_check_ = True
            iter_ = 1
            while tmp_check_:
                e_, l_ = self.energy(normalize=normalize,iprint=False)
                if e_ > 0.0e0:

                    #tmp_amp = self.pulse.amp
                    tmp_tseq = self.pulse.tseq
                    #tmp_freq = self.pulse.freq
                    print (' Iteration : {:>3} Energy is positive, new initial guess'.format(iter_),flush=True)
                    self.pulse = pulse(duration = self.pulse.duration,
                                       nqubit = self.pulse.nqubit,
                                       nwindow = self.pulse.nwindow,
                                       fscale = self.pulse.fscale,
                                       tscale = self.pulse.tscale)
                    #self.pulse.amp = tmp_amp
                    self.pulse.tseq = tmp_tseq
                    #self.pulse.freq[0] = tmp_freq
                else:
                    print (' Iteration : {:>3} Energy is negative, using this initial guess'.
                           format(iter_),flush=True)
                    
                    tmp_check_ = False
                #print(self.pulse.amp,end='')
                #print(self.pulse.tseq,end='')
                #print(self.pulse.freq)
                iter_ += 1
                if iter_ > 100:
                    print(' No. of iteration in echeck exceeded, restart optimization',flush=True)
                    sys.exit('No luck in random search')
        print()
        # initial parameters
        ilist = []
        if method == 'l-bfgs-b':
           
           for i in self.pulse.amp:
               for j in i:
                   ilist.append(j)
           print(' TSEQ fixed at ')
           for i in self.pulse.tseq:
               
               for j in i:
                   #ilist.append(j)
                   print('{:>20.16f} '.format(j),end='')
               print()
           print()
                   
           for i in self.pulse.freq:
               ilist.append(i)

           print('---------------',flush=True)
           print('Iteration: ',self.itera_,flush=True)
           print('---------------',flush=True)
           print(flush=True)
           print('Initial guess: Amplitude, time window (if any), frequency',flush=True)
           cout_ = 0
           for i in ilist:
               print('{:>20.16f}  '.format(i),end='')
               cout_ += 1
               if cout_ == 5:
                   print(flush=True)
                   cout_ = 0
           print(flush=True)
           print(flush=True)

           

           E_ = self.efunc(self.ham.initial_state, self.pulse, self.ham, self.solver,
                           self.nstep, normalize, supdate=True)
           print('Energy    :  {:>20.16f} Ha'.format(self.energy_),flush=True)
           print('Leakage   :  {:>20.16f} %'.format(self.leak*100.),flush=True)
           print(flush=True)
           
               
           res1 = scipy.optimize.minimize(
               self.objfunc, ilist, args = (self.pulse, self.ham, self.solver,
                                            self.nstep,normalize,False),
               method=method,jac=True,
               bounds = self.pulse.constraint,
               callback = self.print_step,
               options = {'maxiter':maxiter,'gtol':gtol,
                          'ftol':ftol,'maxls':maxls}) #, 'iprint':1,'disp':1})

           print('---------------',flush=True)
           print('Final result',flush=True)
           print('---------------',flush=True)
           print(flush=True)

           cout = 0
           qu = 1
           print('| Amplitudes')
           for i in self.pulse.amp:
               print('Qubit ',qu,' : ',end='')
               qu += 1
               for j in i:
                   print('{:>20.16f}  '.format(res1.x[cout]),end='')
                   cout += 1
               print()
           print()
               
           qu = 1
           print('| Time Windows')
           for i in self.pulse.tseq:
               print('Qubit ',qu,' : ',end='')
               qu += 1               
               for j in i:
                   #print('{:>20.16f}  '.format(res1.x[cout]),end='')
                   #cout += 1                   
                   print('{:>20.16f} '.format(j),end='')
               print()
           print()
           
           qu = 1
           print('| Frequencies')
           for i in self.pulse.freq:
               print('Qubit ',qu,' : ',end='')
               qu += 1               
               print('{:>20.16f}  '.format(res1.x[cout]))
               cout += 1 
           print()
           print()
           print('Energy    :  {:>20.16f} Ha'.format(self.energy_),flush=True)
           print('Leakage   :  {:>20.16f} %'.format(self.leak*100.),flush=True)
           print()
           print()
           return (res1.fun,self.leak)

    def energy(self, normalize=False, iprint=True):
        """

        Parameters
        ----------
        normalize : bool
                Whether to normalize the expectation value. Defaults to False.
        iprint : bool
                Whether to print Energy (expectation value) and leakage. Defaults to False.
        """
        e1 = self.efunc(self.ham.initial_state, self.pulse, self.ham,
                        self.solver,self.nstep,normalize, supdate=True)
        #e1, l1 = expect_(self.ham.initial_state, self.pulse, self.ham,
        #                 solver=self.solver, nstep= self.nstep,
        #                 leak_=True, normalize=normalize)
        if iprint:
            print()
            print(' Energy (ctrl-vqe) : {:>18.12f} H'.format(e1))
            print(' Leakage           : {:>18.12f}     %'.format(self.leak*100.))
        return (e1,self.leak)
                            

