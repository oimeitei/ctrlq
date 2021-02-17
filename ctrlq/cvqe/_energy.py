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
from ctrlq.cvqe.evolve import evolve
from math import exp

def energy(self, normalize=True,  twindow=True, flattop=False, exactV = [], shape='square'):
    """

    Parameters
    ----------
    normalize : bool
            Whether to normalize the expectation value. Defaults to True.        
    twindow : bool
            Turn on/off piecewise function. Deaults to True.
    shape : str
         Shape of control pulse. 'square' and 'gaussian' forms are supported. Defaults to 'square'                
    exactV : list
            Supply the exact state to print out overlap (optional).
    """
    from ._flattop import gaussquare        
    from ctrlq.lib.solve import pulsec
    
    if flattop:
        e1, l1 = gaussquare(self.ham.initial_state, self.pulse, self.ham,
                            self.solver,self.nstep,normalize, leak=True)
        return(e1,self.leak)

    if shape=='gaussian':
           
        amp__ = [[] for i in range(self.pulse.nqubit)]
        tlist__ = numpy.linspace(0, self.pulse.duration, self.nstep)
        for i in range(self.pulse.nqubit):
            for j in tlist__:
                tmpamp__ =0.0
                for k in range(self.pulse.ngaus):
                    tmpamp__ += self.pulse.amp[i][k] * exp(-self.pulse.sigma[i][k]**2 *
                                                          (j-self.pulse.mean[i][k])**2)
                amp__[i].append(tmpamp__)
        sfreq__ = numpy.array(self.pulse.freq)/ self.pulse.fscale
        pobjc__ = pulsec(amp__, [[0.0]], sfreq__, self.pulse.duration,
                         self.pulse.nqubit, 0)
        e1 = self.efunc(self.ham.initial_state, pobjc__, self.ham, self.solver,
                        self.nstep, normalize, supdate=True, twindow=False, cobj=True)
    elif shape=='square':    
    
        e1 = self.efunc(self.ham.initial_state, self.pulse, self.ham,
                        self.solver,self.nstep,normalize, supdate=True,twindow=twindow,
                        exactV =exactV )
    else:
        sys.exit('Pulse shape not supported. Try square, gaussian or flattop')
    
    if self.iprint:
        print()
        print(' Energy (ctrl-vqe) : {:>18.12f} H'.format(e1))
        print(' Leakage           : {:>18.12f}     %'.format(self.leak*100.))
    return (e1,self.leak)

def HF(self):
    """

    Get Hartree Fock energy.
    """
    from .omisc import initial_state
    
    istate_ = initial_state(self.ham.istate, 2)
    EHF = functools.reduce(numpy.dot, (istate_.T, self.ham.mham, istate_))

    if self.iprint > 0:
        print(flush=True)
        print(' Energy (HF)       : {:>18.12f} H'.format(EHF.real[0][0]),flush=True)

    return EHF.real[0][0]
                            

