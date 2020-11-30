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

import numpy
from .device import device
from math import *

def pcoef(t, amp=None, tseq=None, freq=None, tfinal = 0.0,
          conj=False, scale=1.):

    sign_ = -1.0 if not conj else 1.0
    plist = []
    for i,j in enumerate(tseq):
        
        if not i:
            plist.append(0.0 < t <= j)
        else:
            plist.append(tseq[i-1] < t <= j)
            
    plist.append(tseq[-1] < t <= tfinal)
    
    coeff = numpy.piecewise(t+0j, plist,
                            [i * numpy.exp(sign_ * scale * 1j * freq * t)
                             for i in amp])
    return coeff    


class pulse:
    """Analog pulse class.
    Define control analog pulse shape. Square pulses are initialized with random guess.

    Parameters
    ----------
    shape : str
        Shape of control pulse. Defaults to 'square' which is the only one currently supported.
    nqubit : int
        Number of qubits. Defaults to 2.
    nwindow : int
        For square pulses, number of time windows (seqments). Defaults to 2.
    duration : float
        Total time duration of the pulse. Defaults to 10.0.
    fscale : float
        Scale the frequency of the pulse. Defaults to 1.0.
    tscale : float 
        Scale the time windows and the total time duration of the pulse. Defaults to 1.0.
    amp_bound : float
        Constrain for amplitude in 2pi GHz. Defaults to 0.02.
    freq_bound : float
        Constrain for drive frequency in 2pi GHz. Defaults to 1.00
    """    
    
    def __init__(self, shape='square', nqubit=2, nwindow=2,
                 ngaus=2, mean_constraint = None, sigma_constraint=None,
                 amp_constraint = None, tseq_constraint = None,
                 freq_constraint = None, fscale=1.0, tscale = 1.0,
                 duration = 10.0, constrain0=True,
                 freq_ini_restrict=False, freq_off = 0.6,
                 amp_bound = 0.02, freq_bound = 1.0):
        import random

        self.amp_bound = amp_bound
        self.freq_bound = freq_bound
        
        constraint = []
        freq = []
            
        if shape=='gaussian':
            amp = []
            mean = []
            sigma = []
            gausamp_ = []

            if not amp_constraint:
                amp_constraint = []
                for i in range(nqubit):
                    amp_ = []
                    for j in range(ngaus):
                        amp_.append(numpy.random.uniform(-numpy.pi*2*amp_bound,
                                                         numpy.pi*2*amp_bound))
                        amp_constraint.append((-numpy.pi*2*amp_bound,
                                               numpy.pi*2*amp_bound))
                    amp.append(amp_)
                constraint.extend(amp_constraint)
            elif isintance(amp_contraint, float):
                tmp_ = []
                amp_ = []
                for i in range(nqubit):
                    amp_ = []
                    for j in range(ngaus):
                        tmp_.append((-amp_constraint, amp_constraint))
                        amp_.append(numpy.random.uniform(-amp_constraint,
                                                         amp_constraint))
                    amp.append(amp_)
                constraint.extend(tmp_)
            else:
                constraint.extend(amp_constraint)

            if not sigma_constraint:
                sigma_constraint = []
                for i in range(nqubit):
                    sigma_ = []
                    for j in range(ngaus):
                        sigma_.append(numpy.random.uniform(0.025, 5.0))  
                        sigma_constraint.append(( 0.025, 5.0)) 
                    sigma.append(sigma_)
                constraint.extend(sigma_constraint)
            elif isintance(sigma_contraint, float):
                tmp_ = []
                sigma_ = []
                for i in range(nqubit):
                    sigma_ = []
                    for j in range(ngaus):
                        tmp_.append((0.1, sigma_constraint))
                        sigma_.append(numpy.random.uniform(0.1, sigma_constraint))
                    sigma.append(sigma_)
                constraint.extend(tmp_)
            else:
                constraint.extend(sigma_constraint)

            if not mean_constraint:
                mean_constraint = []
                for i in range(nqubit):
                    mean_ = []
                    for j in range(ngaus):
                        mean_constraint.append((0.0,duration*tscale))
                        mean_.append(numpy.random.uniform(0.0, duration*tscale))
                    mean.append(sorted(mean_))
            constraint.extend(mean_constraint)
            self.amp = amp
            self.mean = mean
            self.sigma = sigma
            self.ngaus = ngaus
            self.gausamp_ = gausamp_
            
        elif shape == 'square':
            amp= []
            tseq = []
            if not amp_constraint:
                amp_constraint = []
                for i in range(nqubit):
                    amp_ = []
                    for j in range(nwindow):
                        amp_.append(numpy.random.uniform(-numpy.pi*2*amp_bound,
                                                         numpy.pi*2*amp_bound))
                        if (j==0 and not constrain0):
                            continue
                        amp_constraint.append((-numpy.pi*2*amp_bound,
                                               numpy.pi*2*amp_bound))
                    amp.append(amp_)
                    
                constraint.extend(amp_constraint)
                
            elif isinstance(amp_constraint, float):
                tmp_ = []
                amp = []
                for i in range(nqubit):
                    amp_ = []
                    for j in range(nwindow):
                        tmp_.append((-amp_constraint, amp_constraint))
                        amp_.append(numpy.random.uniform(-amp_constraint,
                                                         amp_constraint))
                    amp.append(amp_)
                    
                constraint.extend(tmp_)

            else:
                constraint.extend(amp_constraint)                
            
            if not tseq_constraint:
                tseq_constraint = []
                for i in range(nqubit):
                    tseq_ = []
                    for j in range(nwindow-1):
                        tseq_constraint.append((0.0, duration*tscale))
                        tseq_.append(numpy.random.uniform(0.0, duration*tscale))

                    tseq.append(sorted(tseq_))
            
            self.amp = amp
            self.tseq = tseq
            self.nwindow = nwindow
        else:
            sys.exit('Pulse shape not yet implemented')

        if not freq_constraint:
            dp = device()
            freq_constraint = []
            for i in range(nqubit):
                freq_constraint.append(((dp.w[i]-freq_bound)*fscale,
                                        (dp.w[i]+freq_bound)*fscale))
                if not freq_ini_restrict:
                    freq.append(numpy.random.uniform((dp.w[i]-freq_bound)*fscale,
                                                     (dp.w[i]+freq_bound)*fscale))
                else:
                    rand1 = numpy.random.uniform((dp.w[i]-freq_bound)*fscale,
                                                 (dp.w[i]-freq_off)*fscale)
                    rand2 = numpy.random.uniform((dp.w[i]+freq_off)*fscale,
                                                 (dp.w[i]+freq_bound)*fscale)
                    freq.append(random.choice([rand1,rand2]))
                                
        constraint.extend(freq_constraint)
        self.constraint = constraint
        self.freq = freq
        self.duration = duration
        self.nqubit = nqubit
        self.fscale = fscale
        self.tscale = tscale
            
