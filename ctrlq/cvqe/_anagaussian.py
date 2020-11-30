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
import numpy,sys

def anagaus(self, list1, pobj, hobj, nstep,normalize):
    
    from ctrlq.lib.solve import pulsec
    from ctrlq.lib.agradc import grad_trotter
    from ctrlq.lib.pulse_helper import gaus_getnamp 

    # Analytic gradient for gaussian pulse
    # list1 [[[amp, mean, sigma] for ngaus] for nqubit]
    
    ini_state = hobj.initial_state
    tlist = numpy.linspace(0, pobj.duration, nstep)
    dsham = numpy.diagonal(-1j * hobj.dsham.toarray())

    cout = 0
    for i in range(pobj.nqubit):
        for j in range(pobj.ngaus):
            pobj.amp[i][j] = list1[cout]
            cout += 1
            pobj.mean[i][j] = list1[cout]
            cout += 1
            pobj.sigma[i][j] = list1[cout]
            cout += 1

    sfreq = numpy.array(pobj.freq) / pobj.fscale
    ggrad = gaus_getnamp(pobj.nqubit, pobj.ngaus, pobj.duration,
                             pobj.amp,pobj.sigma, pobj.mean, sfreq,
                             tlist, ini_state, numpy.array(hobj.hdrive),
                             dsham, hobj.states, hobj.mham)

    pobj.gausamp_ = ggrad.amp
    
    pobjc = pulsec(ggrad.amp, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    
    gradient = ggrad.gradient
    for i in range(pobj.nqubit):
        # pobjc.amp - list of amp for every trotter step
        ng1 = self.numgradfreq_2_(i, ggrad.energy, ini_state, pobjc, hobj, nstep, cobj=True)
        gradient.append(ng1)

    self.energy_ = ggrad.energy
    self.leak = 1.0 - ggrad.norm
    self.gradient_ = gradient

    return(ggrad.energy, gradient)

def anagaus_py(self, list1, pobj, hobj, nstep):
    
    from ctrlq.lib.solve import pulsec
    from ctrlq.lib.agradc import grad_trotter
    from ctrlq.lib.pulse_helper import gaus_getnamp #,gaus_gettamp

    # Analytic gradient for gaussian pulse
    # list1 [[[amp, mean, sigma] for ngaus] for nqubit]
    
    ini_state = hobj.initial_state
    tlist = numpy.linspace(0, pobj.duration, nstep)
    dsham = numpy.diagonal(-1j * hobj.dsham.toarray())

    cout = 0
    for i in range(pobj.nqubit):
        for j in range(pobj.ngaus):
            pobj.amp[i][j] = list1[cout]
            cout += 1
            pobj.mean[i][j] = list1[cout]
            cout += 1
            pobj.sigma[i][j] = list1[cout]
            cout += 1

    sfreq = numpy.array(pobj.freq) / pobj.fscale
    
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
    
            tmpamp_ = 0.0
            for k in range(pobj.ngaus):
                tmpamp_ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
    
            amp__[i].append(tmpamp_)

    pobj.gausamp_ = amp__    
    pobjc_ = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    aobj = grad_trotter(tlist, ini_state, pobjc_, numpy.array(hobj.hdrive),
                        dsham, hobj.states, hobj.mham)

    agradient__ = []
    for i in range(pobj.nqubit):
        for j in range(pobj.ngaus):
            gamp_ = 0.0
            gsig_ = 0.0
            gmean = 0.0
            for k in range(nstep):
                
                esigmean_ = aobj.gradient[i][k] * exp(-pobj.sigma[i][j]**2 *
                                                      (tlist[k] - pobj.mean[i][j])**2)
                
                gamp_ += esigmean_
                gsig_ += esigmean_ * pobj.amp[i][j] * -2.0 * pobj.sigma[i][j] *\
                         (tlist[k] - pobj.mean[i][j])**2
                gmean += esigmean_ * pobj.amp[i][j] * 2.0 * pobj.sigma[i][j]**2 *\
                         (tlist[k] - pobj.mean[i][j])
    
            agradient__.extend([gamp_, gsig_, gmean])

    # freq
    for i in range(pobj.nqubit):
        # pobjc.amp - list of amp for every trotter step
        ng1 = self.numgradfreq_2_(i, ggrad.energy, ini_state, pobjc, hobj, nstep, cobj=True)
        agradient__.append(ng1)

    self.energy_ = aobj.energy
    self.leak = 1.0 - aobj.norm

    return(aobj.energy, agradient__)


def numgradamp_gaus(self, iqubit, igaus, ini_state, pobj, hobj, tlist, nstep, delx=0.000005):
    
    from ctrlq.lib.solve import pulsec

    pobj.amp[iqubit][igaus] += delx
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
            tmpamp__ = 0.0
            for k in range(pobj.ngaus):
                tmpamp__ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
            amp__[i].append(tmpamp__)
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    f1 = self.efunc(ini_state, pobjc, hobj, 'trotter', nstep,
                    False, twindow=False, cobj=True)
    
    pobj.amp[iqubit][igaus] -= 2*delx
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
            tmpamp__ = 0.0
            for k in range(pobj.ngaus):
                tmpamp__ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
            amp__[i].append(tmpamp__)
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    f2 = self.efunc(ini_state, pobjc, hobj, 'trotter', nstep,
                    False, twindow=False, cobj=True)
    
    g_ = (f1-f2)/(2*delx)
    pobj.amp[iqubit][igaus] += delx
    return g_

def numgradmean_gaus(self, iqubit, igaus, ini_state, pobj, hobj, tlist, nstep, delx=0.00005):
    
    from ctrlq.lib.solve import pulsec
    
    pobj.mean[iqubit][igaus] += delx
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
            tmpamp__ = 0.0
            for k in range(pobj.ngaus):
                tmpamp__ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
            amp__[i].append(tmpamp__)
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    f1 = self.efunc(ini_state, pobjc, hobj, 'trotter', nstep,
                    False, twindow=False, cobj=True)
    
    pobj.mean[iqubit][igaus] -= 2*delx
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
            tmpamp__ = 0.0
            for k in range(pobj.ngaus):
                tmpamp__ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
            amp__[i].append(tmpamp__)
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    f2 = self.efunc(ini_state, pobjc, hobj, 'trotter', nstep,
                    False, twindow=False, cobj=True)
    
    g_ = (f1-f2)/(2*delx)
    pobj.mean[iqubit][igaus] += delx
    return g_

def numgradsigma_gaus(self, iqubit, igaus, ini_state, pobj, hobj, tlist, nstep, delx=0.0005):
    
    from ctrlq.lib.solve import pulsec
    
    pobj.sigma[iqubit][igaus] += delx
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
            tmpamp__ = 0.0
            for k in range(pobj.ngaus):
                tmpamp__ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
            amp__[i].append(tmpamp__)
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    f1 = self.efunc(ini_state, pobjc, hobj, 'trotter', nstep,
                    False, twindow=False, cobj=True)
    
    pobj.sigma[iqubit][igaus] -= 2*delx
    amp__ = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        for j in tlist:
            tmpamp__ = 0.0
            for k in range(pobj.ngaus):
                tmpamp__ += pobj.amp[i][k] * exp(-pobj.sigma[i][k]**2 * (j - pobj.mean[i][k])**2)
            amp__[i].append(tmpamp__)
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp__, [[0.0]], sfreq, pobj.duration, pobj.nqubit, 0)
    f2 = self.efunc(ini_state, pobjc, hobj, 'trotter', nstep,
                    False, twindow=False, cobj=True)
    
    g_ = (f1-f2)/(2*delx)
    pobj.mean[iqubit][igaus] += delx
    return g_
