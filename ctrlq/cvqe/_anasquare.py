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

import numpy,sys
from copy import deepcopy

def twinamp(x, y, list1):
    cout_ = 0
    
    for i in list1:
        if x<i<=y:
            cout_ += 1
            
    return cout_

def anatwin(self, list1, pobj, hobj, nstep,normalize):
    
    from ctrlq.lib.solve import pulsec
    from ctrlq.lib.agradc import grad_trotter
    from ctrlq.lib.agradc import grad_trotter_normalized
    
    ini_state = hobj.initial_state
    tlist = numpy.linspace(0, pobj.duration, nstep)
    dsham = numpy.diagonal(-1j * hobj.dsham.toarray())

    ampcout = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        if pobj.nwindow==1:
            ampcout[i].append(twinamp(0.0,pobj.duration, tlist))
            continue
        for j in range(pobj.nwindow-1):

            if j == 0:
                ampcout[i].append(twinamp(0.0,pobj.tseq[i][j], tlist))
            else:
                ampcout[i].append(twinamp(pobj.tseq[i][j-1], pobj.tseq[i][j], tlist))
        ampcout[i].append(twinamp(pobj.tseq[i][j], pobj.duration, tlist))

    amp = [[] for j in range(pobj.nqubit)]
    samp = [[] for j in range(pobj.nqubit)]
    cout = 0
    for i in range(pobj.nqubit):
        for j in range(pobj.nwindow):
            
            if j == 0:
                amp[i].append(0.0) #amp=0.0 at t=0.0            
            for k in range(ampcout[i][j]):
                amp[i].append(list1[cout])
            samp[i].append(list1[cout])
            cout += 1

    self.square_amp = samp
    pobj.amp = amp
                
    for i in range(pobj.nqubit):
        pobj.freq[i] = list1[cout]
        cout += 1
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp, pobj.tseq, sfreq, pobj.duration,
                   pobj.nqubit, pobj.nwindow)

    if normalize:
        aobj = grad_trotter_normalized(tlist, ini_state, pobjc, numpy.array(hobj.hdrive),
                                       dsham, hobj.states, hobj.mham)
    else:        
        aobj = grad_trotter(tlist, ini_state, pobjc, numpy.array(hobj.hdrive),
                                       dsham, hobj.states, hobj.mham)
    
    self.square_gamp = aobj.gradient
    
    agradient = []
    for i in range(pobj.nqubit):
        cout_ = 0
        for j in range(pobj.nwindow):
            agrad__ = 0.0
            if j==0:                
                #agrad__ += aobj.gradient[i][0]                
                cout_ += 1
            for k in range(ampcout[i][j]):
                agrad__ += aobj.gradient[i][cout_]
                cout_ += 1
            agradient.append(agrad__)
        
    for i in range(pobj.nqubit):
        ng1 = self.numgradfreq_2_(i, aobj.energy, ini_state, pobj, hobj, nstep,normalize=normalize)
        agradient.append(ng1)
    self.energy_ = aobj.energy
    self.leak = 1.0 - aobj.norm
    self.gradient_ = agradient
    
    return(aobj.energy, agradient)
                   

