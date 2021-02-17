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

def numgradw_2_(self,idx,f1, ini_state, pobj, hobj, nstep, device_,interact,
                delx=0.00005, twindow = False, cobj=False, normalize=False):

    from .drift import transmon4_static
    from .ham import transmon

    # 2-point stencil numerical gradient for the device_.w[idx]

    pi2 = 2 * numpy.pi
    device_.w[idx] += delx * pi2

    hstatic_ = transmon4_static(param=device_, interact=interact)
    hobj_ = transmon(mham= hobj.mham, nqubit=hobj.nqubit, Hstatic=hstatic_)

    f2 = self.efunc(ini_state, pobj, hobj_, 'trotter', nstep,
                    normalize, twindow=twindow, cobj=cobj) #,tmp=idx) ###

    g_ = (f2-f1)/(delx * pi2)

    device_.w[idx] -= delx * pi2

    return g_

def numgradfreq_2_(self,idx,f1, ini_state, pobj, hobj, nstep,
                   delx=0.00005, twindow = False, cobj=False, normalize=False):
    # 2-point stencil numerical gradient for the freq[idx]
    # in pobj

    pobj.freq[idx] += delx 
    f2 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow, cobj=cobj,tmp=idx)
    pobj.freq[idx] -= delx 
    
    g_ = (f2-f1)/(delx)

    return g_

def numgradfreq_3_(self,idx, ini_state, pobj, hobj, nstep,
                   delx=0.00005, twindow = False,normalize=False):
    # 3-point stencil numerical gradient for the freq[idx]
    # in pobj

    pobj.freq[idx] += delx
    f1 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow)

    pobj.freq[idx] -= 2* delx
    f2 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow)
    
    g_ = (f1-f2)/(2*delx)

    return g_

def numgradamp_3_(self,iqubit, iwindow, ini_state, pobj, hobj, nstep,
                   delx=0.00005, twindow = False, cobj=False,normalize=False):
    # 3-point stencil numerical gradient for the amp[iqubit][iwindow]
    # in pobj
    
    pobj.amp[iqubit][iwindow] += delx

    f1 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow, cobj=cobj)
    
    pobj.amp[iqubit][iwindow] -= 2* delx
    f2 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow, cobj=cobj)

    pobj.amp[iqubit][iwindow] += delx
    
    g_ = (f1-f2)/(2*delx)

    return g_

def numgradamp_5_(self,iqubit, iwindow, ini_state, pobj, hobj, nstep,
                   delx=0.00005, twindow = False,normalize=False):
    # 5-point stencil numerical gradient for the amp[iqubit][iwindow]
    # in pobj

    pobj.amp[iqubit][iwindow] += 2*delx
    f1 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow)

    pobj.amp[iqubit][iwindow] -= delx
    f2 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow)

    pobj.amp[iqubit][iwindow] -= 2* delx
    f3 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow)

    pobj.amp[iqubit][iwindow] -= delx
    f4 = self.efunc(ini_state, pobj, hobj, 'trotter', nstep,
                    normalize, twindow=twindow)

    pobj.amp[iqubit][iwindow] += 2*delx
    
    g_ = (-f1+8*f2-8*f3+f4)/(12*delx)

    return g_

def gradfunc(self, list1, pobj, hobj, nstep, test=False):
    import functools,operator, math
    from ctrlq.lib.solve import pulsec
    from ctrlq.lib.agradc import grad_trotter
    
    ini_state = hobj.initial_state
    tlist = numpy.linspace(0, pobj.duration, nstep)
    dsham = numpy.diagonal(-1j * hobj.dsham.toarray())

    
    cout = 0   
    for i in range(pobj.nqubit):
        for j in range(pobj.nwindow):
            pobj.amp[i][j] = list1[cout]
            cout += 1
            
    for i in range(pobj.nqubit):
        pobj.freq[i] = list1[cout]
        cout += 1
   
        
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(pobj.amp, pobj.tseq, sfreq, pobj.duration,
                   pobj.nqubit, pobj.nwindow)

    aobj = grad_trotter(tlist, ini_state, pobjc, numpy.array(hobj.hdrive),
                        dsham, hobj.states, hobj.mham)

    if test:
        cout_ = 0
        for i in range(pobj.nqubit):
            for j in range(pobj.nwindow):
                numag_ = self.numgradamp_5_(i,j,ini_state, pobj,hobj,nstep,delx=1e-8)
                if not math.isclose(numag_, aobj.gradient[i][j],abs_tol=1e-6):
                    cout_ += 1
        print('There are '+str(cout_)+' gradients with absolute difference of more than 1e-6')
        
    agradient = functools.reduce(operator.iconcat, aobj.gradient, [])


    
    for i in range(pobj.nqubit):
        ng1 = self.numgradfreq_2_(i, aobj.energy, ini_state, pobj, hobj, nstep)
        agradient.append(ng1)

    self.energy_ = aobj.energy
    self.leak = 1.0 - aobj.norm


    return (aobj.energy, agradient)
