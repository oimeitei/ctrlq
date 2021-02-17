import scipy, numpy, itertools
import scipy.linalg
from .device import *
from .drift import *
from .ham import *


def twinamp(x, y, list1):
    cout_ = 0

    for i in list1:
        if x<i<=y:
            cout_ += 1

    return cout_

def objfunc_wpulse(self, list1, pobj, hobj, nstep, normalize,
                   interact, freqopt):
    
    from ctrlq.lib.solve import pulsec
    from ctrlq.lib.agradc import grad_trotter
    from ctrlq.lib.agradc import grad_trotter_normalized

    pi2 = 2 * numpy.pi

    device_ = device4() #
    ini_state = hobj.initial_state
    tlist = numpy.linspace(0, pobj.duration, nstep)
    
    ampcout = [[] for i in range(pobj.nqubit)]
    for i in range(pobj.nqubit):
        if pobj.nwindow==1:
            ampcout[i].append(twinamp(0.0,pobj.duration, tlist))
            continue
        for j in range(pobj.nwindow-1):

            if j == 0:
                ampcout[i].append(twinamp(0.0,pobj.tseq[i][j], tlist))
            else:
                ampcout[i].append(twinamp(pobj.tseq[i][j-1],
                                          pobj.tseq[i][j], tlist))
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

    if freqopt:
        for i in range(pobj.nqubit):
            pobj.freq[i] = list1[cout]
            cout += 1

    for i in range(pobj.nqubit):
        device_.w[i] = pi2 * list1[cout]
        cout += 1
        
    sfreq = numpy.array(pobj.freq) / pobj.fscale
    pobjc = pulsec(amp, pobj.tseq, sfreq, pobj.duration,
                   pobj.nqubit, pobj.nwindow)
    
    hstatic_ = transmon4_static(param=device_, interact=interact)
    hobj_ = transmon(mham= hobj.mham, nqubit=hobj.nqubit, Hstatic=hstatic_)
    dsham = numpy.diagonal(-1j * hobj_.dsham.toarray())
    
    if normalize:
        aobj = grad_trotter_normalized(tlist, ini_state, pobjc,
                                       numpy.array(hobj_.hdrive),
                                       dsham, hobj_.states, hobj_.mham)
    else:        
        aobj = grad_trotter(tlist, ini_state, pobjc, numpy.array(hobj_.hdrive),
                                       dsham, hobj_.states, hobj_.mham)

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

    if freqopt:
        for i in range(pobj.nqubit):
            ng1 = self.numgradfreq_2_(i, aobj.energy, ini_state,
                                      pobj, hobj_, nstep,normalize=normalize)
            agradient.append(ng1)
    
    for i in range(pobj.nqubit):
        ng1 = self.numgradw_2_(i, aobj.energy, ini_state, pobj,
                               hobj_, nstep, device_, interact,
                               normalize=normalize)
        agradient.append(ng1)

    self.energy_ = aobj.energy
    self.leak = 1.0 - aobj.norm
    self.gradient_ = agradient

    return(aobj.energy, agradient)
    


def opt_wpulse(self, method = 'l-bfgs-b', maxiter=100, maxls = 20,
               gtol = 1.0e-09, ftol = 1.0e-09, exactE = 0.0,
               normalize = False, interact=['01','03','12','23'],
               device_ = device4(),bounds=[],gradient='analytical',
               freqopt = True, twindow=True,
               shape='square', optiter=True,
               pulse_return=False, w_bound = 2.0):

    from ctrlq.lib.solve import pulsec

    if not freqopt:
        for i in range(self.pulse.nqubit):
            del self.pulse.constraint[-1]
            
    pi2 = 2 * numpy.pi
    if bounds == []:
        apple = 0
        w_bound_ = []
        for i in range(self.pulse.nqubit):
            w_bound_.append((device_.w[i]/pi2 - w_bound,
                             device_.w[i]/pi2 + w_bound))
        self.pulse.constraint.extend(w_bound_)

    ilist = []

    # only square pulse
    for i in range(self.pulse.nqubit):
        for j in range(self.pulse.nwindow):
            ilist.append(self.pulse.amp[i][j])

    if freqopt:
        for i in self.pulse.freq:
            ilist.append(i)

    if self.iprint > 1:
        print('  -----* Entering pulse optimizations *-----',flush=True)
        print('  Pulse parameters: Amplitude, time window (if any), frequency',flush=True)

    E_ = self.efunc(self.ham.initial_state, self.pulse,
                    self.ham, self.solver,
                    self.nstep, normalize, supdate=True, twindow=twindow)

    # adding device w
    for i in range(self.pulse.nqubit):
        ilist.append(device_.w[i]/pi2)

    res1 = scipy.optimize.minimize(
        self.objfunc_wpulse, ilist, args = (self.pulse, self.ham,
                                            self.nstep,normalize,
                                            interact,freqopt),
        method=method,jac=True,
        bounds = self.pulse.constraint,
        callback = self.print_step,
        options = {'maxiter':maxiter,'gtol':gtol,
                   'ftol':ftol,'maxls':maxls}) #, 'iprint':1000,'disp':1000})    

    if self.iprint > 0:
        print(flush=True)
        print('  Pulse optimization ends',flush=True)
        if optiter:
            print(' ',res1.message)
        print('  ------------------------------------------',
              flush=True)
        print(flush=True)
        print('  Printing progress',flush=True)
        print('  --------------------------------------------------------',
              flush=True)
        print('  --------------------------------------------------------',
              flush=True)
        print('  Iter       Energy(H)      Leak(%)  Ediff(H)    Gnorm',
              flush=True)
        for i in range(len(self.elist_)):
            if i == 0:
                idx__ = 0
            else:
                idx__ = i - 1
            print('  {:>3d}  {:>18.12f}  {:>7.4f}  {:>.4e}  {:>.4e}'.format(
                i,self.elist_[i],self.llist_[i]*100.,
                abs(self.elist_[i]-self.elist_[idx__]),
                numpy.linalg.norm(self.glist_[i])))
        print('  --------------------------------------------------------',
              flush=True)
        print('  --------------------------------------------------------',
              flush=True)
        print(flush=True)
        self.elist_ = []
        self.llist_ = []
        self.glist_ = []
    if exactE:
            print('  Error in ctrl-VQE energy : {:>.4e}'.format(self.energy_-exactE),
                  flush=True)

    if optiter:
        if self.iprint > 0:
           cout = 0
           print(flush=True)
           print('  -----* Optimal pulse parameters *-----',flush=True)
                              
           if shape=='square':
               print('  | Amplitudes',flush=True)
               for i in range(self.pulse.nqubit):
                   print('  Qubit ',i+1,' : ',end='',flush=True)
                   
                   cout_ = 0
                   for j in range(self.pulse.nwindow):
                       if cout_ == 4:
                           print(flush=True)
                           print('              ',end='',flush=True)
                           cout_ = 0
                       print(' {:>20.16f}  '.format(res1.x[cout]),end='',
                             flush=True)
                       cout_ += 1
                       cout += 1
                   print(flush=True)
               print(flush=True)
           
               qu = 1
               print('  | Time Windows',flush=True)
               for i in self.pulse.tseq:
                   print('  Qubit ',qu,' : ',end='',flush=True)
                   qu += 1
                   cout_ = 0
                   for j in i:
                       if cout_ ==4:
                           print(flush=True)
                           print('              ',end='',flush=True)
                           cout_ = 0
                           
                       print(' {:>20.16f} '.format(j),end='',flush=True)
                       cout_ += 1
                   print(flush=True)
               print(flush=True)
               
               print('  | Pulse frequency ',flush=True)
               for i in range(self.pulse.nqubit):
                   print('  Qubit ',i+1,' : ',end='',flush=True)
                   if freqopt:
                       print(' {:>20.16f}  '.format(res1.x[cout]),flush=True)
                       cout += 1
                   else:
                       print(' {:>20.16f}  '.format(self.pulse.freq[i]),flush=True)
               print(flush=True)
                

               print('  | Device frequency (w)',flush=True)
               for i in range(self.pulse.nqubit):
                   print('  Qubit ',i+1,' : ',end='',flush=True)
                   
                   print(' {:>20.16f}  '.format(res1.x[cout]),flush=True)
                   cout += 1
               print(flush=True) 

            
    if pulse_return:
        return(self.pulse, self.energy_,self.leak)
    return (self.energy_,self.leak)
    
