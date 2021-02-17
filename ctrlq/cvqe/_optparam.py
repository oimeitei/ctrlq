
import scipy, numpy, itertools
import scipy.linalg
from .device import *
from .drift import *
from .ham import *

def objfunc_param(self, list1, pobj, hobj, solver, nstep, normalize,
                  grad_check,interact, misc=False):
    from scipy.optimize import approx_fprime

    # make hobj here
    device_ = device4()
    pi2 = 2 * numpy.pi
    cout = 0
    for i in interact:
        i1 = int(i[0])
        i2 = int(i[1])
        device_.g[i1][i2] = pi2 * list1[cout]
        cout += 1
    
    hstatic_ = transmon4_static(param=device_, interact=interact)
    hobj_ = transmon(mham= hobj.mham, nqubit=hobj.nqubit, Hstatic=hstatic_)
    
    ini_state = hobj.initial_state
    
    E_ = self.efunc(ini_state, pobj, hobj_, solver, nstep, normalize,
                    supdate=True)
    
    G_ = approx_fprime(list1, self.gfunc_param, 1e-8, ini_state, pobj, hobj_,
                       solver, nstep, normalize,interact)
    
    self.gradient_ = G_
        
    return (E_, G_)

def gfunc_param(self,list1, ini_state, pobj, hobj, solver, nstep, normalize,
                interact):
    
    # make hobj here
    device_ = device4()
    pi2 = 2 * numpy.pi

    cout = 0
    for i in interact:
        i1 = int(i[0])
        i2 = int(i[1])
        device_.g[i1][i2] = pi2 * list1[cout]
        cout += 1
    hstatic_ = transmon4_static(param=device_, interact=interact)
    hobj_ = transmon(mham= hobj.mham, nqubit=hobj.nqubit, Hstatic=hstatic_)
    
    E_ = self.efunc(ini_state, pobj, hobj_, solver, nstep, normalize)

    return E_

        

def print_param(self, xk):
    
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

    
def opt_param(self, method = 'l-bfgs-b', maxiter=100, maxls = 20,
              gtol = 1.0e-09, ftol = 1.0e-09, exactE = 0.0,
              normalize = False, interact=['01','03','12','23'],
              device_ = device4(),bounds=[],
              shape='square', optiter=True, pulse_return=False):

    
    from ctrlq.lib.solve import pulsec
    
    if bounds == []:
        for i in interact:
            bounds.append((0.000,0.030))
    
    pi2 = 2 * numpy.pi
    ilist = []
    for i in interact:
        i1 = int(i[0])
        i2 = int(i[1])

        ilist.append(device_.g[i1][i2]/pi2)

    print(ilist)
    E_ = self.efunc(self.ham.initial_state, self.pulse, self.ham, self.solver,
                    self.nstep, normalize, supdate=True, twindow=True)

    
    res1 = scipy.optimize.minimize(
        self.objfunc_param,ilist,args=(self.pulse, self.ham, self.solver,
                                       self.nstep,normalize,False,interact),
        method=method,jac=True, callback=self.print_param,
        bounds=bounds,
        options = {'maxiter':maxiter,'gtol':gtol,
                   'ftol':ftol,'maxls':maxls}) #, 'iprint':1,'disp':1})
    
    

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
            
    if optiter:
        if self.iprint > 0:
            print(flush=True)
            cout = 0
            print('Optimal coupling constants')
            for i in interact:
                print(' g{:>2} :  {:>20.16f}'.format(i,res1.x[cout],flush=True))
                cout += 1
            print(flush=True)
    if pulse_return:
        return(self.pulse, self.energy_,self.leak)
    return (self.energy_,self.leak)        
