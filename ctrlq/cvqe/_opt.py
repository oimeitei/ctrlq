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

def optimize(self, method = 'l-bfgs-b', maxiter=100, maxls = 20,
             gtol = 1.0e-09, ftol = 1.0e-09, exactE = 0.0,
             normalize = False, gradient = 'numerical',
             shape='square', optiter=True, pulse_return=False):
    """Variational pulse optimization:
    Perform ctrl-VQE pulse optimization using the expectation 
    value of the molecular hamiltonian supplied in ham class.

    Parameters
    ----------
    method : str
         Numerical optimization method. Only supports 'l-bfgs-b'.
    normalize : bool
         Whether to normalize the expectation value. Defaults to False.
    maxiter : int
         Maximum number of iteration for optimization. Defaults to 100.
    maxls : int
         Set the maximum number of line searches in each optimization step (l-bfgs-b line search). Defaults to 20.
    gradient : str
         Method for computing the gradients in the optimization. 'numerical' for numerical gradients for all pulse parameters. 'analytical' for an analytical gradients for the amplitudes and numerical gradients for the frequencies. Defaults to 'numerical'.
    optiter : bool
         If set False, energy and gradients for the initial step can be returned. Defaults to True.
    gtol : float
         Exposes gtol of l-bfgs-b's in scipy. Defaults to 1e-9.
    ftol : float
         Exposes ftol of l-bfgs-b's in scipy. Defaults to 1e-9.
    shape : str
         Shape of control pulse. 'square' and 'gaussian' forms are supported. Defaults to 'square'
    pulse_return: bool
         Returns the pulse object if set True. Defaults to False. 
    """        
    from ctrlq.lib.solve import pulsec

    # initial parameters
    ilist = []
    if method == 'l-bfgs-b':

       if shape=='gaussian':
           for i in range(self.pulse.nqubit):
               for j in range(self.pulse.ngaus):
                   ilist.extend([self.pulse.amp[i][j], self.pulse.mean[i][j],
                                 self.pulse.sigma[i][j]])
       elif shape=='square':                              
           for i in range(self.pulse.nqubit):
               for j in range(self.pulse.nwindow):
                    ilist.append(self.pulse.amp[i][j])
       elif  shape =='flattop': # exp.
           for i in range(self.pulse.nqubit):
               ilist.append(self.pulse.amp[i][0])
       else:
           sys.exit('Pulse shape not supported. Try square, gaussian or flattop')
                   
       for i in self.pulse.freq:
           ilist.append(i)

       if self.iprint > 1:
           print('  -----* Entering pulse optimizations *-----',flush=True)
           print('  Pulse parameters: Amplitude, time window (if any), frequency',flush=True)

       if gradient =='numerical':
           if shape=='square':
               E_ = self.efunc(self.ham.initial_state, self.pulse, self.ham, self.solver,
                               self.nstep, normalize, supdate=True, twindow=True)
           elif shape =='flattop':
               E_ = self.gaussquare_obj(ilist, self.pulse, self.ham, self.solver,
                               self.nstep, normalize, grad=False)
       else:
           if shape=='square':
               self.anatwin(ilist,self.pulse, self.ham, self.nstep,normalize)
           elif shape =='flattop':
               E_ = self.gaussquare_obj(ilist, self.pulse, self.ham, self.solver,
                               self.nstep, normalize, grad=False)
      
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
           E_ = self.efunc(self.ham.initial_state, pobjc__, self.ham, self.solver,
                           self.nstep, normalize, supdate=True, twindow=False, cobj=True)         

       self.print_step(ilist)

       if gradient=='numerical':

           if shape =='square':
               res1 = scipy.optimize.minimize(
                   self.objfunc, ilist, args = (self.pulse, self.ham, self.solver,
                                                self.nstep,normalize,False),
                   method=method,jac=True,
                   bounds = self.pulse.constraint,
                   callback = self.print_step,
                   options = {'maxiter':maxiter,'gtol':gtol,
                              'ftol':ftol,'maxls':maxls}) #, 'iprint':1,'disp':1})
               
           elif shape =='flattop':                   
               res1 = scipy.optimize.minimize(
                   self.gaussquare_obj, ilist, args = (self.pulse, self.ham, self.solver,
                                            self.nstep,normalize),
               method=method,jac=True,
               bounds = self.pulse.constraint,
               callback = self.print_step,
               options = {'maxiter':maxiter,'gtol':gtol,
                          'ftol':ftol,'maxls':maxls}) #, 'iprint':1,'disp':1})
               
       elif gradient == 'analytical':
           
           if shape=='square' or shape=='gaussian':

               if shape=='square':
                   objf__ = self.anatwin
               elif shape=='gaussian':
                   objf__ = self.anagaus
                   
               if optiter:
                   res1 = scipy.optimize.minimize(
                       objf__, ilist, args = (self.pulse, self.ham, self.nstep,normalize),
                       method=method,jac=True,
                       bounds = self.pulse.constraint,
                       callback = self.print_step,
                       options = {'maxiter':maxiter,'gtol':gtol,
                                  'ftol':ftol,'maxls':maxls}) #, 'iprint':1000,'disp':1000})
               else:
                   objf__(ilist,self.pulse, self.ham, self.nstep,normalize)
                
           else:                    
               res1 = scipy.optimize.minimize(
                   self.gradfunc, ilist, args = (self.pulse, self.ham, self.nstep),
                   method=method,jac=True,
                   bounds = self.pulse.constraint,
                   callback = self.print_step,
                   options = {'maxiter':maxiter,'gtol':gtol,
                              'ftol':ftol,'maxls':maxls}) #, 'iprint':1,'disp':1})

       else:
           sys.exit('Gradient method not implemented. Supports \'numerical\' and \'analytical\' only.')

       if self.iprint > 0:
           print(flush=True)
           print('  Pulse optimization ends',flush=True)
           if optiter:
               print(' ',res1.message)
           print('  ------------------------------------------',flush=True)
           print(flush=True)
           print('  Printing progress',flush=True)
           print('  --------------------------------------------------------',flush=True)
           print('  --------------------------------------------------------',flush=True)
           print('  Iter       Energy(H)      Leak(%)  Ediff(H)    Gnorm',flush=True)
           for i in range(len(self.elist_)):
               if i == 0:
                   idx__ = 0
               else:
                   idx__ = i - 1
               print('  {:>3d}  {:>18.12f}  {:>7.4f}  {:>.4e}  {:>.4e}'.format(
                   i,self.elist_[i],self.llist_[i]*100.,abs(self.elist_[i]-self.elist_[idx__]),
                   numpy.linalg.norm(self.glist_[i])))
           print('  --------------------------------------------------------',flush=True)
           print('  --------------------------------------------------------',flush=True)
           print(flush=True)
           self.elist_ = []
           self.llist_ = []
           self.glist_ = []
       
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
                           print(' {:>20.16f}  '.format(res1.x[cout]),end='',flush=True)
                           cout_ += 1
                           cout += 1
                       print(flush=True)
                   print(flush=True)
                   if not self.nstep == len(self.pulse.tseq[0])+1:
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
                   
               elif shape == 'gaussian':

                   amp__ = [[] for i in range(self.pulse.nqubit)]
                   mean__ = [[] for i in range(self.pulse.nqubit)]
                   sigma__ = [[] for i in range(self.pulse.nqubit)]
                   
                   for i in range(self.pulse.nqubit):
                       for j in range(self.pulse.ngaus):
                           amp__[i].append(res1.x[cout])
                           cout += 1
                           mean__[i].append(res1.x[cout])
                           cout += 1
                           sigma__[i].append(res1.x[cout])
                           cout += 1
                           
                   print('  | Amplitudes',flush=True)
                   for i in range(self.pulse.nqubit):
                       print('  Qubit ',i+1,' : ',end='',flush=True)
                       
                       cout_=0
                       for j in range(self.pulse.ngaus):
                           if cout_ ==4:
                               print(flush=True)
                               print('              ',end='',flush=True)
                               cout_ = 0
                           print(' {:>20.16f}  '.format(amp__[i][j]),end='',flush=True)
                           cout_ += 1
                       print(flush=True)
                   print(flush=True)
                           
                   print('  | Mean',flush=True)
                   for i in range(self.pulse.nqubit):
                       print('  Qubit ',i+1,' : ',end='',flush=True)
                       
                       cout_=0
                       for j in range(self.pulse.ngaus):
                           if cout_ ==4:
                               print(flush=True)
                               print('              ',end='',flush=True)
                               cout_ = 0
                           print(' {:>20.16f}  '.format(mean__[i][j]),end='',flush=True)
                           cout_ += 1
                       print(flush=True)
                   print(flush=True)
                           
                   print('  | Variance',flush=True)
                   for i in range(self.pulse.nqubit):
                       print('  Qubit ',i+1,' : ',end='',flush=True)
                       
                       cout_=0
                       for j in range(self.pulse.ngaus):
                           if cout_ ==4:
                               print(flush=True)
                               print('              ',end='',flush=True)
                               cout_ = 0
                           print(' {:>20.16f}  '.format(sigma__[i][j]),end='',flush=True)
                           cout_ += 1
                       print(flush=True)
                   print(flush=True)
                   
               qu = 1
               print('  | Frequencies',flush=True)
               for i in self.pulse.freq:
                   print('  Qubit ',qu,' : ',end='',flush=True)
                   qu += 1               
                   print(' {:>20.16f}  '.format(res1.x[cout]),flush=True)
                   cout += 1 
               print(flush=True)
               print('  Printing ends ',flush=True)
               print('  --------------------------------------',flush=True)
               print(flush=True)
               
       if pulse_return:
           return(self.pulse, self.energy_,self.leak)
       return (self.energy_,self.leak)
