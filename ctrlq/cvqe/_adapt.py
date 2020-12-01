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

import numpy,sys,scipy

def objfun(x, x1, x2, g, tlist):

    glist = []
    tlist1 = []
    gtot1 = 0.0
    gtot2 = 0.0
    
    for i in range(len(tlist)):
        if x1<tlist[i]<=x2:
            glist.append(g[i])
            tlist1.append(tlist[i])
            
    for i in range(len(tlist1)):
        if tlist1[i] <= x:
            gtot1 += glist[i]
        else:
            gtot2 += glist[i]

    gtot = gtot1 * gtot1 + gtot2 * gtot2
    return -gtot

def adapt(self, gradient='analytical', maxiter=200, shape='square',
          normalize=True, maxls=20, maxwindow=20,
          twindow=False, random=True, exactE = 0, thres=1e-4):
    """control class method:
    Perform ctrl-VQE pulse optimization by adaptively increasing the
    number of pulse parameterization one at a time to determine an 
    optimal number for a target accuracy. Only supported for 'square'
    pulse shape.

    Parameters
    ----------
    gradient : str
             Method for computing the gradients in the optimization. 'numerical' for numerical gradients for all pulse parameters. 'analytical' for an analytical gradients for the amplitudes and numerical gradients for the frequencies. Defaults to 'analytical'.
    maxwindow : int
             Maximum number of pulse window (time segments) allowed. Defaults to 20.
    exactE : float
             Provide exact energy to print the error in each adaptive step. Set iprint to < 2 if not provided.
    thres : float
             Convergence threshold to terminate adaptive update. thres is the difference to the energy from a previous adaptive step. Defaults to 0.0001.
    random : bool
             Whether to use random slicing of square pulse or use the duration where gradients are maximum on each halves. Defaults to True.

    """    
    from ctrlq import cvqe

    iter_ = 1
    if self.iprint > 1:
        print(flush=True)
        print('    *----------------------------------------*',flush=True)
        print('    | Adaptive construction of square pulses |',flush=True)
        print('    *----------------------------------------*',flush=True)
        print(flush=True)    
        print('  ----* Adaptive iter ',iter_,' starts *-----',flush=True)
        print('  No of time segments : ',self.pulse.nwindow,flush=True)
        print(flush=True)
    if not random:
        tlist = numpy.linspace(0., self.pulse.duration, self.nstep)
    # First iter
    mypulse, energy, leak =  self.optimize(gradient=gradient, maxiter=maxiter, shape=shape,
                  pulse_return=True, normalize=normalize,
                  maxls=maxls)

    prevE = energy

    if self.iprint > 2:
        print(flush=True)
        print('  Adaptive iter ',iter_,' ends ',flush=True)
        if exactE:
            print('  Error in ctrl-VQE energy : {:>.4e}'.format(energy-exactE),flush=True)
        print('  -------------------------------------',flush=True)
        print(flush=True)
    # Remainder iter
    for nwin in range(self.pulse.nwindow+1, maxwindow):
        iter_ += 1

        if self.iprint > 1:
            print(flush=True)
            print('  ----* Adaptive iter ',iter_,' starts *-----',flush=True)
            print('  No of time segments : ',nwin,flush=True)
            if random:
                print('  Following random pulse slicing',flush=True)
            else:
                print('  Following gradient rule to slice pulse:',flush=True)
                print('       Max{f\'(window-I)^2 + f\'(window-II)^2}',flush=True)
            print(flush=True)
            
        # Slice pulse
        for i in range(self.pulse.nqubit):
            
            if self.pulse.nwindow == 1:

                if random:
                    
                    self.pulse.tseq[i].insert(0,numpy.random.uniform(0.0,
                                              self.pulse.duration))
                else:
                    if gradient == 'analytical':
                        res = scipy.optimize.brute(
                            objfun, (slice(0.0, self.pulse.duration,0.25),),
                            args=(0.0, self.pulse.duration,
                                  self.square_gamp[i], tlist))
                        self.pulse.tseq[i].insert(0, res[0])                 
                    else:
                        sys.exit('Using gradients to slice pulse only works'\
                                 'with analytical gradients')
                if not twindow:
                    self.square_amp[i].insert(-1,self.square_amp[i][0])
                else:
                    self.pulse.amp[i].insert(-1,self.pulse.amp[i][0])
            else:
                
                # Use window with max duration
                tseq__ = []
                for j in range(self.pulse.nwindow-1):
                    if j==0:
                        tseq__.append(self.pulse.tseq[i][0])
                        if j==self.pulse.nwindow-2:
                            tseq__.append(self.pulse.duration -
                                          self.pulse.tseq[i][j]) 
                
                    elif j==self.pulse.nwindow-2:
                        tseq__.append(self.pulse.tseq[i][j] -
                                      self.pulse.tseq[i][j-1])
                        tseq__.append(self.pulse.duration -
                                      self.pulse.tseq[i][j]) 
                    else:
                        tseq__.append(self.pulse.tseq[i][j] -
                                      self.pulse.tseq[i][j-1])
                
                ind = tseq__.index(max(tseq__))
                
                # Slice pulse
                if ind == 0:
                    if random:                        
                        self.pulse.tseq[i].insert(0, numpy.random.uniform(
                            0.0, self.pulse.tseq[i][0]))
                    else:
                        res = scipy.optimize.brute(
                            objfun, (slice(0.0, self.pulse.tseq[i][0],0.25),),
                            args=(0.0,self.pulse.tseq[i][0],
                                  self.square_gamp[i],tlist))
                        self.pulse.tseq[i].insert(0, res[0])
                        
                    if not twindow:
                        self.square_amp[i].insert(0,self.square_amp[i][0])
                    else:
                        self.pulse.amp[i].insert(0,self.pulse.amp[i][0])
                        
                elif ind == len(tseq__)-1:
                    if random:
                        self.pulse.tseq[i].insert(ind, numpy.random.uniform(
                            self.pulse.tseq[i][-1], self.pulse.duration))
                    else:                    
                        res = scipy.optimize.brute(
                            objfun, (slice(self.pulse.tseq[i][-1],
                                           self.pulse.duration,0.25),),
                            args=(self.pulse.tseq[i][-1], self.pulse.duration,
                                  self.square_gamp[i], tlist))
                        self.pulse.tseq[i].insert(ind, res[0])
                        
                    if not twindow:
                        self.square_amp[i].insert(-1,self.square_amp[i][-1])
                    else:
                        self.pulse.amp[i].insert(-1,self.pulse.amp[i][-1])
                        
                else:
                    if random:
                        self.pulse.tseq[i].insert(ind,numpy.random.uniform(
                            self.pulse.tseq[i][ind-1], self.pulse.tseq[i][ind]))
                    else:
                        res = scipy.optimize.brute(
                            objfun, (slice(self.pulse.tseq[i][ind-1],
                                           self.pulse.tseq[i][ind],0.25),),
                            args=(self.pulse.tseq[i][ind-1],
                                  self.pulse.tseq[i][ind],
                                  self.square_gamp[i], tlist))
                        self.pulse.tseq[i].insert(ind,res[0])

                    if not twindow:
                        self.square_amp[i].insert(ind, self.square_amp[i][ind])
                    else:
                        self.pulse.amp[i].insert(ind, self.square_amp[i][ind])                        
                # do we need to sort self.pulse.tseq ?
                
        if not twindow:
            self.pulse.amp = self.square_amp
        self.pulse.nwindow = nwin

        # Add constrains    
        cons_ = []
        for i in range(self.pulse.nqubit):
            for j in range(self.pulse.nwindow):
                cons_.append((-numpy.pi*2*self.pulse.amp_bound,
                              numpy.pi*2*self.pulse.amp_bound))
        
        dp = cvqe.device.device()
        for i in range(self.pulse.nqubit):
            cons_.append(((dp.w[i]-self.pulse.freq_bound)*self.pulse.fscale,
                          (dp.w[i]+self.pulse.freq_bound)*self.pulse.fscale))

        self.pulse.constraint = cons_

        # Optimize pulse
        mypulse, energy, leak = self.optimize(
            gradient=gradient, maxiter=maxiter, shape=shape,
            pulse_return=True, normalize=normalize,
            maxls=maxls)
        Ediff = abs(prevE - energy)

        if self.iprint > 2:
            print(flush=True)
            print('  Adaptive iter ',iter_,' ends ',flush=True)
            if exactE:
                print('  Error in ctrl-VQE energy : {:>.4e}'.format(energy-exactE),flush=True)

        if self.iprint > 1:
            print(flush=True)
            print('  Adaptive energy change   : {:>.4e}'.format(Ediff),flush=True)
            print('  -------------------------------------',flush=True)
            print(flush=True)
            
        if not iter_ == 1 and Ediff < thres:

            if self.iprint > 1:
                print('  Adaptive energy change < {:>.4e} - call it CONVERGED!'.format(thres),
                      flush=True)
                print('  Optimal pulse windows : ',nwin-1)
                print(flush=True)
            break
        prevE = energy                
