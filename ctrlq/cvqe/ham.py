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

import scipy, numpy, itertools
import scipy.linalg
from .device import *
from .omisc import *
import sys

class transmon:
    """Transmon Hamiltonian class.
    Construct a transmon Hamiltonian.

    Parameters
    ----------
    nqubit : int
         Number of qubits. Defaults to 2.
    nstate : int
         Number of states. Defaults to 3.
    mham : numpy.ndarray
         Molecular Hamiltonian in the qubit representation.
    """   

    def __init__(self, nqubit=2, nstate=3, mham=None):

        # static                
        Hstatic = static_ham(nstate, nqubit=nqubit)
        basis_ = cbas_(nstate, nq = nqubit)
        Hamdbas = dresser(Hstatic, basis_)
        self.dsham = msdressed(Hamdbas, Hstatic)
        
        # time dependent
        Hdrive = t_ham(nstate, nqubit=nqubit)      
        self.hdrive = mtdressed(Hamdbas, Hdrive)
        self.nqubit = nqubit
        self.nstate = nstate
        self.mham = mham

        states = []
        for i in itertools.product([0,1], repeat = nqubit):
            sum_ = 0
            cout_ = 0
            for j in reversed(i):
                sum_ += j * nstate**cout_
                cout_ += 1
            states.append(sum_)
        self.states = states
        if nqubit == 2:
            istate = [0,1]
        elif nqubit == 4:
            istate = [0,0,1,1]
        else:
            sys.exit("Provide initial state using self.initialize_psi()")
        istate = initial_state(istate, nstate)
        self.initial_state = istate

    def initialize_psi (self, ket):
        """Initial state vector

        Define initial state. Note that transmon class initializes a state vector by default, \|0\>\|1\> for a two-qubit case and \|0\>\|0\>\|1\>\|1\> for a four-qubit case.

        Parameters
        ----------
        ket : list
          A list of ints with each elements defining the state. e.g. [0,1] for \|0\>\|1\>.

        """

        self.initial_state = initial_state(ket, self.nstate)

    

        
def getham(t, pobj, hobj):
    import functools
    from pulse import pcoef

    nqubit = len(pobj.amp)
    hamdr = 0.0
    for i in range(nqubit):
        hcoef = pcoef(t, amp=pobj.amp[i], tseq=pobj.tseq[i],
                      freq=pobj.freq[i], tfinal=pobj.duration)
        hcoefc = pcoef(t, amp=pobj.amp[i], tseq=pobj.tseq[i],
                       freq=pobj.freq[i], tfinal=pobj.duration,
                       conj=True)

        hamdr += hcoef * hobj.hdrive[i][0].toarray()
        hamdr += hcoefc * hobj.hdrive[i][1].toarray()

    dsham_diag = numpy.diagonal(-1j * hobj.dsham.toarray())
    dsham_diag = dsham_diag*t
    
    matexp_ = numpy.exp(dsham_diag)
    matexp_ = numpy.diag(matexp_)

    hamr_ = functools.reduce(numpy.dot, (matexp_.conj().T, hamdr, matexp_))
    #print(type(hamr_))
    return hamr_
        

def static_ham(nstate, nqubit = 2):

    diag_n = numpy.arange(nstate)
    diag_n = numpy.diagflat(diag_n)
    eye_n = numpy.eye(nstate, dtype=numpy.float64)
    diag_eye = 0.5 * numpy.dot(diag_n, diag_n - eye_n)
    astate = anih(nstate)
    cstate = create(nstate)

    dp = device()

    ham_ = 0.0
    iwork = True
    for i in range(nqubit):

        h_ = dp.w[i]*diag_n - dp.eta[i]*diag_eye

        if not i:
            tmp_ = h_
            tmp_i = astate
        else:
            tmp_ = eye_n
            if i == nqubit-1:
                tmp_i = cstate
            else:
                tmp_i = eye_n
        
        for j in range(1,nqubit):
            if j == i:
                wrk = h_
                wrk_i = astate
            elif j == i+1:
                wrk = eye_n
                wrk_i = cstate
            else:
                wrk = eye_n
                wrk_i = eye_n
                
            tmp_ = numpy.kron(tmp_,wrk)
            if iwork:
                tmp_i = numpy.kron(tmp_i,wrk_i)
                            
        ham_ += tmp_
        if iwork:
            tmp_i += tmp_i.conj().T
            tmp_i *= dp.g[i]
            ham_ += tmp_i
            
            if nqubit == 2:
                iwork = False
    
    return ham_

def t_ham(nstate, nqubit = 2):

    astate = anih(nstate)
    cstate = create(nstate)
    eye_n = numpy.eye(nstate, dtype=numpy.float64)

    hdrive = []
    for i in range(nqubit):

        if not i:
            tmp1 = cstate
            tmp2 = astate
        else:
            tmp1 = eye_n
            tmp2 = eye_n

        for j in range(1,nqubit):
            if j==i:
                wrk1 = cstate
                wrk2 = astate
            else:
                wrk1 = eye_n
                wrk2 = eye_n
                
            tmp1 = numpy.kron(tmp1, wrk1)
            tmp2 = numpy.kron(tmp2, wrk2)

        hdrive.append([tmp1,tmp2])

    return hdrive
             

def dresser(H_, basis_):

    evals, evecs = scipy.linalg.eigh(H_)

    evecs = evecs.T
    res = []
    for i in basis_:
        tmp_ = max(evecs, key=lambda x: numpy.abs(numpy.dot(x,i)))
        res.append(tmp_)
        #res.append(max(evecs, key=lambda x: numpy.abs(numpy.dot(x,i))))

    for i, part in enumerate(res):
        if max(part, key=abs) < 0:
            res[i] = -res[i]
        mask = numpy.abs(res[i]) < 1.e-15
        res[i][mask] = 0.0
        
    return numpy.array(res)
        
def msdressed(dbasis, h_):
    import functools
    
    h__ = functools.reduce(numpy.dot, (dbasis, h_, dbasis.conj().T))
    
    #dbasis = scipy.sparse.csc_matrix(dbasis,dtype=numpy.float64)
    #h_ = scipy.sparse.csc_matrix(h_,dtype = numpy.float64)
    #h__ = dbasis * h_ * dbasis.conj().T
    #
    #mask = numpy.abs(h__.data) < 1.0e-15
    #h__.data[mask] = 0.0e0
    #h__.eliminate_zeros()
    
    h__ = scipy.sparse.csc_matrix(h__,dtype = numpy.float64)
    mask = numpy.abs(h__.data) < 1.0e-15
    h__.data[mask] = 0.0e0
    h__.eliminate_zeros()
    
    return h__
  
def mtdressed(dbasis, h_):

    h__ = []
    for i in h_:
        
        h__.append([msdressed(dbasis, i[0]), msdressed(dbasis, i[1])])

    return h__
