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

import numpy,scipy.sparse
from math import *
import warnings
warnings.filterwarnings('ignore')



def cbas_(n, nq = 2):
    
    bas_ = numpy.eye(n**nq, dtype=numpy.float64)
    return bas_

def qbas_(nstate, idx):

    bas_ = numpy.zeros((nstate,1),dtype=numpy.complex128)
    bas_[idx] = 1.0
    return bas_

def anih(n):
    a_ = numpy.zeros((n,n), dtype=numpy.float64)
    for i in range(n-1):
        a_[i,i+1] = numpy.sqrt(i+1)

    return a_

def create(n):
    c_ = numpy.zeros((n,n),dtype=numpy.float64 )
    for i in range(1,n):
        c_[i,i-1] = numpy.sqrt(i)

    return c_

def initial_state(list1, nstate=2):

    tmp_ = qbas_(nstate,list1[0])
    for idx in list1[1:]:
        wrk_ = qbas_(nstate,idx)
        tmp_ = numpy.kron(tmp_, wrk_)

    return tmp_

