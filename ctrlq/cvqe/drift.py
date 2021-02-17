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
from .device import *
from .omisc import *
import numpy, sys

# 0 -- 1
# | \/ |
# | /\ |
# 3 -- 2

def transmon4_static(nstate=3, param = device4(), nqubit = 4,
                interact= ['01','12','23','03']):

    dp = param
    diag_n = numpy.arange(nstate)
    diag_n = numpy.diagflat(diag_n)
    eye_n = numpy.eye(nstate, dtype=numpy.float64)
    diag_eye = 0.5 * numpy.dot(diag_n, diag_n - eye_n)
    astate = anih(nstate)
    cstate = create(nstate)
    
    ham_ = 0.0
    iwork = True
    for i in range(nqubit):

        h_ = dp.w[i]*diag_n - dp.eta[i]*diag_eye

        if not i:
            tmp_ = h_
        else:
            tmp_ = eye_n
        
        for j in range(1,nqubit):
            if j == i:
                wrk = h_
            elif j == i+1:
                wrk = eye_n
            else:
                wrk = eye_n                
            tmp_ = numpy.kron(tmp_,wrk)
            
        ham_ += tmp_

    if '01' in interact or '10' in interact:
        i01 = kron(astate, cstate, eye_n, eye_n)
        i01 += i01.conj().T
        i01 *= dp.g[0][1]
        ham_ += i01

    if '02' in interact or '20' in interact:
        i02 = kron(astate, eye_n, cstate, eye_n)
        i02 += i02.conj().T
        i02 *= dp.g[0][2]
        ham_ += i02

    if '03' in interact or '30' in interact:
        i03 = kron(astate, eye_n, eye_n, cstate)
        i03 += i03.conj().T
        i03 *= dp.g[0][3]
        ham_ += i03
        
    if '12' in interact or '21' in interact:
        i12 = kron(eye_n, astate, cstate, eye_n)
        i12 += i12.conj().T
        i12 *= dp.g[1][2]
        ham_ += i12
        
    if '13' in interact or '31' in interact:
        i13 = kron(eye_n, astate, eye_n, cstate)
        i13 += i13.conj().T
        i13 *= dp.g[1][3]
        ham_ += i13

    if '23' in interact or '32' in interact:
        i23 = kron(eye_n, eye_n, astate, cstate)
        i23 += i23.conj().T
        i23 *= dp.g[2][3]
        ham_ += i23
    
    return ham_
    
def kron(*args):
    tmp_ = args[0]
    for i in args[1:]:
        tmp_ = numpy.kron(tmp_, i)
        
    return tmp_
