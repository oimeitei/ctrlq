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


class device:
    
    def __init__(self):
        pi2 = 2 * numpy.pi
        self.w =[ pi2 * 4.808049015463495,
                  pi2 * 4.833254817254613,
                  pi2 * 4.940051121317842,
                  pi2 * 4.795960998582043]
        self.g = [pi2 *0.018312874435769682,
                  pi2 *0.021312874435769682,
                  pi2 *0.019312874435769682,
                  pi2 *0.020312874435769682]
        self.eta = [pi2* 0.3101773613134229,
                    pi2* 0.2916170385725456,
                    pi2* 0.3301773613134229,
                    pi2* 0.2616170385725456]

class device4:

    def __init__(self):
        pi2 = 2 * numpy.pi
        self.w =[ pi2 * 4.808049015463495,
                  pi2 * 4.833254817254613,
                  pi2 * 4.940051121317842,
                  pi2 * 4.795960998582043]
        self.eta = [pi2* 0.3101773613134229,
                    pi2* 0.2916170385725456,
                    pi2* 0.3301773613134229,
                    pi2* 0.2616170385725456]
        self.g = [[0.0, pi2 * 0.018312874435769682, pi2 * 0.019312874435769682,
                   pi2 * 0.020312874435769682 ],
                  [pi2 * 0.018312874435769682, 0.0, pi2 * 0.021312874435769682,
                   pi2 * 0.018312874435769682],
                  [pi2 * 0.019312874435769682, pi2 * 0.021312874435769682, 0.0,
                   pi2 * 0.019312874435769682],
                  [pi2 * 0.020312874435769682, pi2 * 0.018312874435769682,
                   pi2 * 0.019312874435769682, 0.0]]

class device4_ldetune:

    def __init__(self):
        pi2 = 2 * numpy.pi
        self.w =[ pi2 * 5.327178905327263,
                  pi2 * 5.076070923892718,
                  pi2 * 5.327178905327263,
                  pi2 * 5.076070923892718]
        self.eta = [pi2* 0.3101773613134229,
                    pi2* 0.2916170385725456,
                    pi2* 0.3301773613134229,
                    pi2* 0.2616170385725456]
        self.g = [[0.0, pi2 * 0.018312874435769682, pi2 * 0.019312874435769682,
                   pi2 * 0.020312874435769682 ],
                  [pi2 * 0.018312874435769682, 0.0, pi2 * 0.021312874435769682,
                   pi2 * 0.018312874435769682],
                  [pi2 * 0.019312874435769682, pi2 * 0.021312874435769682, 0.0,
                   pi2 * 0.019312874435769682],
                  [pi2 * 0.020312874435769682, pi2 * 0.018312874435769682,
                   pi2 * 0.019312874435769682, 0.0]]
