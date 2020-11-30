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

def evolve(ini_vec, pobj, hobj, solver='ode', nstep=2000, twindow=True):
    import scipy.integrate
    import numpy, scipy, sys
    from ctrlq.lib.solve import solve_func
    from ctrlq.lib.trotter import solve_trotter, solve_trotter2
    from ctrlq.lib.agradc import grad_trotter

    dsham = numpy.diagonal(-1j * hobj.dsham.toarray())    
    tlist = numpy.linspace(0, pobj.duration, nstep)
    
    if solver == 'ode':
        r = scipy.integrate.ode(solve_func)
        r.set_integrator('zvode',method='adams',
                 with_jacobian=False,
                 rtol=1e-10, atol=1e-12,
                 lband=None, uband=None,
                 order=12,
                 nsteps=500) # There are parameters to set here.
                
        r.set_f_params(pobj, numpy.array(hobj.hdrive), dsham)
        r.set_initial_value(ini_vec, 0.0)

        for t in tlist[1:]:
            tmp_ = r.integrate(t)
            r.set_initial_value(tmp_, t)
        
        return tmp_
    
    elif solver == 'trotter':
        
        if not twindow:
            tmp_ = solve_trotter2(tlist, ini_vec, pobj, numpy.array(hobj.hdrive), dsham)
        else:        
            tmp_ = solve_trotter(tlist, ini_vec, pobj, numpy.array(hobj.hdrive), dsham)

        return tmp_

    else:

        sys.exit('Solver doesn\'t exits or not implemented')

