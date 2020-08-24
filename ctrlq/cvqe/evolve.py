import numpy, scipy
#from solve import getham, solve_func, solve_trotter #, evolve_odeint
from ctrlq.lib.solve import solve_func
from ctrlq.lib.trotter import solve_trotter
import sys


def evolve(ini_vec, pobj, hobj, solver='ode', nstep=2000):
    import scipy.integrate

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
        
        tmp_ = solve_trotter(tlist, ini_vec, pobj, numpy.array(hobj.hdrive), dsham)
      
        return tmp_

    else:

        sys.exit('Solver doesn\'t exits or not implemented')

