import numpy
from .device import device
#todo add gaussian pulse
from math import *

def pcoef(t, amp=None, tseq=None, freq=None, tfinal = 0.0,
          conj=False, scale=1.):

    sign_ = -1.0 if not conj else 1.0
    plist = []
    for i,j in enumerate(tseq):
        
        if not i:
            plist.append(0.0 < t <= j)
        else:
            plist.append(tseq[i-1] < t <= j)
            
    plist.append(tseq[-1] < t <= tfinal)
    
    coeff = numpy.piecewise(t+0j, plist,
                            [i * numpy.exp(sign_ * scale * 1j * freq * t)
                             for i in amp])
    return coeff    


class pulse:
    """Analog pulse class.
    Define control analog pulse shape. Square pulses are initialized with random guess.

    Parameters
    ----------
    shape : str
        Shape of control pulse. Defaults to 'square' which is the only one currently supported.
    nqubit : int
        Number of qubits. Defaults to 2.
    nwindow : int
        For square pulses, number of time windows (seqments). Defaults to 2.
    duration : float
        Total time duration of the pulse. Defaults to 10.0.
    fscale : float
        Scale the frequency of the pulse. Defaults to 1.0.
    tscale : float 
        Scale the time windows and the total time duration of the pulse. Defaults to 1.0.
    """    

    def __init__(self, shape='square', nqubit=2, nwindow=2,
                 amp_constraint = None, tseq_constraint = None,
                 freq_constraint = None, fscale=1.0, tscale = 1.0, duration = 10.0):
        if shape == 'square':
            
            constraint = []
            amp= []
            tseq = []
            freq = []
            if not amp_constraint:
                amp_constraint = []
                for i in range(nqubit):
                    amp_ = []
                    for j in range(nwindow):
                        amp_constraint.append((-numpy.pi*2*0.02,
                                               numpy.pi*2*0.02))
                        amp_.append(numpy.random.uniform(-numpy.pi*2*0.02,
                                                         numpy.pi*2*0.02))
                    amp.append(amp_)
                    
                constraint.extend(amp_constraint)
                
            elif isinstance(amp_constraint, float):
                tmp_ = []
                amp = []
                for i in range(nqubit):
                    amp_ = []
                    for j in range(nwindow):
                        tmp_.append((-amp_constraint, amp_constraint))
                        amp_.append(numpy.random.uniform(-amp_constraint,
                                                         amp_constraint))
                    amp.append(amp_)
                    
                constraint.extend(tmp_)

            else:
                constraint.extend(amp_constraint)
                
            
            if not tseq_constraint:
                tseq_constraint = []
                for i in range(nqubit):
                    tseq_ = []
                    for j in range(nwindow-1):
                        tseq_constraint.append((0.0, duration*tscale))
                        tseq_.append(numpy.random.uniform(0.0, duration*tscale))

                    tseq.append(sorted(tseq_))
            #constraint.extend(tseq_constraint)
            
            if not freq_constraint:
                dp = device()
                freq_constraint = []
                for i in range(nqubit):
                    freq_constraint.append(((dp.w[i]-1.)*fscale,
                                            (dp.w[i]+1.)*fscale))

                    #tmp__ = numpy.random.uniform((dp.w[i]-1.)*0.01,
                    #                             (dp.w[i]+1.)*0.01)
                    #freq.append(tmp__*100.0)
                    freq.append(numpy.random.uniform((dp.w[i]-1.)*fscale,
                                                     (dp.w[i]+1.)*fscale))
                                
            constraint.extend(freq_constraint)
            self.constraint = constraint
            self.amp = amp
            self.tseq = tseq
            self.freq = freq
            self.duration = duration
            self.nqubit = nqubit
            self.nwindow = nwindow
            self.fscale = fscale
            self.tscale = tscale
        else:
            sys.exit('Pulse shape not yet implemented')
            
