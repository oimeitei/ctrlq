Kernel for ctrl-VQE
*******************
Run ctrl-VQE methods by instantiating :meth:`cvqe.kernel.control()`
class, takes :meth:`cvqe.pulse()` and :meth:`cvqe.transmon()` as positional
arguments. 

The following methods are implemented in CtrlQ:

Pulse optimization
==================
Perform ctrl-VQE variational pulse optimization with
:meth:`cvqe.kernel.control.optimize` method. The objective function is the
molecular energy. Square and gaussian pulse shapes are currently
supported. Efficient analytical gradients are also implemented for the pulse
parameters.

Measure energy
==============
Get the molecular energy from the state prepared with control pulses using
:meth:`cvqe.kernel.control.energy`. Supply the optimal pulse shape using 
:meth:`cvqe.pulse`::
	mypulse = cvqe.pulse(shape='square',duration=10.0)
	mypulse.amp = [[-0.02,-0.03],[0.02,0.03]] # assumed two-qubits
	mypulse.tseq = [[5.0],[5.0]] # two windows
	mypulse.freq = [30.0,30.1]
	
	myctrl = cvqe.control(mypulse, myham) # check Transmon Hamiltonian for myham
	energy, leak = myctrl.energy()

Get the initial energy (Hartree Fock) using :meth:`cvqe.kernel.control.HF`

Adaptive parameterization
=========================
With adaptive parameterization, grow the number of pulse parameters to
optimize with ctrl-VQE. Starting from twindow time segment sqaure pulse,
adaptively increase the number of pulse windows one at a time. This scheme
ultimately avoids over parameterization and determines the neccesary number of
pulse windows. 

Perform the adaptive pulse update using :meth:`cvqe.kernel.control.adapt`. Works only with square pulses with time segments aka windows. Usually start
with one pulse window and grow the number of windows.

.. toctree::
   :maxdepth: 4
.. automodule:: ctrlq.cvqe.kernel
    :members:
