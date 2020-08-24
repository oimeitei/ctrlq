Tutorial
========

It is very simple to use CtrlQ for simulating a state-preparation on a Trasmon
qubit device using analog pulses. This tutorial shows how to use CtrlQ and
only consist of few steps::

  from ctrlq import cvqe

  # define a pulse using cvqe.pulse()
  mypulse = cvqe.pulse()

  # construct a device Hamiltonian using cvqe.transmon()
  myham = cvqe.transmon(mham = molecule_hamiltonian) # see below for molecule_hamiltonian

  # initialize control
  myctrl = cvqe.control(mypulse, myham)

  # perform a pulse optimization
  myctrl.optimize()


Running CtrlQ consist of the following steps:

  1. Construct molecular Hamiltonian in the qubit representation.
  2. Define a pulse shape.
  3. Construct a device Hamiltonian
  4. Run ctrl-VQE.

Molecular Hamiltonian
---------------------

Molecular Hamiltonian in the qubit representation can simply be supplied into
``cvqe.transmon()`` as a ``numpy.ndarray``::

  myham = cvqe.transmon(mham = MyMolecularHamilonian)

`Qiskit <https://qiskit.org/>`_ package can be used to construct the molecular
Hamiltonian. See the documentations for the package. If you have qiskit
package installed, the molecular Hamiltonian for H :sub:`2`, HeH :sup:`+` and
LiH molecular systems can be obtained from ``mole.molecule``::

  from mole import molecule

  # set the bond distance in angstom using dist
  # for H2
  MyMolecularHamiltonian = molecule.h2(dist=0.75)

  # for HeH+
  MyMolecularHamiltonian = molecule.hehp(dist=1.0)

  # for LiH
  MyMolecularHamiltonian = molecule.lih(dist=1.5)

The molecular Hamiltonian of H :sub:`2` and HeH :sup:`+` is mapped to
two-qubits and LiH to four-qubits using the parity mapping and Z2Symmetries
reduction. 
  

Pulse shape
-----------
Pulse shapes can be defined using the ``cvqe.pulse`` class. Simply initialize
as in the example above ``mypulse = cvqe.pulse()``. The amplitudes, time
segments and the frequencies are all random guess. To set the amplitudes,
simply supply a list with the shape ``(No. of qubits, No. of time windows)``. To
set the time segments, supply a list with shape ``(No. of qubits, No. of time
windows - 1)``. And to set the frequencies, supply a list with shape ``(No. of
qubits)``. As an example for a two-qubit case with four-time seqment square
pulses::

  from ctrlq import cvqe

  # duration is the total pulse duration
  mypulse = cvqe.pulse(nqubit = 2, nwindow = 4, duration = 10.0)

  # set amplitudes
  mypulse.amp = [[0.05, 0.04, 0.03, 0.02], # for qubit 1
                 [0.03, 0.01, 0.05, 0.06]] # for qubit 2

  # set time window		 
  mypulse.tseq = [[2.3, 5.6, 6.1], # for qubit 1
                  [1.0, 3.4, 5.6]] # for qubit 2

  # set frequencies
  mypulse.freq = [ 29.0, 31.0] # for two-qubits

  
Device Hamiltonian
------------------
The device Hamiltonian can be easily set using ``cvqe.transmon()`` as shown in
the above example. Here, the molecular Hamiltonian ``mham`` is required as a
``numpy.ndarray``. Furthermore, initial state other the default set by
``cvqe.transmon()`` can also be set::

  from ctrq import cvqe
  from mole.molecule import h2

  # molecular Hamiltonian
  cham = h2()

  # pulse
  mypulse = cvqe.pulse()
  
  # construct a device Hamiltonian with 2-qubits and 3-levels
  myham = cvqe.transmon(nqubit=2, nstate=3, mham=cham)

  # use a non-default initial state of |1> |1>
  myham.initialize_psi([1,1])

  
Run ctrl-VQE
------------------
To perform ctrl-VQE pulse optimization, use the ``cvqe.control()`` class. Set
the time-evolution solver, either ``solver='ode'`` or ``solver='trotter'`` and
also the time-step ``nstep=``. The optimization can be then called using
``cvqe.control.optimize()``. Also specify whether to use the normalize
expectation value or an unnormalized one using ``normalize=True`` or
``normalize=False``. To just achieve a state-preparation using a pulse and
compute the expectation value of the molecular Hamiltonain, call
``cvqe.control.energy()``. Here, again specify whether or not to return the
normalize expectation value.

Now, we can run ctrl-VQE using a custom pulse shape::

    from ctrlq import cvqe
    from mole.molecule import h2

    # molecular Hamiltonian
    cham = h2()

    # duration is the total pulse duration
    mypulse = cvqe.pulse(nqubit = 2, nwindow = 4, duration = 10.0)

    # set amplitudes
    mypulse.amp = [[0.05, 0.04, 0.03, 0.02], # for qubit 1
                   [0.03, 0.01, 0.05, 0.06]] # for qubit 2

    # set time window		 
    mypulse.tseq = [[2.3, 5.6, 6.1], # for qubit 1
                    [1.0, 3.4, 5.6]] # for qubit 2

    # set frequencies
    mypulse.freq = [ 29.0, 31.0] # for two-qubits
  
    # construct a device Hamiltonian with 2-qubits and 3-levels
    myham = cvqe.transmon(nqubit=2, nstate=3, mham=cham)

    # use a non-default initial state of |1> |1>
    myham.initialize_psi([1,1])

    # control class
    myctrl = cvqe.control(mypulse, myham, nstep=1000, solver='trotter')

    # measure the expectation value using mypulse
    Energy, leakage = myctrl.energy(normalize=True)

    # perform variational pulse optimization
    Energy, leakage = myctrl.optimize(normalize=True)

Check ``/ctrlq/example`` for more examples.
    
    
