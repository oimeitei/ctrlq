.. CtrlQ documentation master file, created by
   sphinx-quickstart on Sun Aug 23 14:31:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


*******************
CtrlQ documentation
*******************
CtrlQ is an open-source tool designed to simulate a gate-free state
preparation on a Transmon qubit device using analog control pulses. The analog
control pulses can be variationally shaped to drive an initial state to a
target state in the framework of ctrl-VQE. In molecular systems, ctrl-VQE can
be used to drive the initial Hartree Fock state to the full configuration
interaction (FCI) state with substantial pulse duration speedups as compared
to a gate-based compilation.

The control quantum program (CtrlQ) is written in python with bindings to C++
codes for highly efficient time-evolution of quantum systems either using an
ordinary-differential-equation or the Suzuki-Trotter expansion.

**Please cite:**

OR Meitei, BT Gard, GS Barron, DP Pappas, SE Economou, E Barnes, NJ Mayhall,
Gate-free state preparation for fast variational quantum eigensolver
simulations: ctrl-VQE, `arXiv:2008.04302 <https://arxiv.org/abs/2008.04302>`_

.. toctree::
   :maxdepth: 4

   install
   tutorial
   pulse
   ham
   kernel
