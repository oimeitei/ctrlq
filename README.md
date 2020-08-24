CtrlQ
=====
CtrlQ is an open-source tool designed to simulate a gate-free state preparation on a Transmon qubit device using analog control pulses. The analog control pulses can be variationally shaped to drive an initial state to a target state in the framework of ctrl-VQE. In molecular systems, ctrl-VQE can be used to drive the initial Hartree Fock state to the full configuration interaction (FCI) state with substantial pulse duration speedups as compared to a gate-based compilation. 

The control quantum program (CtrlQ) is written in python with bindings to C++ codes for highly efficient time-evolution of quantum systems either using an ordinary-differential-equation or the Suzuki-Trotter expansion.

**Reference**
OR Meitei, BT Gard, GS Barron, DP Pappas, SE Economou, E Barnes, NJ Mayhall, Gate-free state preparation for fast variational quantum eigensolver simulations: ctrl-VQE
[arXiv:2008.04302](https://arxiv.org/abs/2008.04302)

## Installation
Detailed information for installation is provided in the documentation and consist of the following simple steps to build CtrlQ

1. Get the source code from github:

       git clone --recursive https://github.com/oimeitei/ctrlq.git

2. Configure with cmake and compile

       cd ctrl
       mkdir build && cd build
       cmake ..
       make
      
3. Run test

       python -m unittest discover


## Documentation
Documentation is available online in html format [here](https://ctrlq.readthedocs.io)
and also can be found in the ``/ctrlq/doc`` directory. The documentation
contains detailed installation instruction including linking to optimized MKL
libraries and tutorials to run CtrlQ.




