Installation
============

Obtaining the Source Code
-------------------------
Get the source code form github::

  git clone https://github.com/oimeitei/ctrlq.git


Requirements
------------

To build CtrlQ from source the following dependencies are required.

   - CMake 4.0 or higher
   - GCC 4.8 or newer (Intel C++ 16.0 or newer)
   - Python 3.5 or higher
   - Numpy 1.8 or newer
   - Scipy 1.4 or newer
   - Eigen library
   - Pybind11
   - MKL library (optional)
   - Qiskit (optional)

Note: Pybind11 can be simply obtained with CtrlQ when you clone
with --recursive option from github. The C++ Eigen library does not need to be
compiled as it is only header. Linking to MKL BLAS libraries speeds up the
code significantly.

Qiskit is only required to construct the molecular Hamiltonian in the qubit
representation. The molecular Hamiltonian can be simply supplied as a ``numpy.ndarray``.

Building CtrQ
-------------

The main steps in configuring and compiling CtrlQ involves the following
simple steps::

  cd {top-level-dir}
  mkdir build
  cd build
  cmake [options] ..
  make

After compilation, add the top-level-dir path to ``PYTHONPATH`` to make python
find CtrlQ program simply by::

  export PYTHONPATH=/path/to/ctrlq:$PYTHONPATH
  
Make sure that EIGEN_INCLUDE is available as an environment variable ::
  
  export EIGEN_INCLUDE=/path/to/include/eigen
  
or set it as a cmake option::
  
  cmake -DEIGEN_INCLUDE=/path/to/include/eigen ..

Linking to MKL library
^^^^^^^^^^^^^^^^^^^^^^

To link the program to MKL library, configure with::

  cmake -DMKL=ON ..

Make sure that cmake finds the correct MKL libraries. One of the following can
help cmake find the correct MKL libraries::

  source /path/to/intel/mkl/bin/mklvars.sh intel64

or::

  export LD_LIBRARY_PATH=/path/to/intel/mkl/lib/intel64:$LD_LIBRARY_PATH

Even after a successful compilation, there may be error at runtime if the
program is not linked to all the MKL libraries required, for example::

  MKL FATAL ERROR: Cannot load libmkl_avx512.so or libmkl_def.so.

In such cases, repeat the configuration with::

  cmake -DMKL=ON -DMKL_LIBS=$MKLROOT/lib/intel64/libmkl_avx512.so;$MKLROOT/lib/intel64/libmkl_def.so;$MKLROOT/lib/intel64/libmkl_core.so ..

and compile again. Another hacky solution is to preload the libraries::

  export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_avx512.so;$MKLROOT/lib/intel64/libmkl_def.so;$MKLROOT/lib/intel64/libmkl_core.so


Running test
------------

To make sure that CtrlQ has been build correctly as intended it is important
to run the test set included. This can be simply run at the
``{top-level-dir}`` by::

  python -m unittest discover

