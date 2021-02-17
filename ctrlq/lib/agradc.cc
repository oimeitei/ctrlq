/* Copyright 2020 Oinam Romesh Meitei

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <vector>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "getham.h"
#include "agradc.h"
#include "grad_ana.h"

namespace py = pybind11;

/* Analytical gradients for pulse amplitude at every trotter step */

gradc grad_trotter(
                               std::vector<double> &tlist,
                               std::vector<std::complex<double> > &ini_vec,
                               pulsec pobj,
                               std::vector< std::vector< Eigen::SparseMatrix
			       <double,0,ptrdiff_t> > > hdrive,
                               std::vector< std::complex<double> > dsham,
			       std::vector< int> &states,
			       Eigen::MatrixXcd &cHam){

  gradc gradd= grad_ana(tlist, ini_vec, pobj, hdrive, dsham, states, cHam);

  return gradd;
}

gradc grad_trotter_normalized(
                               std::vector<double> &tlist,
                               std::vector<std::complex<double> > &ini_vec,
                               pulsec pobj,
                               std::vector< std::vector< Eigen::SparseMatrix
			       <double,0,ptrdiff_t> > > hdrive,
                               std::vector< std::complex<double> > dsham,
			       std::vector< int> &states,
			       Eigen::MatrixXcd &cHam){

  gradc gradd= grad_ana_normalized(tlist, ini_vec, pobj, hdrive, dsham, states, cHam);

  return gradd;
}
PYBIND11_MODULE(agradc,m){
  m.def("grad_trotter", &grad_trotter, "gtrotter");
  m.def("grad_trotter_normalized", &grad_trotter_normalized, "gtrotterN");
  py::class_<gradc>(m,"gradc")
    .def(py::init<
	 double &, double &,
	 std::vector<std::vector< double > > & > ())
    .def_readonly("energy", &gradc::energy)
    .def_readonly("norm", &gradc::norm)
    .def_readonly("gradient",&gradc::gradient);
}
