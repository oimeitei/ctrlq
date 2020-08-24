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
#include <cmath>
#include <vector>
#include "getham.h"

namespace py = pybind11;



Eigen::MatrixXcd solve_func(double &t, std::vector<std::complex<double>> &y,
			    pulsec &pobj,
			    std::vector< std::vector< Eigen::SparseMatrix<double,0,ptrdiff_t>>> &hdrive,
			    std::vector< std::complex<double> > &dsham){

  int dsham_len = dsham.size();
  Eigen::SparseMatrix<std::complex<double>>
    matexp_(dsham_len, dsham_len);
  
  Eigen::SparseMatrix<std::complex<double>> H =
    getham(t, pobj, hdrive, dsham, dsham_len, matexp_);
  
  Eigen::Map<Eigen::VectorXcd> y_(y.data(), y.size());
  auto H_ = std::complex<double>(0.0,-1.0) * H * y_;
  return H_;
}



PYBIND11_MODULE(solve,m){
  m.def("solve_func", &solve_func, "solve_func ");
  py::class_<pulsec>(m, "pulsec")
    .def(py::init<
         std::vector< std::vector< double > > &,
         std::vector< std::vector< double > > &,
         std::vector< double > &, double &, int &,
         int & > ())
    .def_readonly("amp", &pulsec::amp)
    .def_readonly("tseq", &pulsec::tseq)
    .def_readonly("freq", &pulsec::freq)
    .def_readonly("duration", &pulsec::duration)
    .def_readonly("nqubit", &pulsec::nqubit)
    .def_readonly("nwindow", &pulsec::nwindow);
}
