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

namespace py = pybind11;

Eigen::MatrixXcd solve_trotter(		       
			       std::vector<double> &tlist,
			       std::vector<std::complex<double>> &ini_vec,
			       pulsec pobj,
			       std::vector< std::vector< Eigen::SparseMatrix<double,0,ptrdiff_t>>> hdrive,
			       std::vector< std::complex<double> > dsham){

  Eigen::SparseMatrix<std::complex<double>> H_ ;
  Eigen::MatrixXcd H1_;
  int dsham_len = dsham.size();
  Eigen::SparseMatrix<std::complex<double>>
    matexp_(dsham_len, dsham_len);
  
  Eigen::Map<Eigen::VectorXcd> trot_(ini_vec.data(), ini_vec.size());
  int tlen = tlist.size();
  double tau = tlist[tlen-1] / tlen;
  std::complex<double> im(0.0,-tau);

  for (int t=0; t<tlen; t++){
    H_ = getham(tlist[t], pobj, hdrive, dsham, dsham_len, matexp_);
    H1_ = im * Eigen::MatrixXcd(H_);

    trot_ = H1_.exp() * trot_;
  }

  return trot_;
}  

Eigen::MatrixXcd solve_trotter2(
			std::vector<double> &tlist,
			std::vector<std::complex<double>> &ini_vec,
			pulsec pobj,
			std::vector< std::vector< Eigen::SparseMatrix
			<double,0,ptrdiff_t>>> hdrive,
			std::vector< std::complex<double> > dsham){
  
  Eigen::SparseMatrix<std::complex<double>> H_ ;
  Eigen::MatrixXcd H1_;
  
  int dsham_len = dsham.size();
  int tlen = tlist.size();
  
  Eigen::SparseMatrix<std::complex<double> > hamdr;
  Eigen::SparseMatrix<std::complex<double> > hamdR;
  std::complex<double> hcoef;

  Eigen::SparseMatrix<std::complex<double>>
    matexp_(dsham_len, dsham_len);
  
  Eigen::Map<Eigen::VectorXcd> trot_(ini_vec.data(), ini_vec.size());

  
  double tau = tlist[tlen-1] / tlen;
  std::complex<double> im(0.0,-tau);

  for (int t=0; t<tlen; t++){
    H_ = getham3( tlist[t], t, pobj, hdrive, dsham, dsham_len, matexp_,
		  hamdr, hamdR, hcoef);
    H1_ = im * Eigen::MatrixXcd(H_);
    trot_ = H1_.exp() * trot_;
  }
  return trot_;
}

PYBIND11_MODULE(trotter,m){
  m.def("solve_trotter", &solve_trotter, "trotter");
  m.def("solve_trotter2", &solve_trotter2, "trotter2");
}
