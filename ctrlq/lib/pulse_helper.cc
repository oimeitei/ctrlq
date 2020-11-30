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
#include <vector>
#include "fmath.hpp"
#include "pulsec.h"
#include "agradc.h"
#include "grad_ana.h"
#include "pulsehelper.h"

#include <iostream>
#include <iomanip>
#include <chrono>

namespace py = pybind11;

gausgrad gaus_getnamp(int nqubit, int &ngaus, double duration,					     
						 std::vector< std::vector<double >> &amp,
						 std::vector< std::vector<double >> &sigma,
						 std::vector< std::vector<double >> &mean,
						 std::vector<double > freq,
						 std::vector<double> &tlist,
						 std::vector<std::complex<double>> &ini_vec,
						 std::vector< std::vector< Eigen::SparseMatrix
									   <double,0,ptrdiff_t>>> hdrive,
						 std::vector< std::complex<double> > dsham,
						 std::vector< int> &states,
						 Eigen::MatrixXcd &cham){
			       

  double tmpamp;
  int tlen = tlist.size();
  
  std::vector< std::vector< std::vector<double > > >
    expterm(nqubit, std::vector<std::vector<double>> (ngaus, std::vector< double> (tlen)));
  
  std::vector<std::vector<double >> tseq;
  int nwindow = 0;
  double gamp, gsig, gmean, esigmean;
  std::vector<double > agradient;
  
  std::vector<std::vector<double >> tamp(nqubit, std::vector<double> (tlen));
  int i,j,k;
  for (i=0; i<nqubit; i++){
    for (j=0;j<ngaus; j++){
      for (k=0;k<tlen; k++){
	expterm[i][j][k] = fmath::expd(-sigma[i][j] * sigma[i][j] *
				       (tlist[k] - mean[i][j]) *
				       (tlist[k] - mean[i][j]));
	tamp[i][k] += amp[i][j] * expterm[i][j][k];
      }
    }
  }

  pulsec pobj(tamp, tseq, freq, duration, nqubit, nwindow);

  gradc aobj = grad_ana(tlist, ini_vec, pobj, hdrive, dsham, states, cham);

  for(int i=0; i< nqubit; i++){
    for(int j=0; j< ngaus; j++){
      gamp = 0.0;
      gsig = 0.0;
      gmean = 0.0;
      for(int k=0; k< tlen;k++){
	esigmean = aobj.gradient[i][k] * expterm[i][j][k];
	
	gamp += esigmean;
	gsig += esigmean * amp[i][j] * -2.0 * sigma[i][j] *
	  (tlist[k] - mean[i][j])*(tlist[k] - mean[i][j]);
	gmean += esigmean * amp[i][j] * 2.0 * sigma[i][j] * sigma[i][j] *
	  (tlist[k] - mean[i][j]);

      }
      agradient.push_back(gamp);
      agradient.push_back(gsig);
      agradient.push_back(gmean);
    }
  }
  gausgrad gausobj(aobj.energy, aobj.norm, agradient, tamp);

  return gausobj;
}

std::vector<double> gaus_gettamp(int &nqubit, int &ngaus,
				std::vector<double> &tlist,
				std::vector< std::vector<double >> &amp,
				std::vector< std::vector<double >> &sigma,
				std::vector< std::vector<double >> &mean,
				std::vector< std::vector<double >> &gradient_){

  double gamp, gsig, gmean, esigmean;
  std::vector<double > agradient;
  int tlen = tlist.size();

  for(int i=0; i< nqubit; i++){
    for(int j=0; j< ngaus; j++){
      gamp = 0.0;
      gsig = 0.0;
      gmean = 0.0;
      for(int k=0; k< tlen;k++){
	esigmean = gradient_[i][k] * fmath::expd(-sigma[i][j]*sigma[i][j] *
						(tlist[k] - mean[i][j]) *
						(tlist[k] - mean[i][j]));
	gamp += esigmean;
	gsig += esigmean * amp[i][j] * -2.0 * sigma[i][j] *
	  (tlist[k] - mean[i][j])*(tlist[k] - mean[i][j]);
	gmean += esigmean * amp[i][j] * 2.0 * sigma[i][j] * sigma[i][j] *
	  (tlist[k] - mean[i][j]);

      }
      agradient.push_back(gamp);
      agradient.push_back(gsig);
      agradient.push_back(gmean);
    }
  }
  return agradient;
}
	
	
PYBIND11_MODULE(pulse_helper,m){
  m.def("gaus_getnamp", &gaus_getnamp, "gaus_getnamp");
  m.def("gaus_gettamp", &gaus_gettamp, "gaus_gettamp");
  py::class_<gausgrad>(m,"gausgrad")
    .def(py::init<
	 double &, double &,
	 std::vector<double> &,
	 std::vector<std::vector< double > > & > ())
    .def_readonly("energy", &gausgrad::energy)
    .def_readonly("norm", &gausgrad::norm)
    .def_readonly("gradient",&gausgrad::gradient)
    .def_readonly("amp",&gausgrad::amp);
}
