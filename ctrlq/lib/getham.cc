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
 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>
#include "pulsec.h"


/* Piecewise square pulse */

std::complex<double> pcoef(double &t, std::vector< double > &amp,
			   std::vector< double > &tseq,
			   double &freq,
			   double &tfinal){

  double  etmp;
  int i, tlen;
  std::complex <double>  coef;

  
  std::complex<double> etmp1(0.0, -1.0*freq*t);
  etmp1 = std::exp(etmp1);
  
  tlen = tseq.size();
  for (i=0;i<tlen;i++){
    if (i==0){
      if (0.0 < t && t <= tseq[i]){
	coef = amp[i] * etmp1;
      }
    }
    else {
      if (tseq[i-1] < t && t <= tseq[i]){
	coef = amp[i] * etmp1;
      }
    }
  }
  if (tseq[tlen-1] < t && t <= tfinal){
    coef = amp[tlen] * etmp1;
  
  }
  return coef;
}


Eigen::SparseMatrix<std::complex<double> >
getham(double &t, pulsec &pobj,
       std::vector< std::vector< Eigen::SparseMatrix<double,0,ptrdiff_t> > > &hdrive,
       std::vector< std::complex<double> > &dsham, int &dsham_len,
       Eigen::SparseMatrix<std::complex<double> > &matexp_){

  // dsham is different from python version, here it's the diagonal of
  // -1j*hobj.dsham in the python version.
 
  Eigen::SparseMatrix<std::complex<double> > hamdr;
  std::complex<double> hcoef, hcoefc;
 
  int i;
  for (i=0;i<pobj.nqubit;i++) {
    hcoef = pcoef( t, pobj.amp[i], pobj.tseq[i], pobj.freq[i],
		   pobj.duration);
    
    hcoefc = std::conj(hcoef);

    if (i==0){
      hamdr = hcoef * hdrive[i][0];
    } else {
      hamdr += hcoef * hdrive[i][0];
    }

    hamdr += hcoefc * hdrive[i][1];
  }
      
  for (i=0;i<dsham_len; i++){
    matexp_.coeffRef(i,i) = std::exp(dsham[i] * t);
  }

  Eigen::SparseMatrix<std::complex<double> >
  hamr_ = (matexp_.conjugate().transpose() * hamdr);
  hamr_ = (hamr_ * matexp_).pruned();
  
  return hamr_;
}
