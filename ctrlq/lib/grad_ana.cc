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
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <vector>
#include <typeinfo>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "getham.h"
#include "agradc.h"

gradc grad_ana(
	       std::vector<double> &tlist,
	       std::vector<std::complex<double>> &ini_vec,
	       pulsec pobj,
	       std::vector< std::vector< Eigen::SparseMatrix
	       <double,0,ptrdiff_t>>> hdrive,
	       std::vector< std::complex<double> > dsham,
	       std::vector< int> &states,
	       Eigen::MatrixXcd &cHam){

  Eigen::SparseMatrix<std::complex<double> > hamdr;
  Eigen::SparseMatrix<std::complex<double> > hamdR;
  std::complex<double> hcoef; //, hcoefc;
  
  int i,j;
  int dsham_len = dsham.size();
  int tlen = tlist.size();
  int nstate = states.size();
  int &nqubit = pobj.nqubit;
  double energy;
  
  std::complex<double> psit_g;
  
  Eigen::SparseMatrix<std::complex<double>>
    matexp_(dsham_len, dsham_len);

  Eigen::SparseMatrix<std::complex<double>> hD, Hd_;
  Eigen::MatrixXcd H1_, P_;
  Eigen::VectorXcd psi_;
  Eigen::VectorXcd states_(nstate);
  
  std::vector< Eigen::VectorXcd > O_(nqubit,
				     Eigen::VectorXcd (dsham_len));
  
  std::vector< Eigen::VectorXcd > g1__(nqubit,
				       Eigen::VectorXcd (dsham_len));
  
  std::vector< Eigen::VectorXcd > g1_(nqubit,
				       Eigen::VectorXcd (nstate));
  
  Eigen::Map<Eigen::VectorXcd> ket_(ini_vec.data(), ini_vec.size());
  std::vector< Eigen::SparseMatrix<std::complex<double> > > hamd(nqubit);
  std::vector<std::vector<double> > t_grad(nqubit,
					   std::vector<double> (tlen));
    
  double tau = tlist[tlen-1] / tlen;
  std::complex<double> im(0.0,-tau);
  std::complex<double> imp(0.0,tau);
    
  for (int t=0; t<tlen; t++){
  
    hD = getham2(tlist[t], t, pobj, hdrive, dsham, dsham_len, matexp_, hamd, hamdr, hamdR, hcoef);
    H1_ = im * Eigen::MatrixXcd(hD);
    ket_ = H1_.exp() * ket_;    
  }

  for (i=0; i<nqubit; i++){
    g1__[i] = hamd[i] * ket_;
  }

  for (i=0; i<nstate; i++){
    states_[i] = ket_[states[i]];
  }
  double nrm =  states_.norm();
    
  for (int i = 0; i<nqubit; i++){
    for (int j=0; j<nstate; j++){
      g1_[i][j] = g1__[i][states[j]];
    }
  }  

  psi_ = cHam * states_;

  std::complex<double> energy_ = states_.conjugate().transpose() * psi_;
  energy = energy_.real();

  psi_ = psi_.conjugate();
  Eigen::Transpose<Eigen::VectorXcd> psi1_ = psi_.transpose();
  P_ = Eigen::MatrixXcd::Identity(dsham_len, dsham_len);

  double tau1 = 2.0 * tau;

  for (i=0; i<nqubit; i++){
    
    psit_g = psi1_ * g1_[i];
    t_grad[i][tlen-1] = tau1 * psit_g.imag();
  }
        
  for (int idx=tlen-1; idx > 0; idx--){
    Hd_ = getham3( tlist[idx], idx, pobj, hdrive, dsham, dsham_len, matexp_, hamdr, hamdR, hcoef);
    hD = getham2(tlist[idx-1], idx-1, pobj, hdrive, dsham, dsham_len,
		 matexp_, hamd, hamdr, hamdR, hcoef);


    H1_ = (imp*Eigen::MatrixXcd(Hd_)).exp();
    ket_ = H1_ * ket_;
    P_ = P_ * H1_.conjugate().transpose();

    for (i=0; i<nqubit; i++){
      O_[i] = P_ * hamd[i] * ket_;
    }
    
    for (i=0; i<nqubit; i++){
      for (j=0; j<nstate; j++){
	g1_[i][j] = O_[i][states[j]];
      }
    }

    for (i=0; i<nqubit; i++){
      psit_g = psi1_ * g1_[i];
      t_grad[i][idx-1] = tau1 * psit_g.imag();
    }    
  }  
  gradc g1(energy, nrm, t_grad);
    
  return g1;
}


gradc grad_ana_normalized(
                               std::vector<double> &tlist,
                               std::vector<std::complex<double>> &ini_vec,
                               pulsec pobj,
                               std::vector< std::vector< Eigen::SparseMatrix
			       <double,0,ptrdiff_t>>> hdrive,
                               std::vector< std::complex<double> > dsham,
			       std::vector< int> &states,
			       Eigen::MatrixXcd &cHam){

  Eigen::SparseMatrix<std::complex<double> > hamdr;
  Eigen::SparseMatrix<std::complex<double> > hamdR;
  std::complex<double> hcoef; 
  
  int i,j;
  int dsham_len = dsham.size();
  int tlen = tlist.size();
  int nstate = states.size();
  int &nqubit = pobj.nqubit;
  double energy,nrm2, nrm4;
  
  std::complex<double> psit_g,psit_gN;
  
  Eigen::SparseMatrix<std::complex<double>>
    matexp_(dsham_len, dsham_len);

  Eigen::SparseMatrix<std::complex<double>> hD, Hd_;
  Eigen::MatrixXcd H1_, P_;
  Eigen::VectorXcd psi_,psi_N;
  Eigen::VectorXcd states_(nstate);

  std::vector< Eigen::VectorXcd > O_(nqubit,
				     Eigen::VectorXcd (dsham_len));
  
  std::vector< Eigen::VectorXcd > g1__(nqubit,
				       Eigen::VectorXcd (dsham_len));
  
  std::vector< Eigen::VectorXcd > g1_(nqubit,
				       Eigen::VectorXcd (nstate));
  
  Eigen::Map<Eigen::VectorXcd> ket_(ini_vec.data(), ini_vec.size());
  std::vector< Eigen::SparseMatrix<std::complex<double> > > hamd(nqubit);
  std::vector<std::vector<double> > t_grad(nqubit,
					   std::vector<double> (tlen));
    
  double tau = tlist[tlen-1] / tlen;
  std::complex<double> im(0.0,-tau);
  std::complex<double> imp(0.0,tau);
    
  for (int t=0; t<tlen; t++){
  
    hD = getham2(tlist[t], t, pobj, hdrive, dsham, dsham_len, matexp_, hamd, hamdr, hamdR, hcoef);
    H1_ = im * Eigen::MatrixXcd(hD);
    ket_ = H1_.exp() * ket_;    
  }

  for (i=0; i<nqubit; i++){
    g1__[i] = hamd[i] * ket_;
  }

  for (i=0; i<nstate; i++){
    states_[i] = ket_[states[i]];
  }
  double nrm =  states_.norm();
  nrm2 = nrm*nrm;
  nrm4 = nrm2*nrm2;
    
  for (int i = 0; i<nqubit; i++){
    for (int j=0; j<nstate; j++){
      g1_[i][j] = g1__[i][states[j]];
    }
  }  

  psi_ = cHam * states_;
  psi_N = states_;

  std::complex<double> energy_ = states_.conjugate().transpose() * psi_;
  energy = energy_.real();

  psi_ = psi_.conjugate();
  Eigen::Transpose<Eigen::VectorXcd> psi1_ = psi_.transpose();
  psi_N = psi_N.conjugate();
  Eigen::Transpose<Eigen::VectorXcd> psi1_N = psi_N.transpose();
  P_ = Eigen::MatrixXcd::Identity(dsham_len, dsham_len);
  double tau1 = 2.0 * tau;
  nrm4 = tau1/nrm4;

  for (i=0; i<nqubit; i++){
    
    psit_g = psi1_ * g1_[i];
    psit_gN = psi1_N * g1_[i];  
    t_grad[i][tlen-1] = (psit_g.imag()*nrm2 - energy*psit_gN.imag())*nrm4;    
  }
        
  for (int idx=tlen-1; idx > 0; idx--){
    Hd_ = getham3( tlist[idx], idx, pobj, hdrive, dsham, dsham_len, matexp_, hamdr, hamdR, hcoef);
    hD = getham2(tlist[idx-1], idx-1, pobj, hdrive, dsham, dsham_len,
		 matexp_, hamd, hamdr, hamdR, hcoef);

    H1_ = (imp*Eigen::MatrixXcd(Hd_)).exp();
    ket_ = H1_ * ket_;
    P_ = P_ * H1_.conjugate().transpose();

    for (i=0; i<nqubit; i++){
      O_[i] = P_ * hamd[i] * ket_;
    }
    
    for (i=0; i<nqubit; i++){
      for (j=0; j<nstate; j++){
	g1_[i][j] = O_[i][states[j]];
      }
    }

    for (i=0; i<nqubit; i++){
      psit_g = psi1_ * g1_[i];
      psit_gN = psi1_N * g1_[i];
      t_grad[i][idx-1] = (psit_g.imag()*nrm2 - energy*psit_gN.imag())*nrm4;
    }    
  }
  energy = energy/nrm2;
  gradc g1(energy, nrm, t_grad);
    
  return g1;
}
