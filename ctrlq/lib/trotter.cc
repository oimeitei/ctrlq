#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <vector>
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
  //I = I * tau;
  
  for (int t=0; t<tlen; t++){
    H_ = getham(tlist[t], pobj, hdrive, dsham, dsham_len, matexp_);
    //H1_ = I * Eigen::MatrixXcd(getham(tlist[t], pobj, hdrive, dsham));
    H1_ = im * Eigen::MatrixXcd(H_);
    
    trot_ = H1_.exp() * trot_;
  }
  return trot_;
}  



PYBIND11_MODULE(trotter,m){
  m.def("solve_trotter", &solve_trotter, "trotter");
}
