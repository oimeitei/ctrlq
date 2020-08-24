#ifndef GETHAM_H
#define GETHAM_H
#include "pulsec.h"


Eigen::SparseMatrix<std::complex<double>>
getham(double &t, pulsec &pobj,
       std::vector< std::vector< Eigen::SparseMatrix<double,0,ptrdiff_t>>> &hdrive,
       std::vector< std::complex<double> > &dsham, int &dsham_len,
       Eigen::SparseMatrix<std::complex<double>> &matexp_);



#endif
