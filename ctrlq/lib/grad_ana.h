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

#ifndef AGRADC_H

gradc grad_ana(
                               std::vector<double> &tlist,
                               std::vector<std::complex<double>> &ini_vec,
                               pulsec pobj,
                               std::vector< std::vector< Eigen::SparseMatrix
			       <double,0,ptrdiff_t>>> hdrive,
                               std::vector< std::complex<double> > dsham,
			       std::vector< int> &states,
			       Eigen::MatrixXcd &cHam);
gradc grad_ana_normalized(
                               std::vector<double> &tlist,
                               std::vector<std::complex<double>> &ini_vec,
                               pulsec pobj,
                               std::vector< std::vector< Eigen::SparseMatrix
			       <double,0,ptrdiff_t>>> hdrive,
                               std::vector< std::complex<double> > dsham,
			       std::vector< int> &states,
			       Eigen::MatrixXcd &cHam);
#endif
 
