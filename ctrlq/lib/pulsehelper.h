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

#ifndef PULSEHELPER_H
struct gausgrad{
  double energy;
  double norm;
  std::vector<double> gradient;
  std::vector<std::vector<double >> amp;

  gausgrad(double &energy_,double &norm_,
	std::vector<double> &gradient_,
	std::vector<std::vector<double > > &amp_):	
  energy(energy_), norm(norm_),gradient(gradient_),amp(amp_){};
};

#endif
 
