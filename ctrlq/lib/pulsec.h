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

/* A pulsec struct */

#ifndef PULSEC_H

struct pulsec{
  std::vector< std::vector< double > > amp;
  std::vector< std::vector< double > > tseq;
  std::vector< double > freq;

  double duration;
  int nqubit;
  int nwindow;

  // the python version pulse has more attributes
  pulsec( std::vector< std::vector< double > > &amp_,
         std::vector< std::vector< double > > &tseq_,
         std::vector< double > &freq_,
         double &duration_, int &nqubit_, int &nwindow_) :
  amp(amp_), tseq(tseq_), freq(freq_), duration(duration_),
    nqubit(nqubit_), nwindow(nwindow_) {};
};

#endif
