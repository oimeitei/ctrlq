
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
