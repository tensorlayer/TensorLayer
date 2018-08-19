%module pafprocess
%{
  #define SWIG_FILE_WITH_INIT
  #include "pafprocess.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

//%apply (int DIM1, int DIM2, int* IN_ARRAY2) {(int p1, int p2, int *peak_idxs)}
//%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3) {(int h1, int h2, int h3, float *heatmap), (int f1, int f2, int f3, float *pafmap)};
%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3) {(int p1, int p2, int p3, float *peaks), (int h1, int h2, int h3, float *heatmap), (int f1, int f2, int f3, float *pafmap)};
%include "pafprocess.h"
