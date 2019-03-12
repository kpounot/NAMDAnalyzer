#ifndef COMPINTSCATFUNC_H
#define COMPINTSCATFUNC_H

#include <complex.h>


void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float complex *out, int out_dim0, int out_dim1, 
                     int binSize, int minFrames, int maxFrames, int nbrTimeOri);

#endif
