#ifndef LIBFUNC_H
#define LIBFUNC_H

void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous);



void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float *out, int out_dim0, int out_dim1, 
                     int nbrTS, int nbrTimeOri);


void getDistances(float *maxSel, int maxSize, float *minSel, int minSize, 
                  float *out, float *cellDims, int sameSel);

void compDistances(float *maxSel, int maxSize, float *minSel, int minSize, 
                  float *out, float *cellDims);

void compDistances_same(float *maxSel, int maxSize, float *minSel, int minSize, 
                  float *out, float *cellDims);

void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames, 
                int *selAtoms, int sel_dim0,
                int *out, float *cellDims, float distance);


int getParallelBackend();


#endif
