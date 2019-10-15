#ifndef LIBFUNC_H
#define LIBFUNC_H




int getDCDCoor(char *fileName, int *frames, int nbrFrames, int nbrAtoms, int *selAtoms, 
                int selAtomsSize, int *dims, int nbrDims, int cell, int *startPos, float *outArr);

int getDCDCell(char *fileName, int *frames, int nbrFrames, int *startPos, double *outArr);


void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *cellDims,
                        float *out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous);



void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float *out, int out_dim0, int out_dim1, 
                     int nbrTS, int nbrTimeOri);



void getDistances(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                  float *out, float *cellDims, int nbrFrames, int sameSel);


void getRadialNbrDensity(float *sel1, int sel1_size, float *sel2, int sel2_size, 
                         float *out, int outSize, float *cellDims, int nbrFrames, int sameSel, 
                         float maxR, float dr);


void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames,
                int *refSel, int refSize, int *outSel, int outSelSize,
                int *out, float *cellDims, float distance);


void waterOrientAtSurface(float *waterO, int sizeO, float *watVec, float *prot, 
                          int sizeP, float *out, float *cellDims, int nbrFrames, float maxR, int maxN);

void setWaterDistPBC(float *water, int sizeW, float *prot, int sizeP, float *cellDims, int nbrFrames,
                     int nbrWAtoms);

int getParallelBackend();


#endif
