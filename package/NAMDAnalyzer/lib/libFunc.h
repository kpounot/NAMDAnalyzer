#ifndef LIBFUNC_H
#define LIBFUNC_H

void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *cellDims, float *out, int size_out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous);



void compIntScatFunc(float *atomPos, int atomPos_dim0, int atomPos_dim1, int atomPos_dim2, 
                     float *qVecs, int qVecs_dim0, int qVecs_dim1, int qVecs_dim2, 
                     float *out, int out_dim0, int out_dim1, 
                     int nbrTS, int nbrTimeOri);


void getDistances(float *sel1, int size_sel1, float *sel2, int size_sel2, float *cellDims, float *out);


void getWithin(float *allAtoms, int nbrAtoms, int nbrFrames, 
                int *selAtoms, int sel_dim0, float *cellDims, int *out, float distance);

void getAngles(  float *sel1, int size_sel1,
                 float *vertices,
                 float *sel2, int size_sel2, 
                 float *out );


#endif
