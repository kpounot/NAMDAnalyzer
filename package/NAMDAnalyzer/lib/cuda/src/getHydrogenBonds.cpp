#include "../../libFunc.h"


void getHydrogenBonds_wrapper(  float *acceptors, int size_acceptors, int nbrFrames,
                                float *donors, int size_donors,
                                float *hydrogens, int size_hydrogens, 
                                float *out, int maxTime, int step, 
                                int nbrTimeOri, float maxR, float minAngle, int continuous);

void getHydrogenBonds(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous)
{
    getHydrogenBonds_wrapper(acceptors, size_acceptors, nbrFrames, donors, size_donors, 
                                hydrogens, size_hydrogens, out, maxTime, 
                                step, nbrTimeOri, maxR, minAngle, continuous);

}

