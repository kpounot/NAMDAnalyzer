#include "../../libFunc.h"


void getHBCorr_wrapper(  float *acceptors, int size_acceptors, int nbrFrames,
                                float *donors, int size_donors,
                                float *hydrogens, int size_hydrogens, 
                                float *cellDims,
                                float *out, int maxTime, int step, 
                                int nbrTimeOri, float maxR, float minAngle, int continuous);

void getHBCorr(  float *acceptors, int size_acceptors, int nbrFrames,
                        float *donors, int size_donors,
                        float *hydrogens, int size_hydrogens, 
                        float *cellDims,
                        float *out, int maxTime, int step, int nbrTimeOri,
                        float maxR, float minAngle, int continuous)
{
    getHBCorr_wrapper(acceptors, size_acceptors, nbrFrames, donors, size_donors, 
                                hydrogens, size_hydrogens, cellDims, out, maxTime, 
                                step, nbrTimeOri, maxR, minAngle, continuous);

}






void getHBNbr_wrapper(  float *acceptors, int size_acceptors, int nbrFrames,
                                float *donors, int size_donors,
                                float *hydrogens, int size_hydrogens, 
                                float *cellDims,
                                float *out, 
                                float maxR, float minAngle );


void getHBNbr(  float *acceptors, int size_acceptors, int nbrFrames,
                float *donors, int size_donors,
                float *hydrogens, int size_hydrogens, 
                float *cellDims,
                float *out, 
                float maxR, float minAngle )
{
    getHBNbr_wrapper(acceptors, size_acceptors, nbrFrames, donors, size_donors,
                                hydrogens, size_hydrogens, cellDims, out, 
                                maxR, minAngle );
}
