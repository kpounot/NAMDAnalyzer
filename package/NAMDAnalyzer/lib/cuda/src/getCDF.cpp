#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void getCDF( float *dist, int size_dist, float *out, int size_out, 
             float maxR, float dr, float normF )
{

    for(int idx=0; idx < size_dist; ++idx)
    {
        d = dist[idx];

        if(d > maxR)
            continue;



    }

}
