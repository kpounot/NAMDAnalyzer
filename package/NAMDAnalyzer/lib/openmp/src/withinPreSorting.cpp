#include <cstdio>
#include <cmath>
#include <list>
#include <omp.h>

#include "../../libFunc.h"


void withinPreSorting(float *allAtoms, int allSize, float *selAtoms, int selSize, 
                      int *out, float distance)
{
    for(int selId=0; selId < selSize; ++selId)
    {
        float sel_x = abs(selAtoms[3*selId]);
        float sel_y = abs(selAtoms[3*selId + 1]);
        float sel_z = abs(selAtoms[3*selId + 2]);

        for(int atomId=0; atomId < allSize; ++atomId)
        {
            if(out[atomId] != 1)
            {

                int keep_x = 0;
                int keep_y = 0;
                int keep_z = 0;

                float atom_x = abs(allAtoms[3*atomId]);
                float atom_y = abs(allAtoms[3*atomId + 1]);
                float atom_z = abs(allAtoms[3*atomId + 2]);

                if(atom_x < sel_x + distance && atom_x > sel_x - distance)
                    keep_x = 1;
                if(atom_y < sel_y + distance && atom_y > sel_y - distance)
                    keep_y = 1;
                if(atom_z < sel_z + distance && atom_z > sel_z - distance)
                    keep_z = 1;

                out[atomId] = keep_x*keep_y*keep_z;
            }
        }
    }

}
