#include <cstdio>
#include <cmath>
#include <list>

#include "getWithin.h"


void getWithin(float *allAtoms, int nbrAtoms, float *selAtoms, int sel_dim0, float *cellDims,
                                                                                    int *out, float distance)
{
    /*  The function computes, for each atom in sel, the distance with all other atoms in atomPos.
     *  Then, Periodic Boundary Conditions (PBC) are applied using cell dimensions in cellDims.
     *  Finally, computed distances are compared to required 'within distance' and if lower or equal, the
     *  corresponding entry is set to 1 in the out array.
     *
     *  Input:  allAtoms    -> flattened 2D array of atom positions for a given frame
     *                         dimensions: atoms (0), coordinates (1)
     *          selAtoms    -> same as atomPos but contains only the atoms from which distance will be 
     *                         computed
     *          nbrAtoms    -> total number of atoms in the system
     *          cellDims    -> dimensions of the cell in x, y and z (used for periodic boundary conditions) 
     *          out         -> output array of floats to store within atoms, must be 1D array of size nbrAtoms 
     *          distance    -> distance (not squared) within which atoms should be kept */



    //Declaring some variables to avoid unecessary memory allocations
    float dist_0;
    float dist_1;
    float dist_2;
    float dr;

    float squaredDist = distance*distance;

    std::list<int> atomIds;
    for(int i = 0; i < nbrAtoms; ++i)
        atomIds.push_back(i);


    std::list<int> eraseList;


    for(int selAtom=0; selAtom < sel_dim0; ++selAtom)
    {

        printf("Finding neighbours around atom %d of %d...\r", selAtom+1, sel_dim0);

        for(std::list<int>::iterator it=atomIds.begin(); it != atomIds.end(); ++it)
        {

            // Computes distances for given timestep and atom
            dist_0 = allAtoms[3 * *(it)] - selAtoms[3*selAtom];
            dist_1 = allAtoms[3 * *(it) + 1] - selAtoms[3*selAtom+1];
            dist_2 = allAtoms[3 * *(it) + 2] - selAtoms[3*selAtom+2];

        
            // Applying PBC corrections
            dist_0 = dist_0 - cellDims[0] * round( dist_0 / cellDims[0] );
            dist_1 = dist_1 - cellDims[1] * round( dist_1 / cellDims[1] );
            dist_2 = dist_2 - cellDims[2] * round( dist_2 / cellDims[2] );


            if(dist_0 > distance || dist_1 > distance || dist_2 > distance) 
                continue;


            dr = dist_0*dist_0 + dist_1*dist_1 + dist_2*dist_2;


            if(dr <= squaredDist) 
            {
                out[*it] = 1;
                eraseList.push_back(*it);
            }

        } // atoms indices loop



        for(auto const& id: eraseList)
        {
            atomIds.remove(id); // Remove already found atoms from the search list
        }


        eraseList.clear(); // Clearing eraseList for the next iteration


    } // sel atoms loop

}

