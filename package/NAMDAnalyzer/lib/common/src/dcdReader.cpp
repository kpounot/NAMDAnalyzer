#include <stdio.h>
#include <stdlib.h>

#include "../../libFunc.h"


enum DCDREADER_ERRORS
{
    SUCCESS         = 0,
    FILE_READ_ERR   = -1,
    OUT_OF_RANGE    = -2,
    WRONG_OUT_DIMS  = -3
} error_code;


int getDCDCoor(char *fileName, int *frames, int nbrFrames, int nbrAtoms, int *selAtoms, 
                int selAtomsSize, int *dims, int nbrDims, int cell, int *startPos, float *outArr)
{
    FILE *dcdFile;

    float *record = (float*) malloc(nbrAtoms*sizeof(float)); // Used to store coordinates for each frame

    int seek;


    dcdFile = fopen(fileName, "rb");
    if(dcdFile == NULL)
    {
        error_code = FILE_READ_ERR; 
        return error_code;
    }



    for(int frameId=0; frameId < nbrFrames; ++frameId)
    {

        int frame = frames[frameId];

        for(int dimId=0; dimId < nbrDims; ++dimId)
        {
            if(cell)
                seek = fseek(dcdFile, startPos[frame] + dims[dimId]*(4*nbrAtoms+8) + 60, SEEK_SET);
            else
                seek = fseek(dcdFile, startPos[frame] + dims[dimId]*(4*nbrAtoms+8) + 4, SEEK_SET);

            if(seek != 0)
            {
                error_code = OUT_OF_RANGE;
                return error_code;
            }


            // Reads the record, x, y and z coordinates for each atom, flanked by integers of 4 bytes.
            int read = fread(record, 4, nbrAtoms, dcdFile);
            if(read != nbrAtoms)
            {
                error_code = WRONG_OUT_DIMS;
                return error_code;
            }


            // Copy coordinates in 'out' array
            for(int atomId=0; atomId < selAtomsSize; ++atomId)
                outArr[nbrDims*nbrFrames*atomId + nbrDims*frameId + dimId] = record[ selAtoms[atomId] ];
        }
    }


    fclose(dcdFile);
    free(record);

    error_code = SUCCESS;
    return error_code;
} 


