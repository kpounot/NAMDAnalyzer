#include <stdio.h>
#include <stdlib.h>

#include "../../libFunc.h"
#include "getEndian.h"


#define swapBytes(val) \
    ( (((val) >> 24) & 0x000000FF) | (((val) >>  8) & 0x0000FF00) | \
      (((val) <<  8) & 0x00FF0000) | (((val) << 24) & 0xFF000000) )





enum DCDREADER_ERRORS
{
    SUCCESS         = 0,
    FILE_READ_ERR   = -1,
    OUT_OF_RANGE    = -2,
    WRONG_OUT_DIMS  = -3
};


int getDCDCoor(char *fileName, int *frames, int nbrFrames, int nbrAtoms, int *selAtoms, 
                int selAtomsSize, int *dims, int nbrDims, int cell, int *startPos, float *outArr,
                char byteorder)
{
    FILE *dcdFile;

    char *record = (char*) malloc(nbrAtoms*sizeof(float)); // Used to store coordinates for each frame

    int seek;

    char sysbyteorder = getEndian();


    dcdFile = fopen(fileName, "rb");
    if(dcdFile == NULL)
    {
        enum DCDREADER_ERRORS error_code = FILE_READ_ERR; 
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
                enum DCDREADER_ERRORS error_code = OUT_OF_RANGE;
                return error_code;
            }


            // Reads the record, x, y and z coordinates for each atom, flanked by integers of 4 bytes.
            int read = fread(record, 4, nbrAtoms, dcdFile);
            if(read != nbrAtoms)
            {
                enum DCDREADER_ERRORS error_code = WRONG_OUT_DIMS;
                return error_code;
            }

            if(sysbyteorder != byteorder)
            {
                for(int i=0; i < nbrAtoms; ++i)
                    swapBytes( *(int*) &record[4*i] );
            }


            // Copy coordinates in 'out' array
            for(int atomId=0; atomId < selAtomsSize; ++atomId)
            {
                float res = *(float*) &record[ 4*selAtoms[atomId] ];
                outArr[nbrDims*nbrFrames*atomId + nbrDims*frameId + dimId] = res;
            }
        }
    }


    fclose(dcdFile);
    free(record);

    enum DCDREADER_ERRORS error_code = SUCCESS;
    return error_code;
} 


