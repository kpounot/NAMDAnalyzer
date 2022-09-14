#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

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
    FILE_SEEK_ERR   = -3
};



int getDCDCoor(
        char *fileName, 
        int *frames, 
        int nbrFrames, 
        int nbrAtoms, 
        int *selAtoms, 
        int selAtomsSize, 
        int *dims, 
        int nbrDims, 
        int cell, 
        long long *startPos, 
        float *outArr,
        char byteorder
) {
    FILE *dcdFile;
    // Used to store coordinates for each frame
    char *record = (char*) malloc(nbrAtoms * 4);
    char sysbyteorder = getEndian();
    int seek;

    long long pos;
    float res;

    dcdFile = fopen(fileName, "rb");
    if(dcdFile == NULL)
    {
        enum DCDREADER_ERRORS error_code = FILE_READ_ERR; 
        return error_code;
    }


    fseek(dcdFile, 0, SEEK_END);
    long long fileSize = ftell(dcdFile);
    for(int frameId=0; frameId < nbrFrames; ++frameId)
    {
        int frame = frames[frameId];

        for(int dimId=0; dimId < nbrDims; ++dimId)
        {

            pos = startPos[frame] + dims[dimId] * (4 * nbrAtoms + 8);
            pos += cell ? 60 : 4;
            seek = fseeko64(dcdFile, pos, SEEK_SET);

            if(seek != 0)
            {
                enum DCDREADER_ERRORS error_code = OUT_OF_RANGE;
                return error_code;
            }

            fread(record, 4, selAtoms[selAtomsSize - 1] + 1, dcdFile);

            // Copy coordinates in 'out' array
            #pragma omp parallel for 
            for(int atomId=0; atomId < selAtomsSize; ++atomId)
            {
                if(sysbyteorder != byteorder)
                    swapBytes(*record);

                outArr[nbrDims*nbrFrames*atomId + nbrDims*frameId + dimId] =
                    *(float*) &record[4 * selAtoms[atomId]];
            }
        }
    }

    fclose(dcdFile);
    free(record);

    enum DCDREADER_ERRORS error_code = SUCCESS;
    return error_code;
} 


