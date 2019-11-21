#include <stdio.h>
#include <stdlib.h>

#include "../../libFunc.h"
#include "getEndian.h"


#define swapBytes(val) \
    ( (((val) >> 56) & 0x00000000000000FF) | (((val) >> 40) & 0x000000000000FF00) | \
      (((val) >> 24) & 0x0000000000FF0000) | (((val) >>  8) & 0x00000000FF000000) | \
      (((val) <<  8) & 0x000000FF00000000) | (((val) << 24) & 0x0000FF0000000000) | \
      (((val) << 40) & 0x00FF000000000000) | (((val) << 56) & 0xFF00000000000000) )



enum DCDCELL_ERRORS
{
    SUCCESS         = 0,
    FILE_READ_ERR   = -1,
    OUT_OF_RANGE    = -2,
    WRONG_OUT_DIMS  = -3
};



int getDCDCell(char *fileName, int *frames, int nbrFrames, int *startPos, double *outArr, char byteorder)
{
    FILE *dcdFile;

    char *record = (char*) malloc(6*sizeof(double)); // Used to store coordinates for each frame

    int seek;

    char sysbyteorder = getEndian();


    dcdFile = fopen(fileName, "rb");
    if(dcdFile == NULL)
    {
        enum DCDCELL_ERRORS error_code = SUCCESS;
        return error_code;
    }



    for(int frameId=0; frameId < nbrFrames; ++frameId)
    {
        int frame = frames[frameId];

        seek = fseek(dcdFile, startPos[frame] + 4, SEEK_SET);

        if(seek != 0)
        {
            enum DCDCELL_ERRORS error_code = OUT_OF_RANGE;
            return error_code;
        }


        // Reads the record, x, y and z coordinates for each atom, flanked by integers of 4 bytes.
        int read = fread(record, 8, 6, dcdFile);
        if(read != 6)
        {
            enum DCDCELL_ERRORS error_code = WRONG_OUT_DIMS;
            return error_code;
        }



        if(sysbyteorder != byteorder)
        {
            for(int i=0; i < 6; ++i)
                swapBytes( *(long*) &record[8*i] );
        }

        for(int i=0; i < 6; ++i)
            outArr[6*frameId + i] = *(double*) &record[8*i];
    }


    fclose(dcdFile);
    free(record);

    enum DCDCELL_ERRORS error_code = SUCCESS;
    return error_code;
} 


