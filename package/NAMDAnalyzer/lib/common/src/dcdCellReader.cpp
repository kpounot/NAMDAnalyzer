#include <stdio.h>
#include <stdlib.h>

#include "../../libFunc.h"


enum DCDCELL_ERRORS
{
    SUCCESS         = 0,
    FILE_READ_ERR   = -1,
    OUT_OF_RANGE    = -2,
};



int getDCDCell(char *fileName, int *frames, int nbrFrames, int *startPos, double *outArr)
{
    FILE *dcdFile;

    double *record = (double*) malloc(6*sizeof(double)); // Used to store coordinates for each frame

    int seek;



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
        fread(record, 8, 6, dcdFile);

        for(int i=0; i < 6; ++i)
            outArr[6*frameId + i] = record[i];
    }


    fclose(dcdFile);
    free(record);

    enum DCDCELL_ERRORS error_code = SUCCESS;
    return error_code;
} 


