#include <stdio.h>


enum DCDREADER_ERROR
{
    SUCCESS        = 0,
    FILE_NOT_FOUND = -1,
    OUT_OF_RANGE   = -2 
} error_code;


int getCoordinates(const char *fileName, const int *frames, const int nbrAtoms, const int cell,
                      const int *startPos) 
{

}
