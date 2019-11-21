#include "getEndian.h"


char getEndian()
{
    char res;

    short int number = 0x1;
    char *numPtr = (char*)&number;

    if(numPtr[0] == 1)
        res = '>';
    else
        res = '<';

    return res;
}

