import numpy as np


def fromSliceToArange(sliceObj, maxN):
    """ Converts a slice object to a numpy.arange object. 
    
        :arg sliceObj: slice object to be converted
        :arg maxN:     maximum number of element in returned range. Useful when slice.stop attribute is None.
    
    """

    
    #_Process start argument
    if sliceObj.start is None:
        start = 0
    else:
        start = sliceObj.start


    #_Process stop argument
    if sliceObj.stop is None:
        stop = maxN
    else:
        stop = sliceObj.stop

    #_Process step argument
    if sliceObj.step is None:
        step = 1
    else:
        step = sliceObj.step


    return np.arange(start, stop, step)
