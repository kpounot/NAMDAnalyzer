"""

Classes
^^^^^^^

"""

import numpy as np

from NAMDAnalyzer.lib.pylibFuncs import py_getDCDCell

class DCDCell:
    """ This is a simple class that works as an accessor for cell dimensions from a dcd file.

        It is not intended to be used on its own, but to be called from :class:`DCDReader` class
        using its *cellDims* attribute: ``data.cellDims[2:10]``.

        :arg data: a class instance that inherits from :class:`DCDReader`. 
                    It can be a :class:`Dataset` instance.

    """

    def __init__(self, data):

        self.data = data


    def __getitem__(self, frames):

        frameSel = np.arange(self.data.nbrFrames, dtype=int)

        if isinstance(frames, slice):
            start = frames.start if frames.start is not None else 0
            stop  = frames.stop if frames.stop is not None else self.data.nbrFrames
            step  = frames.step if frames.step is not None else 1

            frameSel  = np.arange( start, stop, step )


        elif isinstance(frames, (int, list, np.ndarray, np.int32, np.int64)):
            if isinstance(frames, (int, np.int32, np.int64)):
                frameSel = np.array([frames])
            elif isinstance(frames, list):
                frameSel = np.array(frames)
            else:
                frameSel = frames

        else:
            print("Selection couldn't be understood, please use slicing to select cell dimensions.")
            return


        #_Extract coordinates given selected atoms, frames and coordinates
        out = np.zeros( (len(frameSel), 6 ), dtype='float64')
        for idx, f in enumerate(self.data.dcdFiles):
            fileFrames = np.arange(self.data.initFrame[idx], self.data.stopFrame[idx], dtype=int)
            fileFrames, id1, id2 = np.intersect1d(fileFrames, frameSel, return_indices=True)

            tmpOut = np.ascontiguousarray(out[id2], dtype='float64')

            py_getDCDCell(bytearray(f, 'utf-8'), id1.astype('int32'), 
                            self.data.startPos[idx].astype('int32'), 
                            tmpOut)

            out[id2] = tmpOut



        return np.ascontiguousarray(out[ :,[0,2,5] ]).astype('float32')






