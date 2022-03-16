"""

Classes
^^^^^^^

"""

import numpy as np

try:
    from NAMDAnalyzer.lib.pylibFuncs import py_getDCDCell
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")


class DCDCell:
    """ This is a simple class that works as an accessor for cell dimensions
        from a dcd file.

        It is not intended to be used on its own, but to be called from
        :class:`DCDReader` class using its *cellDims* attribute:
        ``data.cellDims[2:10]``, for example.

        :arg data: a class instance that inherits from :class:`DCDReader`.
                    It can be a :class:`Dataset` instance.

    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, frames):
        frameSel = np.arange(self.data.nbrFrames, dtype=int)
        coorSel  = np.arange(3)

        if isinstance(frames, slice):
            start = frames.start if frames.start is not None else 0
            stop  = (frames.stop if frames.stop is not None
                     else self.data.nbrFrames)
            step  = frames.step if frames.step is not None else 1
            frameSel = np.arange(start, stop, step)

        elif isinstance(frames, (int, list, range,
                                 np.ndarray, np.int32, np.int64)):
            if isinstance(frames, (int, np.int32, np.int64)):
                frameSel = np.array([frames])
            elif isinstance(frames, list):
                frameSel = np.array(frames)
            else:
                frameSel = frames

        elif len(frames) == 2:
            if isinstance(frames[0], slice):
                start = frames[0].start if frames[0].start is not None else 0
                stop  = (frames[0].stop if frames[0].stop is not None
                         else self.data.nbrFrames)
                step  = frames[0].step if frames[0].step is not None else 1

                frameSel  = np.arange(start, stop, step)

            if isinstance(frames[0], (int, list, np.ndarray)):
                frameSel = np.array(
                    [frames[0]]) if isinstance(frames[0], int) else frames[0]

            if isinstance(frames[1], slice):
                start = frames[1].start if frames[1].start is not None else 0
                stop  = frames[1].stop if frames[1].stop is not None else 3
                step  = frames[1].step if frames[1].step is not None else 1

                coorSel = np.arange(start, stop, step)

            if isinstance(frames[1], (int, list, np.ndarray)):
                coorSel = np.array(
                    [frames[1]]) if isinstance(frames[1], int) else frames[1]

        elif len(frames) > 2:
            print("Too many dimensions requested, maximum is 2.")
            return

        else:
            print("Selection couldn't be understood, please use "
                  "slicing to select cell dimensions.")
            return

        # Extract coordinates given selected atoms, frames and coordinates
        out = np.zeros((len(frameSel), 6), dtype='float64')

        # If no cell information is present, returns cell
        # dimensions big enough so that it does not have any effect
        if self.data.cell == 0:
            out += 100000

        for idx, f in enumerate(self.data.dcdFiles):
            if isinstance(f, str):
                fileFrames = np.arange(self.data.initFrame[idx],
                                       self.data.stopFrame[idx],
                                       dtype=int)
                fileFrames, id1, id2 = np.intersect1d(
                    fileFrames, frameSel, return_indices=True)

                tmpOut = np.ascontiguousarray(out[id2], dtype='float64')

                self._processErrorCode(
                    py_getDCDCell(bytearray(f, 'utf-8'),
                                  id1.astype('int32'),
                                  self.data.startPos[idx].astype('int64'),
                                  tmpOut,
                                  ord(self.data.byteorder)))

                out[id2] = tmpOut

            elif isinstance(f, np.ndarray):
                out += 100000

        return np.ascontiguousarray(
            out[:, [0, 2, 5]]).astype('float32')[:, coorSel]


    def _processErrorCode(self, error_code):
        """ Used to process return value of py_getDCDCoor function. """
        if error_code == 0:
            return

        if error_code == -1:
            raise IOError("Error while reading the file. "
                          "Please check file path or access permissions.\n")

        if error_code == -2:
            raise IndexError("Out of range index. "
                             "Please check again requested slices.\n")