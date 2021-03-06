"""

Classes
^^^^^^^

"""


import sys

import numpy as np

import matplotlib.pyplot as plt

try:
    from NAMDAnalyzer.lib.pylibFuncs import py_getHBCorr, py_getHBNbr
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")


class HydrogenBonds:
    """ This class defines methods to compute hydrogen bonds
        auto-correlation functions.

        Some plotting methods are also available to quicly check the results.

        :arg data:          a :class:`Dataset` class instance containing
                            trajectories data
        :arg acceptors:     selection of acceptors atoms for hydrogen bonds
        :arg donors:        selection of donors atoms for hydrogen bonds
        :arg hydrogens:     selection of hydrogens bound to donors
                            (optional, if None, hydrogens
                             will be guessed from donors list)
        :arg maxTime:       maximum time interval to be used for correlation
                            in number of frames (optional, default 100)
        :arg step:          number of frames between each time interval
                            points (optional, default 1)
        :arg nbrTimeOri:    number of time origins to be averaged over
                            (optional, default 25)
        :arg maxR:          maximum distance to allow for hydrogen bonding,
                            angström (optional, default 2.8)
        :arg minAngle:      minimum angle to allow for hydrogen bonding
                            (optional, default 130)

        References:
           * D.C. Rapaport (1983): Hydrogen bonds in water,
             Molecular Physics: An International Journal at the
             Interface Between Chemistry and Physics, 50:5, 1151-1162

    """


    def __init__(self, data, acceptors='hbacceptors', donors='hbdonors',
                 hydrogens=None, maxTime=50, nbrTimeOri=20, step=1,
                 maxR=2.5, minAngle=130):

        self.data       = data

        self.acceptors  = acceptors
        self.donors     = donors
        self.hydrogens  = hydrogens

        self.step       = step
        self.maxTime    = maxTime
        self.nbrTimeOri = nbrTimeOri
        self.maxR       = maxR
        self.minAngle   = minAngle

        self.times = np.arange(0, maxTime, step)

        self.cc = np.ascontiguousarray(
            np.zeros(self.times.size), dtype='float32')
        self.ic = np.ascontiguousarray(
            np.zeros(self.times.size), dtype='float32')

        self.nbrHB = None


    def _processDonors(self, donors):
        """ This function takes a list of indices corresponding to
            selected hydrogen bond donor atoms. Then, bound hydrogens are
            found and two lists of same size are returned. One containing
            all hydrogens, and the other their associated donors.

        """

        donors = self.data.selection(donors)

        # Gets hydrogens bound to donors
        allH    = self.data.selection('hbhydrogens')
        boundH  = np.argwhere(self.data.getBoundAtoms(donors))[:, 0]

        boundH = np.intersect1d(boundH, allH)
        boundH.sort()


        # Finds donor atom linked to each selected hydrogen.
        # Generates a donor list of same siez as boundH
        bonds1  = self.data.psfData.bonds[:, ::2]
        bonds2  = self.data.psfData.bonds[:, 1::2]

        selBonds1   = np.argwhere(
            np.isin(bonds1, self.data.psfData.atoms[boundH][:, 0].astype(int)))
        selBonds2   = np.argwhere(
            np.isin(bonds2, self.data.psfData.atoms[boundH][:, 0].astype(int)))

        outDonors = np.concatenate(
            (bonds2[selBonds1[:, 0], selBonds1[:, 1]],
             bonds1[selBonds2[:, 0], selBonds2[:, 1]])) - 1
        outDonors.sort()

        # Keeps the donors only
        outDonors = outDonors[np.argwhere(np.isin(outDonors, donors))[:, 0]]



        return outDonors, boundH



# --------------------------------------------
# Computation methods
# --------------------------------------------
    def compAutoCorrel(self, continuous=1):
        """ Computes the hydrogen bonds autocorrelation function.

            Both distances and angles are computed exactly,
            without any approximation, to single-point precision.

            :arg continuous: set to 1, continuous correlation is computed
                             set to 0 for intermittent case

            The result, a 1D array containing correlations for all time
            intervals is stored in :py:attr:`cc`  or :py:attr:`ic`
            attributes variable for continuous or intermittent types
            respectively.

        """

        if continuous == 1:
            self.cc *= 0
        else:
            self.ic *= 0


        self.times = (np.arange(0, self.maxTime, self.step, dtype='int32')
                      * self.data.dcdFreq[0] * self.data.timestep * 1e12)

        # To store final result for each time interval
        corr = np.ascontiguousarray(np.zeros(self.times.size), dtype='float32')

        oriList = ((self.data.nbrFrames - self.maxTime)
                   * np.random.random(self.nbrTimeOri)).astype(int)

        for idx, frame in enumerate(oriList):
            print("Computing time origin %i of %i with %i frames..."
                  % (idx + 1, self.nbrTimeOri, self.times.size), end='\r')

            corr *= 0  # Reset corr to its initial state

            # Updating selection for each time origin
            acceptors = self.data.selection(self.acceptors
                                            + ' frame %i' % frame)


            # Processes selections
            if self.hydrogens is None:
                donors, hydrogens = self._processDonors(
                    self.donors + ' frame %i' % frame)
            else:
                donors = self.data.selection(
                    self.donors + ' frame %i' % frame)
                hydrogens = self.data.selection(
                    self.hydrogens + ' frame %i' % frame)


            # Gets coordinates and cell dimensions
            allAtoms = self.data.dcdData[
                :, frame:frame + self.maxTime:self.step]

            cellDims = self.data.cellDims[frame:frame + self.maxTime:self.step]


            # Computes hydrogen bond autocorrelation for given frames
            py_getHBCorr(allAtoms[acceptors],
                         self.times.size,
                         allAtoms[donors], allAtoms[hydrogens],
                         cellDims,
                         corr, self.maxTime,
                         self.step, self.nbrTimeOri, self.maxR,
                         self.minAngle, continuous)



            if continuous == 1:
                self.cc += corr / (corr[0] * self.nbrTimeOri)
            else:
                self.ic += corr / (corr[0] * self.nbrTimeOri)




    def compHBNumber(self, frames=None):
        """ Computes the number of hydrogen bonds for each selected frame.

            Both distances and angles are computed exactly,
            without any approximation, to single-point precision.

            :arg frames: the frames from which the number of hydrogen
                         bonds will be determined.

            The result, a 1D array containing the number of hydrogen bonds
            for each frame is stored in the :py:attr:`nbrHB` class attribute.

        """

        if frames is None:
            frames = np.arange(self.data.nbrFrames)


        # Get the acceptors for each selected frame
        acceptors = self.data.selection(self.acceptors)
        acceptors = self.data.dcdData[acceptors, frames]

        # Processes selections
        if self.hydrogens is None:
            donors, hydrogens = self._processDonors(self.donors)
        else:
            donors = self.data.selection(self.donors)
            hydrogens = self.data.selection(self.hydrogens)

        donors = self.data.dcdData[donors, frames]
        hydrogens = self.data.dcdData[hydrogens, frames]


        cellDims = self.data.cellDims[frames]

        # To store final result for each time interval
        out = np.ascontiguousarray(np.zeros(acceptors.shape[1]),
                                   dtype='float32')

        py_getHBNbr(acceptors, acceptors.shape[1], donors, hydrogens,
                    cellDims, out, self.maxR, self.minAngle)


        self.nbrHB = out



# --------------------------------------------
# Plotting methods
# --------------------------------------------
    def plotAutoCorrel(self, corrType='continuous'):
        """ Used to quickly plot autocorrelation function, either continuous or
            intermittent depending on the value of corrType parameter.

            :arg corrType: type of correlation to plot
            :type corrType: str

        """


        if corrType == 'continuous':
            data = self.cc
            yLabel = '$C_{continuous}(t)$'
        else:
            data = self.ic
            yLabel = '$C_{intermittent}(t)$'

        plt.plot(self.times, data, marker='o')
        plt.xlabel('Time [ps]')
        plt.ylabel(yLabel)

        plt.show(block=False)
