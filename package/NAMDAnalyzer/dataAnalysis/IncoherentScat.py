"""

Classes
^^^^^^^

"""

import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

from scipy.fft import fft, fftfreq, fftshift

from threading import Thread

from NAMDAnalyzer.helpersFunctions import ConstantsAndModels as CM

try:
    from NAMDAnalyzer.lib.pylibFuncs import py_compIntScatFunc
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")



class IncoherentScat:
    """ This class provides methods to convert trajectories into
        experimental-like Quasi-Elastic Neutron Scattering spectra (QENS) or
        Dynamic Structure Factors (DSF).
        Mean-Squared Displacements (MSD) can also be computed directly
        from trajectories.

        Some plotting methods are also available to quicly check the results.

        :arg dataset: a :class:`Dataset` class instance containing
                      trajectories data.

    """



    def __init__(self, dataset):

        self.dataset = dataset

        self.DSF       = None
        self.interFunc  = None
        self.scatFunc   = None
        self.qVals      = None
        self.MSD        = None


# --------------------------------------------
# Computation methods
# --------------------------------------------
    def compIntermediateFunc(self, qValList, nbrTimeOri=50,
                             selection='protNonExchH',
                             alignCOM=True, frames=slice(0, None, 1),
                             nbrTS=200):
        """ This method computes intermediate function for all q-value
            (related to scattering angle)

            :arg qValList:   list of q-values to be used
            :arg nbrTimeOri: number of time origins to be averaged over
            :arg selection:  atom selection
                             (usually 'protNonExchH' or 'waterH')
            :arg alignCOM:   whether center of mass should be aligned or not
            :arg frames:     either None to select all frames, an int,
                             or a slice object
            :arg nbrTS:      number of time steps to be used
                             (number of points ni the x-axis output)

            Result is stored in *interFunc* attribute as the following tuple:
                - **corr** auto-correlation function as a
                           (nbr of q-values, nbr of timesteps) shaped array.
                - **ts**   timesteps as calculated from selected frames

        """


        if type(selection) == str:
            selection = self.dataset.selection(selection)

        qValList = np.array(qValList)
        self.qVals = qValList

        # Align center of mass if necessary
        if alignCOM:
            print('Aligning center of mass of selected atoms...\n')
            atomPos = self.dataset.getAlignedCenterOfMass(selection,
                                                          frames=frames)
            print('Done\n')

        else:
            atomPos = self.dataset.dcdData[selection, frames]

        # Computes random q vectors
        qArray = []
        for qIdx, qVal in enumerate(qValList):
            qList = [CM.getRandomVec(qVal) for i in range(15)]
            qArray.append(np.array(qList))

        qArray = np.array(qArray)
        qArray = np.ascontiguousarray(qArray, dtype=np.float32)


        scatLength = self.dataset.psfData.nScatLength_inc[selection]
        scatLength = scatLength.astype('float32')


        corr = np.zeros((qArray.shape[0], 2 * nbrTS), dtype=np.float32)


        # Get timestep array
        step = frames.step if frames.step is not None else 1

        step *= ((atomPos.shape[1] - nbrTimeOri) / nbrTS)

        timestep = np.arange(nbrTS) * step 
        timestep *= self.dataset.timestep 
        timestep *= self.dataset.dcdFreq[0]

        print("Computing intermediate scattering function...\n")

        py_compIntScatFunc(atomPos, qArray, corr, nbrTS, 
                           nbrTimeOri, scatLength)

        # Convert to complex array
        corr = corr[:, ::2] + 1j * corr[:, 1::2]

        self.interFunc = corr, timestep

        print("\nDone\n")



    def compDSF(self, qValList, nbrTimeOri=50, resFunc=None,
                selection='protNonExchH', alignCOM=True,
                frames=slice(0, None, 1), norm=True, nbrTS=200):
        """ This method performs a multiplication of the inverse Fourier
            transform given resolution function with the computed
            intermediate function to get the instrumental dynamic
            structure factor.

            :arg qValList:   list of q-values to be used
            :arg nbrTimeOri: number of time origins to be averaged over
            :arg resFunc:    resolution function to be used
                             (default a standard gaussian)
            :arg selection:  atom selection
                             (usually 'protNonExchH' or 'waterH')
            :arg alignCOM:   whether center of mass should be aligned or not
            :arg frames:     either None to select all frames, an int,
                             or a slice object
            :arg norm:       whether result should be normalized
                             by first point or not
            :arg nbrTS:      number of time steps to be used
                             (number of points ni the x-axis output)

            Result is stored in *DSF* attribute as the following tuple:
                - **dsf**  auto-correlation function as a
                           (nbr of q-values, nbr of timesteps) shaped array.
                - **ts**   timesteps as calculated from selected frames

        """

        if resFunc is None:
            resFunc = CM.FTresFuncSPHERES


        # Computes intermediate scattering function if None
        self.compIntermediateFunc(qValList, nbrTimeOri, selection,
                                  alignCOM, frames, nbrTS)

        print("Using given resolution function to compute elastic "
              "incoherent structure factors.\n")

        intFunc, timesteps = self.interFunc

        # Gets resolution function
        resolution = resFunc(timesteps)

        dsf = resolution * intFunc  # Computes the DSF

        if norm:
            dsf /= dsf[:, 0][:, np.newaxis]

        self.DSF = dsf, timesteps  # Returns the DSF and time




    def compScatteringFunc(self, qValList, nbrTimeOri=50, resFunc=None,
                           selection='protNonExchH', alignCOM=True,
                           frames=slice(0, None, 1), norm=True, nbrTS=200):
        """ This method calls the :func:`compDSF()` method, performs a
            Fourier transform on the result and computes the power
            spectrum of it.

            :arg qValList:   list of q-values to be used
            :arg nbrTimeOri: number of time origins to be averaged over
            :arg selection:  atom selection
                             (usually 'protNonExchH' or 'waterH')
            :arg alignCOM:   whether center of mass should be aligned or not
            :arg frames:     either None to select all frames, an int,
                             or a slice object
            :arg nbrTS:      number of time steps to be used
                             (number of points ni the x-axis output)

            Result is stored in *scatFunc* attribute as the following tuple:
                - **scatFunc** spectra as a
                              (nbr of q-values, nbr of timesteps) shaped array.
                - **energies** energies in ueV obtained from timesteps

        """


        self.compDSF(qValList, nbrTimeOri, resFunc, selection,
                     alignCOM, frames, norm, nbrTS)

        print("Using Fourier transform on all DSF to obtain "
              "full scattering function.\n")

        dsf, timesteps = self.DSF

        # Performs the Fourier transform
        scatFunc = fftshift(fft(dsf, axis=1), axes=1)
        scatFunc = np.absolute(scatFunc)**2 / scatFunc.shape[1]

        # Convert time to energies in micro-electron volts
        timeIncr = timesteps[1] - timesteps[0]
        energies = 4.135662668e-15 * fftshift(
            fftfreq(timesteps.size, d=timeIncr)) * 1e6

        self.scatFunc = scatFunc, energies



    def compMSD(self, nbrFrames=20, selection='protNonExchH',
                frames=slice(0, None, 1), alignCOM=True, nbrSubFrames=10):
        """ Computes the Mean-Squared Displacement for the given number of
            frames, which should correspond to the max time scale probed
            by the instrument.

            :arg nbrFrames:     number of frames to be used (time interval)
            :arg selection:     atom selection to be used to compute MSD
            :arg frames:        either None to select all frames,
                                or a slice object
            :arg alignCOM:      whether center of mass should be
                                aligned or not
            :arg nbrSubFrames:  number of sub-selection of frames to be
                                averaged on. Basically, the selected frames
                                are sliced to generate the given number
                                of subsets. The MSD is computed for each
                                subset, then the average and the standard
                                deviation are computed.

            Result is stored in *MSD* attribute

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.dataset.selection(selection)

        # ALign center of masses if required
        if alignCOM:
            atomPos = self.dataset.getAlignedCenterOfMass(
                'all', selection, frames).astype('float32')
        else:
            atomPos = self.dataset.dcdData[selection, frames].astype('float32')


        # Parses the 'frames' argument
        if isinstance(frames, slice):
            start  = frames.start if frames.start is not None else 0
            stop   = (frames.stop - nbrFrames if frames.stop is not None
                      else self.dataset.nbrFrames - nbrFrames)
            step   = int(np.ceil(((stop - nbrFrames) - start) / nbrSubFrames))
        elif frames is None:
            start = 0
            stop  = self.dataset.nbrFrames - nbrFrames
            step  = int(np.ceil(stop / nbrSubFrames))
        else:
            raise TypeError('The argument `frames` should be either None '
                            'or a `slice` object')


        frames = np.arange(start, stop - nbrFrames, step)


        # Computes intermediate scattering function for one timestep,
        # averaged over time origins
        tmpMSD = []
        for frm in frames:
            displacement = np.sum((
                atomPos[:, frm + nbrFrames:frm + step + nbrFrames]
                - atomPos[:, frm:frm + step])**2,
                axis=2)
            tmpMSD.append(displacement.mean())

        tmpMSD = np.array(tmpMSD)

        self.MSD = tmpMSD.mean(), tmpMSD.std()



# --------------------------------------------
# Conversion methods (for nPDyn)
# --------------------------------------------
    def convertScatFunctoEISF(self):
        """ First finds the index corresponding to the 0 energy
            transfer on x axis in *scatFunc* attribute.
            Then returns an array containing the intensity for each q-value.

        """

        # Get the zero energy transfer index
        elasticIdx = np.argwhere(self.scatFunc[0] == np.max(
            self.scatFunc[0]))[0][1]

        return self.scatFunc[0][:, elasticIdx]


# --------------------------------------------
# Plotting methods
# --------------------------------------------
    def plotIntermediateFunc(self):
        """ Plots the intermediate scattering function for each q-value. """

        intF, times = self.interFunc
        qValList    = self.qVals

        # Use a fancy colormap
        normColors = colors.Normalize(vmin=np.min(qValList),
                                      vmax=np.max(qValList))
        cmap = cm.get_cmap('winter')


        for idx, qVal in enumerate(qValList):
            plt.plot(times, intF[idx].real, label="%.2f" % qVal,
                     c=cmap(normColors(qVal)))
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'$I(q,t)$')
            plt.legend(fontsize=12)

        plt.tight_layout()
        return plt.show(block=False)



    def plotDSF(self):
        """ Plots the DSF for each q-value.

        """

        DSF, times = self.DSF
        qValList    = self.qVals

        # Use a fancy colormap
        normColors = colors.Normalize(vmin=np.min(qValList),
                                      vmax=np.max(qValList))
        cmap = cm.get_cmap('winter')


        for idx, qVal in enumerate(qValList):
            plt.plot(times, DSF[idx].real, label="%.2f" % qVal,
                     c=cmap(normColors(qVal)))
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'$DSF(q,t)$')
            plt.legend(fontsize=12)

        plt.tight_layout()
        return plt.show(block=False)




    def plotScatteringFunc(self):
        """ Plots the MSD for each q-value.

        """

        scatF, energies = self.scatFunc
        qValList        = self.qVals

        # Use a fancy colormap
        normColors = colors.Normalize(vmin=np.min(qValList),
                                      vmax=np.max(qValList))
        cmap = cm.get_cmap('winter')

        ax = plt.subplot(111, projection='3d')

        for qIdx, qVal in enumerate(qValList):
            ax.plot(energies, np.absolute(scatF[qIdx]), qVal,
                    zorder=50 - qVal,
                    zdir='y', c=cmap(normColors(qVal)))
            ax.set_xlabel(r'Energy ($\mu$eV)', labelpad=15)
            ax.set_ylabel(r'q ($\AA^{-1}$)', labelpad=15)
            ax.set_zlabel(r'$S(q,\omega)$', labelpad=15)

        plt.tight_layout()
        return plt.show(block=False)
