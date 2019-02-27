import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

from scipy.fftpack import fft, irfft, fftfreq, fftshift

from ..helpersFunctions import ConstantsAndModels as CM 
from ..dataParsers.dcdParser import NAMDDCD

from ..lib.pycompIntScatFunc import py_compIntScatFunc




class BackScatData(NAMDDCD):

    def __init__(self, parent, dcdFile=None, stride=1):

        NAMDDCD.__init__(self, parent, dcdFile, stride)

        self.EISF       = None
        self.interFunc  = None
        self.scatFunc   = None
        self.qVals      = None


#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compIntermediateFunc(self, qValList, minFrames, maxFrames, nbrBins=60, nbrTimeOri=20,
            selection='protNonExchH', alignCOM=True, begin=0, end=None):
        """ This method computes intermediate function for all q-value (related to scattering angle)

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    nbrTimeOri  -> number of time origins to be averaged over
                    selection   -> atom selection
                    alignCOM    -> whether center of mass should be aligned or not
                    begin       -> first frame to be used
                    end         -> last frame to be used 
                    
            Returns an (nbr of q-values, timesteps) shaped array. """


        if type(selection) == str:
            selection = self.parent.getSelection(selection)

        qValList = np.array(qValList)
        self.qVals = qValList


        if alignCOM and not self.COMAligned:
            self.setCenterOfMassAligned(begin, end)


        atomPos = self.dcdData[selection, begin:end]

        #_Computes random q vectors
        qArray = []
        for qIdx, qVal in enumerate(qValList):
            qList = [CM.getRandomVec(qVal) for i in range(12)] 
            qArray.append( np.array(qList) )

        qArray = np.array(qArray)
        qArray = np.ascontiguousarray(qArray, dtype=np.float32)


        corr = np.zeros( (qValList.size, nbrBins), dtype=np.cdouble ) 


        timestep = []
        for i in range(nbrBins):
            frameIncr = int( (maxFrames - minFrames + 1) / nbrBins )
            timestep.append( (minFrames + i*frameIncr) * self.timestep * self.dcdFreq[0] )


        print("Computing intermediate scattering function...\n")

        #_Calling the C routine to compute intermediate scattering function
        py_compIntScatFunc(atomPos, qArray, corr, nbrBins, minFrames, maxFrames, nbrTimeOri)
            

        self.interFunc = corr, np.array(timestep)
        print("\nDone\n")






    def compEISF(self, qValList, minFrames, maxFrames, nbrBins=50, nbrTimeOri=20, resFunc=None, 
                                                selection='protNonExchH', alignCOM=True, begin=0, end=None):
        """ This method performs a multiplication of the inverse Fourier transform given resolution 
            function with the computed intermediate function to get the convoluted signal, 
            which can be used to compute MSD. 

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    nbrTimeOri  -> number of time origins to be averaged over
                    resFunc     -> resolution function to be used (optional, default resFuncSPHERES)
                    selection   -> atom selection (optional, default 'protein')
                    alignCOM    -> whether center of mass should be aligned or not
                    begin       -> first frame to be used (optional, default 0)
                    end         -> last frame to be used (optional, default None) """

        if resFunc == None:
            resFunc = CM.FTresFuncSPHERES


        #_Gets intermediate scattering function        
        self.compIntermediateFunc(qValList, minFrames, maxFrames, nbrBins, nbrTimeOri,
                                                                selection, alignCOM, begin, end)

        print("Using given resolution function to compute elastic incoherent structure factors.\n")

        intFunc, timesteps = self.interFunc 

        #_Gets resolution function 
        resolution = resFunc(timesteps)

        eisf = resolution * intFunc #_Computes the EISF

        self.EISF = eisf / eisf[:,0][:,np.newaxis], timesteps #_Returns the normalized EISF and time 




    def compScatteringFunc(self, qValList, minFrames, maxFrames, nbrBins=50, nbrTimeOri=20, resFunc=None,
                                                selection='protNonExchH', alignCOM=True, begin=0, end=None):
        """ This method calls getIntermediateFunc several times for different time steps, given by
            the number of frames, which will start from minFrames and by incremented to reach maxFrames
            in the given number of bins.

            Then, a Fourier transform is performed to compute the scattering function.

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    nbrTimeOri  -> number of time origins to be averaged over
                    selection   -> atom selection (optional, default 'protein')
                    alignCOM    -> whether center of mass should be aligned or not
                    begin       -> first frame to be used (optional, default 0)
                    end         -> last frame to be used (optional, default None) """

        self.compEISF(qValList, minFrames, maxFrames, nbrBins, nbrTimeOri,
                                                        resFunc, selection, alignCOM, begin, end)

        print("Using Fourier transform on all EISF to obtain full scattering function.\n")

        scatFunc, timesteps = self.EISF 

        #_Performs the Fourier transform
        scatFunc = fftshift( fft(scatFunc, axis=1), axes=1 ) / scatFunc.shape[1]

        #_Convert time to energies in micro-electron volts
        timeIncr = (timesteps[1:] - timesteps[:-1])[0]
        energies = 4.135662668e-15 * fftshift( fftfreq(timesteps.size, d=timeIncr) ) * 1e6

        self.scatFunc = scatFunc, energies




    def compMSD(self, frameNbr, selection='protNonExchH', begin=0, end=None):
        """ Computes the Mean-Squared Displacement for the given number of frames, which should correspond
            to the max time scale probed by the instrument.

            Input:  nbrFrame  -> number of frames to be used (time interval)
                    selection -> atom selection to be used to compute MSD
                    begin     -> first frame to be used
                    end       -> last frame to be used """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.getSelection(selection)

        #_Computes atoms positions
        atomPos = self.alignCenterOfMass(selection, begin, end)

        #_Computes intermediate scattering function for one timestep, averaged over time origins
        displacement = atomPos[:,frameNbr:] - atomPos[:,:-frameNbr]

        error = np.std(displacement, axis=1) 
        error = error.mean()

        msd = np.sum( (displacement)**2, axis=2) 
        msd = msd.mean( 1 ) #_Averaging over time origins 

        msd = msd.mean()    #_Averaging over atoms


        return msd, error



#---------------------------------------------
#_Data accession methods
#---------------------------------------------

    def getEISF(self):
        """ Accession method for self.EISF attribute. """

        if not self.EISF:
            return "No EISF was computed yet. Use compEISF method before using this."
        else:
            return self.EISF


    def getIntermediateFunc(self):
        """ Accession method for self.interFunc attribute. """

        if not self.interFunc:
            return ("No intermediate function was computed yet. Use compIntermedateFunc method " 
                    + "before using this.")
        else:
            return self.interFunc


    def getScatFunc(self):
        """ Accession method for self.scatFunc attribute. """

        if not self.scatFunc:
            return ("No scattering function was computed yet. Use compScatFunc method "
                    + "before using this.")
        else:
            return self.scatFunc


#---------------------------------------------
#_Conversion methods (for nPDyn)
#---------------------------------------------
    def convertScatFunctoEISF(self):
        """ First finds the index corresponding to the 0 energy transfer on x axis.
            Then returns an array containing the intensity for each q-value. """

        #_Get the zero energy transfer index
        elasticIdx = np.argwhere(self.scatFunc[0] == np.max(self.scatFunc[0]))[0][1]

        return self.scatFunc[0][:,elasticIdx]


#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotIntermediateFunc(self):
        """ This method calls self.backScatData.getIntermediateFunc to obtain the intermediate 
            scattering function.
            Then, a plot is generated, showing self-correlation for each q-values in qValList. """

        intF, times = self.getIntermediateFunc()
        qValList    = self.qVals
        
        #_Use a fancy colormap
        normColors = colors.Normalize(vmin=np.min(qValList), vmax=np.max(qValList))
        cmap = cm.get_cmap('winter')

 
        for idx, qVal in enumerate(qValList):
            plt.plot(times, intF[idx].real, label="%.2f" % qVal, c=cmap(normColors(qVal)))
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'$I(q,t)$')
            plt.legend(fontsize=12)

        plt.tight_layout()
        return plt.show(block=False)



    def plotEISF(self):
        """ This method calls self.backScatData.getEISF to obtain the Elastic Incoherent Structure Factor.
            Then, a plot is generated, showing self-correlation for each q-values in qValList. """

        EISF, times = self.getEISF()
        qValList    = self.qVals
        
        #_Use a fancy colormap
        normColors = colors.Normalize(vmin=np.min(qValList), vmax=np.max(qValList))
        cmap = cm.get_cmap('winter')

 
        for idx, qVal in enumerate(qValList):
            plt.plot(times, EISF[idx].real, label="%.2f" % qVal, c=cmap(normColors(qVal)))
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'$EISF(q,t)$')
            plt.legend(fontsize=12)

        plt.tight_layout()
        return plt.show(block=False)




    def plotScatteringFunc(self):
        """ This method calls self.backScatDatagetScatteringFunc to obtain the scattering function S(q,omega).
            Then, a plot is generated, showing self-correlation for each q-values in qValList. """

        scatF, energies = self.getScatFunc()
        qValList        = self.qVals
        
        #_Use a fancy colormap
        normColors = colors.Normalize(vmin=np.min(qValList), vmax=np.max(qValList))
        cmap = cm.get_cmap('winter')

        ax = plt.subplot(111, projection='3d')

        for qIdx, qVal in enumerate(qValList):
            ax.plot( energies, np.absolute(scatF[qIdx]), qVal, zorder=50-qVal, 
                                                            zdir='y', c=cmap(normColors(qVal)) )
            ax.set_xlabel(r'Energy ($\mu$eV)', labelpad=15)
            ax.set_ylabel(r'q ($\AA^{-1}$)', labelpad=15)
            ax.set_zlabel(r'$S(q,\omega)$', labelpad=15)

        return plt.show(block=False)


