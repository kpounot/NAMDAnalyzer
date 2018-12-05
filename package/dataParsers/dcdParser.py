import os, sys
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D


from collections import namedtuple

from ..dataManipulation.molFit_quaternions import alignAllMol
from ..dataConverters.backscatteringDataConvert import BackScatData
from .dcdReader import DCDReader


class NAMDDCD(DCDReader, BackScatData):
    """ This class contains methods for trajectory file analysis. """

    def __init__(self, dcdFile=None):
    
        DCDReader.__init__(self)
        BackScatData.__init__(self)

        if dcdFile:
            self.dcdFile = dcdFile
            self.importDCDFile(dcdFile)



#---------------------------------------------
#_Data modifiers
#---------------------------------------------
    def binDCD(self, binSize):
        """ Binning method for dcd data. The binning is performed along the axis 1 of the dataset,
            which corresponds to frames dimension. """

        nbrLoops = int(self.dcdData.shape[1] / binSize)

        #_Performs the binning
        for i in range(nbrLoops):
            self.dcdData[:,i] = np.mean( self.dcdData[:, i*binSize : i*binSize+binSize], axis=1)
            self.dcdFreq[i] = np.sum( self.dcdFreq[i*binSize : i*binSize+binSize] )

        #_Free the memory
        self.dcdData = np.delete(self.dcdData, range(nbrLoops,self.dcdData.shape[1]), axis=1)
        self.dcdFreq = np.delete(self.dcdFreq, range(nbrLoops,self.dcdFreq.size))

        #_Returns only the useful information
        self.dcdData = self.dcdData[:,:nbrLoops]
        self.dcdFreq = self.dcdFreq[:nbrLoops]


        
#---------------------------------------------
#_Data analysis methods
#---------------------------------------------
    def getSTDperAtom(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Computes the standard deviation for each atom in selection and for frames between
            begin and end. 
            Returns the standard deviation averaged over time.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   begin     -> first frame to be used
                   end       -> last frame to be used + 1
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, begin, end)
        else:
            data = self.dcdData[selection, begin:end]

        #_Computes the standard deviation
        if mergeXYZ:
            std = np.sqrt(np.sum(data**2, axis=2))
            std = np.std(std, axis=1)

        else:
            std = np.std(data, axis=1)

        return std


    def getRMSDperAtom(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Computes the RMSD for each atom in selection and for frames between begin and end.
            Returns the RMSD averaged over time.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   begin     -> first frame to be used
                   end       -> last frame to be used + 1
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, begin, end)
        else:
            data = self.dcdData[selection, begin:end]

        #_Computes the standard deviation
        if mergeXYZ:
            rmsd = np.sqrt(np.sum(data**2, axis=2))

        
        rmsd = np.apply_along_axis(lambda arr: (arr - rmsd[:,0])**2, 0, rmsd)
        rmsd = np.mean(rmsd, axis=1)

        return rmsd


    def getRMSDperFrame(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Computes the RMSD for each atom in selection and for frames between begin and end.
            Returns the RMSD averaged over all selected atoms.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   begin     -> first frame to be used
                   end       -> last frame to be used + 1
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, begin, end)
        else:
            data = self.dcdData[selection, begin:end]

        #_Computes the standard deviation
        if mergeXYZ:
            rmsd = np.sqrt(np.sum(data**2, axis=2))

        
        rmsd = np.apply_along_axis(lambda arr: (arr - rmsd[:,0])**2, 0, rmsd)
        rmsd = np.mean(rmsd, axis=0)

        return rmsd
 

    def getCenterOfMass(self, selection="all", begin=0, end=None):
        """ Computes the center of mass of the system for atoms between firstAtom and lastATom,
            and for frames between begin and end. """

        try: #_Check if a psf file has been loaded
            #_Get the indices corresponding to the selection
            if type(selection) == str:
                selection = self.getSelection(selection)

            atomMasses = self.getAtomsMasses(selection)

        except AttributeError:
            print("No .psf file was loaded, please import one before using this method.")
            return

        atomMasses = atomMasses.reshape(1, atomMasses.size)

        centerOfMass = np.dot(atomMasses, self.dcdData[selection,begin:end,0])
        centerOfMass = np.row_stack( (centerOfMass, np.dot(atomMasses, 
                                                        self.dcdData[selection,begin:end,1])) )
        centerOfMass = np.row_stack( (centerOfMass, np.dot(atomMasses, 
                                                        self.dcdData[selection,begin:end,2])) )

        centerOfMass = centerOfMass / np.sum(atomMasses)

        return centerOfMass


    def getAlignedData(self, selection, begin=0, end=None):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Returns a similar array as the initial dataSet but with aligned coordinates."""

        centerOfMass = self.getCenterOfMass(selection, begin, end)
        alignData    = np.copy(self.dcdData[selection,begin:end,:])   

        alignData = alignAllMol(alignData, centerOfMass)
        
        return alignData



    def alignCenterOfMass(self, selection='all', begin=0, end=None):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be used for center of mass
            drift corrections if no global angular momentum is present. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        centerOfMass = self.getCenterOfMass(selection, begin, end)

        dataSet = np.copy( self.dcdData[selection, begin:end] )

        #_Substract the center of mass coordinates to each atom for each frame
        for i in range(dataSet.shape[1]):
            dataSet[:,i,:] = dataSet[:,i,:] - centerOfMass[:,i]

        return dataSet



#---------------------------------------------
#_Plotting methods
#---------------------------------------------

    def plotSTDperAtom(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Plot the standard deviation along the axis 0 of dataSet.
            This makes use of the 'getSTDperAtom method.

            If mergeXYZ is True, then computes the distance to the origin first. """

        std = self.getSTDperAtom(selection, align, begin, end, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[begin:end])

        if mergeXYZ:
            plt.plot(xRange, std)
            plt.ylabel(r'$STD \ (\AA)$')

        else:
            #_In case of three columns for (x, y, z) coordinates, generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, std[:,0])
            ax[0].set_ylabel(r'$STD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, std[:,1])
            ax[1].set_ylabel(r'$STD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, std[:,2])
            ax[2].set_ylabel(r'$STD \ along \ Z \ (\AA)$')

        plt.xlabel(r'$Atom \ index$')

        plt.tight_layout()
        return plt.show(block=False)


    def plotRMSDperAtom(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Plot the RMSD along the axis 0 of dataSet.
            This makes use of the 'getRMSDperAtom method.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        rmsd = self.getRMSDperAtom(selection, align, begin, end, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[begin:end])

        if mergeXYZ:
            plt.plot(xRange, rmsd)
            plt.ylabel(r'$RMSD \ (\AA)$')

        else:
            #_In case of three columns for (x, y, z) coordinates, generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, rmsd[:,0])
            ax[0].set_ylabel(r'$RMSD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, rmsd[:,1])
            ax[1].set_ylabel(r'$RMSD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, rmsd[:,2])
            ax[2].set_ylabel(r'$RMSD \ along \ Z \ (\AA)$')

        plt.xlabel(r'$Atom \ index$')

        plt.tight_layout()
        return plt.show(block=False)


    def plotRMSDperFrame(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Plot the RMSD along the axis 1 of dataSet.
            This makes use of the 'getRMSDperFrame method.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        rmsd = self.getRMSDperFrame(selection, align, begin, end, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[begin:end])

        if mergeXYZ:
            plt.plot(xRange, rmsd)
            plt.ylabel(r'$RMSD \ (\AA)$')

        else:
            #_In case of three columns for (x, y, z) coordinates, generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, rmsd[:,0])
            ax[0].set_ylabel(r'$RMSD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, rmsd[:,1])
            ax[1].set_ylabel(r'$RMSD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, rmsd[:,2])
            ax[2].set_ylabel(r'$RMSD \ along \ Z \ (\AA)$')

        plt.xlabel(r'Time (s)')

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
            plt.plot(times, EISF[idx].real, label=qVal, c=cmap(normColors(qVal)))
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'$EISF(q,t)$')
            plt.legend(fontsize=12)

        plt.tight_layout()
        return plt.show(block=False)



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
            plt.plot(times, intF[idx].real, label=qVal, c=cmap(normColors(qVal)))
            plt.xlabel(r'Time (s)')
            plt.ylabel(r'$I(q,t)$')
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


