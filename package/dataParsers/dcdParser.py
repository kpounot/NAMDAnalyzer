import os, sys
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D


from scipy.io import FortranFile
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from struct import *

from collections import namedtuple

from ..dataManipulation import *


class NAMDDCD:
    """ This class contains methods for trajectory file analysis. """

    def __init__(self, dcdfile, parent=None):

        self.parent = parent

        try:
            dcdFile = FortranFile(dcdfile, 'r')
        except:
            print("Error while reading the file.\n" + "Please check file path/extension.")
            return

        #_Get some simulation parameters (frames, steps and dcd frequency)
        header = dcdFile.read_record('i4')
        self.nbrFrames  = header[1]
        dcdFreq         = header[3]
        self.nbrSteps   = header[4]
        self.timestep   = unpack('f', pack('i', header[10]))[0] * 0.04888e-12 

        #_Skip the next record
        dcdFile.read_record(dtype='i4')

        #_Get the number of atoms
        self.nbrAtoms = dcdFile.read_record(dtype='i4')[0]

        #_Allocating memory to make the process much faster
        self.dataSet = np.zeros((self.nbrAtoms, 3*self.nbrFrames))

        #_Check the first entry to determine how data are organized
        firstEntry = dcdFile.read_record('f')

        if firstEntry.shape == self.nbrAtoms:
            self.dataSet[:,0] = firstEntry
            #_Read data for all frames and store coordinated in self.dataSet 
            for i in range(3 * self.nbrFrames - 1):
                self.dataSet[:,i+1] = dcdFile.read_record('f')

        else: #_We need to skip each four lines
            self.dataSet[:,0] = dcdFile.read_record('f')
            self.dataSet[:,1] = dcdFile.read_record('f')
            self.dataSet[:,2] = dcdFile.read_record('f')
            for i in range(self.nbrFrames - 1):
                dcdFile.read_record('f')
                self.dataSet[:,3 + 3*i] = dcdFile.read_record('f')
                self.dataSet[:,4 + 3*i] = dcdFile.read_record('f')
                self.dataSet[:,5 + 3*i] = dcdFile.read_record('f')

        #_The dataset is reshaped so that we have atom index in axis 0, frame number in axis 1, 
        #_and (x, y, z) coordinates in axis 2
        self.dataSet = self.dataSet.reshape(self.dataSet.shape[0], self.nbrFrames, 3)

        #_Converting dcdFreq to an array of size nbrFrames for handling different dcdFreq 
        #_during conversion to time
        self.dcdFreq = np.zeros(self.dataSet.shape[1]) + dcdFreq



    def binDCD(self, binSize):
        """ Binning method for dcd data. The binning is performed along the axis 1 of the dataset,
            which corresponds to frames dimension. """

        nbrLoops = int(self.dataSet.shape[1] / binSize)

        #_Performs the binning
        for i in range(nbrLoops):
            self.dataSet[:,i] = np.mean( self.dataSet[:, i*binSize : i*binSize+binSize], axis=1)
            self.dcdFreq[i] = np.sum( self.dcdFreq[i*binSize : i*binSize+binSize] )

        #_Free the memory
        self.dataSet = np.delete(self.dataSet, range(nbrLoops,self.dataSet.shape[1]), axis=1)
        self.dcdFreq = np.delete(self.dcdFreq, range(nbrLoops,self.dcdFreq.size))

        #_Returns only the useful information
        self.dataSet = self.dataSet[:,:nbrLoops]
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
            selection = self.parent.psfData.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, begin, end)
        else:
            data = self.dataSet[selection, begin:end]

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
            selection = self.parent.psfData.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, begin, end)
        else:
            data = self.dataSet[selection, begin:end]

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
            selection = self.parent.psfData.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, begin, end)
        else:
            data = self.dataSet[selection, begin:end]

        #_Computes the standard deviation
        if mergeXYZ:
            rmsd = np.sqrt(np.sum(data**2, axis=2))

        
        rmsd = np.apply_along_axis(lambda arr: (arr - rmsd[:,0])**2, 0, rmsd)
        rmsd = np.mean(rmsd, axis=0)

        return rmsd
 

    def getCenterOfMass(self, selection="all", begin=0, end=None):
        """ Computes the center of mass of the system for atoms between firstAtom and lastATom,
            and for frames between begin and end. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.psfData.getSelection(selection)

        try: #_Check if a psf file has been loaded
            atomMasses = self.parent.psfData.getAtomsMasses(selection)
        except AttributeError:
            print("No .psf file was loaded, please import one before using this method.")
            return

        atomMasses = atomMasses.reshape(1, atomMasses.size)

        centerOfMass = np.dot(atomMasses, self.dataSet[selection,begin:end,0])
        centerOfMass = np.row_stack( (centerOfMass, np.dot(atomMasses, 
                                                        self.dataSet[selection,begin:end,1])) )
        centerOfMass = np.row_stack( (centerOfMass, np.dot(atomMasses, 
                                                        self.dataSet[selection,begin:end,2])) )

        centerOfMass = centerOfMass / np.sum(atomMasses)

        return centerOfMass


    def getAlignedData(self, selection, begin=0, end=None):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Returns a similar array as the initial dataSet but with aligned coordinates."""

        centerOfMass = self.getCenterOfMass(selection, begin, end)
        alignData    = np.copy(self.dataSet[selection,begin:end,:])   

        alignData = molFit_q.alignAllMol(alignData, centerOfMass)
        
        return alignData



    def alignCenterOfMass(self, selection='all', begin=0, end=None):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be used for center of mass
            drift corrections if no global angular momentum is present. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.psfData.getSelection(selection)

        centerOfMass = self.getCenterOfMass(selection, begin, end)

        dataSet = np.copy( self.dataSet[selection, begin:end] )

        #_Substract the center of mass coordinates to each atom for each frame
        for i in range(dataSet.shape[1]):
            dataSet[:,i,:] = dataSet[:,i,:] - centerOfMass[:,i]

        return dataSet




    def getIntermediateFunc(self, qValList, minFrames, maxFrames, nbrBins=100, selection='protNonExchH', 
                                                                                    begin=0, end=None):
        """ This method computes intermediate function for all q-value (related to scattering angle)

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    selection   -> atom selection
                    begin       -> first frame to be used
                    end         -> last frame to be used 
                    
            Returns an (nbr of q-values, timesteps) shaped array. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.psfData.getSelection(selection)

        qValList = np.array(qValList).reshape(len(qValList), 1) #_For convenience

        #_Computes atoms positions
        atomPos = np.sqrt(np.sum(self.alignCenterOfMass('all', begin, end)[selection]**2, axis=2))

        intermediateFunc = np.zeros(qValList.size) #_Initial array of shape (number of q-values,)
        timestep = []
        for it in range(nbrBins):
            nbrFrames = minFrames + int(it * (maxFrames - minFrames) / nbrBins )
            
            #_Computes intermediate scattering function for one timestep, averaged over time origins
            corr = np.array( [(atomPos[:,i + nbrFrames] - atomPos[:,i]) 
                                                    for i in range(0, atomPos.shape[1] - nbrFrames)] )
            
            corr = qValList * np.mean(corr, axis=0) #_Get the exponent

            corr = np.exp( 1j * corr ) #_Exponentiation

            corr = np.mean( corr , axis=1 ) #_Averaging over atoms
            
            #_Add the intermediate function to the result array
            intermediateFunc = np.row_stack( (intermediateFunc, corr) )

            #_Compute time step
            timestep.append(self.timestep * nbrFrames * self.dcdFreq[0])

        return intermediateFunc[1:].T, timestep




    def getEISF(self, qValList, minFrames, maxFrames, nbrBins=100, resFunc=None, 
                                                selection='protNonExchH', begin=0, end=None):
        """ This method performs a multiplication of the inverse Fourier transform given resolution 
            function with the computed intermediate function to get the convoluted signal, 
            which can be used to compute MSD. 
            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    resFunc     -> resolution function to be used (optional, default resFuncSPHERES)
                    selection   -> atom selection (optional, default 'protein')
                    begin       -> first frame to be used (optional, default 0)
                    end         -> last frame to be used (optional, default None) """

        if resFunc == None:
            resFunc = self.parent.resFuncSPHERES

        #_Gets intermediate scattering function        
        intFunc, timesteps = self.getIntermediateFunc(qValList, minFrames, maxFrames, nbrBins, selection,
                                                                                        begin, end)

        #_Gets resolution function 
        resolution = resFunc(fftfreq(intFunc.shape[1]))

        #_Inverse Fourier transform of resolution
        resolution = ifft(resolution)

        eisf = resolution * intFunc #_Computes the EISF

        return eisf / eisf[:,0][:,np.newaxis], timesteps #_Returns the normalized EISF and time 




    def getScatteringFunc(self, qValList, minFrames, maxFrames, nbrBins=100, resFunc=None,
                                                        selection='protNonExchH', begin=0, end=None):
        """ This method calls getIntermediateFunc several times for different time steps, given by
            the number of frames, which will start from minFrames and by incremented to reach maxFrames
            in the given number of bins.

            Then, a Fourier transform is performed to compute the scattering function.

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    selection   -> atom selection (optional, default 'protein')
                    begin       -> first frame to be used (optional, default 0)
                    end         -> last frame to be used (optional, default None) """

        scatFunc, timesteps = self.getEISF(qValList, minFrames, maxFrames, nbrBins, resFunc, 
                                                                             selection, begin, end)

        #_Performs the Fourier transform
        scatFunc = fftshift( fft(scatFunc, axis=1), axes=1 ) / scatFunc.shape[1]

        #_Convert time to energies
        energies = fftshift( 6.582119514e-2 * 2 * np.pi * fftfreq(scatFunc.shape[1], d=1/nbrBins) )  

        return scatFunc, energies

        

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


    def plotEISF(self, qValList, minFrames, maxFrames, resFunc=None, nbrBins=100, selection='protNonExchH',
                                                                                    begin=0, end=None):
        """ This method calls self.getEISF to compute the Elastic Incoherent Structure Factor.
            Then, a plot is generated, showing self-correlation for each q-values in qValList. """

        EISF, times = self.getEISF(qValList, minFrames, maxFrames, nbrBins, resFunc, selection, begin, end)
        
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



    def plotIntermediateFunc(self, qValList, minFrames, maxFrames, nbrBins=100, selection='protNonExchH',
            begin=0, end=None):
        """ This method calls self.getIntermediateFunc to compute the intermediate scattering function.
            Then, a plot is generated, showing self-correlation for each q-values in qValList. """

        intF, times = self.getIntermediateFunc(qValList, minFrames, maxFrames, nbrBins, selection, begin, end)
        
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



    def plotScatteringFunc(self, qValList, minFrames, maxFrames, resFunc=None, nbrBins=100, 
                                                        selection='protNonExchH', begin=0, end=None):
        """ This method calls self.getScatteringFunc to compute the scattering function S(q,omega).
            Then, a plot is generated, showing self-correlation for each q-values in qValList. """

        scatF, energies = self.getScatteringFunc(qValList, minFrames, maxFrames, nbrBins, 
                                                                            resFunc, selection, begin, end)
        
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


