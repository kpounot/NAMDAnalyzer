import os, sys
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

from threading import Thread, RLock


from collections import namedtuple

from ..dataManipulation import molFit_quaternions as molFit_q
from .dcdReader import DCDReader



class NAMDDCD(DCDReader):
    """ This class contains methods for trajectory file analysis. """
    
    def __init__(self, parent, dcdFile=None):
    
        self.parent = parent

        DCDReader.__init__(self)

        if dcdFile:
            self.importDCDFile(dcdFile)



    def getWithin(self, distance, usrSel, outSel=None, frame=-1, batchSize=2000):
        """ Selects all atoms that within the given distance of the given selection and frame.
    
            Input:  distance    -> distance in angstrom, within which to select atoms
                    selection   -> initial selection from which distance should be computed
                    outSel      -> specific selection for output. If not None, after all atoms within the
                                   given distance have been selected, the selected can be restricted
                                   further using a keyword or a list of indices. Only atoms that are
                                   present in the 'within' list and in the 'outSel' list are returned.
                    frame       -> frame number to be used for atom selection

            Returns the list of selected atom indices. """


        #_Get the indices corresponding to the selection
        if type(usrSel) == str:
            usrSel = self.parent.getSelection(usrSel)

        if type(outSel) == str:
            outSel = self.parent.getSelection(outSel)

    
        distance = distance**2


        usrSel      = self.dcdData[usrSel,frame]        #_Get initial atom selection
        frameAll    = self.dcdData[:,frame]             #_Get all atom in given frame
        keepIdx     = np.zeros(self.dcdData.shape[0])   #_Initialize boolean array for atom selection


        pList = []
        batchIdx = np.arange(0, frameAll.shape[0], batchSize)
        for idx in batchIdx:
            vecDist = frameAll[idx:idx+batchSize]

            pList.append( Thread(target=self.kernelWithin, args=(vecDist, usrSel, distance, keepIdx, idx)) )

            try:
                pList[-1].start()
            except MemoryError:
                print("Memory Error: resulting array too big, try to reduce batchSize" 
                                                                         + " or use smaller selection\n")


        for p in pList:
            p.join()
            


        if outSel is not None:
            outSelBool = np.zeros(self.dcdData.shape[0]) #_Creates an array of boolean for logical_and
            outSelBool[outSel] = 1  #_Sets selected atoms indices to 1

            keepIdx = np.logical_and( outSelBool, keepIdx )


        return np.argwhere( keepIdx )[:,0]




    def kernelWithin(self, M0, M1, distance, outVec, idx):

        vecDist = np.sum( (M0[:,np.newaxis,:] - M1)**2, axis=2 )

        outVec[ np.unique( np.argwhere(vecDist <= distance)[:,0] + idx ) ] = 1






#---------------------------------------------
#_Data modifiers
#---------------------------------------------
    def binDCD(self, binSize):
        """ Binning method for dcd data. The binning is performed along the axis 1 of the dataset,
            which corresponds to frames dimension. """

        if binSize==1:
            return

        print('Binning trajectories...')

        nbrLoops = int(self.dcdData.shape[1] / binSize)

        #_Performs the binning
        for i in range(nbrLoops):
            self.dcdData[:,i] = self.dcdData[:, i*binSize : i*binSize+binSize].mean(axis=1)
            self.dcdFreq[i] = np.sum( self.dcdFreq[i*binSize : i*binSize+binSize] )

        #_Free the memory
        self.dcdData = np.delete(self.dcdData, range(nbrLoops,self.dcdData.shape[1]), axis=1)
        self.dcdFreq = np.delete(self.dcdFreq, range(nbrLoops,self.dcdFreq.size))

        print('Done\n')


        
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
            selection = self.parent.getSelection(selection)

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
            selection = self.parent.getSelection(selection)

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
            selection = self.parent.getSelection(selection)

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
                selection = self.parent.getSelection(selection)

            atomMasses = self.parent.getAtomsMasses(selection)

        except AttributeError:
            print("No .psf file was loaded, please import one before using this method.")
            return

        atomMasses = atomMasses.reshape(1, atomMasses.size, 1)

        centerOfMass = np.dot(self.dcdData[selection,begin:end].T, atomMasses).T

        centerOfMass = np.sum(centerOfMass, axis=0) / np.sum(atomMasses) #_Summing over weighed atoms

        return centerOfMass


    def getAlignedData(self, selection, begin=0, end=None):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Returns a similar array as the initial dataSet but with aligned coordinates."""

        if type(selection) == str:
            selection = self.parent.getSelection(selection)

        centerOfMass = self.getCenterOfMass(selection, begin, end)
        alignData    = np.copy(self.dcdData[selection,begin:end])   

        alignData = molFit_q.alignAllMol(alignData, centerOfMass)
        
        return alignData



    def alignCenterOfMass(self, selection='all', begin=0, end=None):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be used for center of mass
            drift corrections if no global angular momentum is present. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.getSelection(selection)

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

