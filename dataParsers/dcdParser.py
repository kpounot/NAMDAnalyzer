import os, sys
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy.io import FortranFile

from collections import namedtuple

sys.path.append('../')
import dataManipulation.molFit as molFit


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
        self.dcdFreq    = header[3]
        self.nbrSteps   = header[4]

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
        #_and (x, y, z) coordinates in axis 3
        self.dataSet = self.dataSet.reshape(self.dataSet.shape[0], self.nbrFrames, 3)

        
    #---------------------------------------------
    #_Data analysis methods
    #---------------------------------------------

    def getSTDperAtom(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Computes the standard deviation for each atom in selection and for frames between
            begin and end.

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
            centerOfMass = self.getCenterOfMass(selection, begin, end)
            data         = np.copy(self.dataSet[selection, begin:end])   
            data         = molFit.alignAllMol(data, centerOfMass)

        else:
            data = self.dataSet[selection, begin:end]

        #_Computes the standard deviation
        if mergeXYZ:
            std = np.sqrt(np.sum(data**2, axis=2))
            std = np.std(std, axis=1)

        else:
            std = np.std(data, axis=1)

        return std


    def getCenterOfMass(self, selection="all", begin=0, end=None):
        """ Computes the center of mass of the system for atoms between firstAtom and lastATom,
            and for frames between begin and end. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = parent.psfData.getSelection(selection)

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

        alignData = molFit.alignAllMol(alignData, centerOfMass)
        
        return alignData

        

    #---------------------------------------------
    #_Plotting methods
    #---------------------------------------------

    def plotSTDperAtom(self, selection, align=False, begin=0, end=None, mergeXYZ=True):
        """ Plot the standard deviation along the axis 1 of dataSet.
            This makes use of the 'getSTDperAtom method.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        rmsd = self.getSTDperAtom(selection, align, begin, end, mergeXYZ)
        xRange = np.arange(rmsd.shape[0])

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
        plt.show(block=False)
