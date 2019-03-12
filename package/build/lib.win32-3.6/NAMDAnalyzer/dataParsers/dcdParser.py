import os, sys
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..lib.pygetWithin import py_getWithin
from ..lib.pygetCenterOfMass import py_getCenterOfMass
from ..lib.pysetCenterOfMassAligned import py_setCenterOfMassAligned

from collections import namedtuple

from ..dataManipulation import molFit_quaternions as molFit_q
from .dcdReader import DCDReader
from .psfParser import NAMDPSF



class NAMDDCD(DCDReader, NAMDPSF):
    """ This class contains methods for trajectory file analysis. 
        It's te second class to be called, after NAMDPSF.
        Here a dcd file is optional and can be added after initialization"""
    
    def __init__(self, psfFile, dcdFile=None, stride=1):
    
        NAMDPSF.__init__(self, psfFile)

        self.stride = stride
        self.COMAligned = False #_To check if center of mass were aligned

        DCDReader.__init__(self, stride)

        if dcdFile:
            self.importDCDFile(dcdFile)



    def getWithin(self, distance, usrSel, outSel=None, frame=-1, getSameResid=True):
        """ Selects all atoms that within the given distance of the given selection and frame.
    
            Input:  distance    -> distance in angstrom, within which to select atoms
                    usrSel      -> initial selection from which distance should be computed
                    outSel      -> specific selection for output. If not None, after all atoms within the
                                   given distance have been selected, the selected can be restricted
                                   further using a keyword or a list of indices. Only atoms that are
                                   present in the 'within' list and in the 'outSel' list are returned.
                    frame       -> frame number to be used for atom selection
                    getSameResid-> if True, select all atoms in the same residue before returning the list 

            Returns the list of selected atom indices. """


        #_Get the indices corresponding to the selection
        if type(usrSel) == str:
            usrSel = self.getSelection(usrSel)

        if type(outSel) == str:
            outSel = self.getSelection(outSel)

    
        frameAll    = np.ascontiguousarray(self.dcdData[:,frame], dtype='float32') #_Get frame coordinates

        usrSel = np.ascontiguousarray(self.dcdData[usrSel,frame], dtype='float32')

        #_Get cell dimensions for given frame and make array contiguous
        cellDims    = np.ascontiguousarray(self.cellDims[:,frame], dtype='float32')

        #_Initialize boolean array for atom selection
        keepIdx     = np.zeros(self.dcdData.shape[0], dtype=int)


        py_getWithin(frameAll, usrSel, cellDims, keepIdx, distance)


        if outSel is not None:
            #_Creates an array of boolean for logical_and
            outSelBool = np.zeros(self.dcdData.shape[0], dtype=bool) 
            outSelBool[outSel] = 1  #_Sets selected atoms indices to 1

            keepIdx = np.logical_and( outSelBool, keepIdx.astype(bool) )


        keepIdx = np.argwhere( keepIdx )[:,0]

        if getSameResid:
            keepIdx = self.getSameResidueAs(keepIdx)


        return keepIdx



    def getWithinCOM(self, distance, COM, outSel=None, frame=-1, getSameResid=False):
        """ Selects all atoms that within the given distance of the given selection and frame.
    
            Input:  distance    -> distance in angstrom, within which to select atoms
                    COM         -> Center of mass for the desired frame
                    outSel      -> specific selection for output. If not None, after all atoms within the
                                   given distance have been selected, the selected can be restricted
                                   further using a keyword or a list of indices. Only atoms that are
                                   present in the 'within' list and in the 'outSel' list are returned.
                    frame       -> frame number to be used for atom selection
                    getSameResid-> if True, select all atoms in the same residue before returning the list 

            Returns the list of selected atom indices. """


        #_Get the indices corresponding to the selection
        if type(outSel) == str:
            outSel = self.getSelection(outSel)

    
        frameAll    = np.ascontiguousarray(self.dcdData[:,frame], dtype='float32') #_Get frame coordinates

        usrSel      = np.ascontiguousarray(COM[np.newaxis,:], dtype='float32')

        #_Get cell dimensions for given frame and make array contiguous
        cellDims    = np.ascontiguousarray(self.cellDims[:,frame], dtype='float32')

        #_Initialize boolean array for atom selection
        keepIdx     = np.zeros(self.dcdData.shape[0], dtype=int)


        py_getWithin(frameAll, usrSel, cellDims, keepIdx, distance)


        if outSel is not None:
            #_Creates an array of boolean for logical_and
            outSelBool = np.zeros(self.dcdData.shape[0], dtype=bool) 
            outSelBool[outSel] = 1  #_Sets selected atoms indices to 1

            keepIdx = np.logical_and( outSelBool, keepIdx.astype(bool) )


        keepIdx = np.argwhere( keepIdx )[:,0]

        if getSameResid:
            keepIdx = self.getSameResidueAs(keepIdx)


        return keepIdx





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
 



    def getCenterOfMass(self, selection, begin=0, end=None, step=1):
        """ Computes the center of mass for selected atoms and frames. """

        if type(selection) == str:
            selection = self.getSelection(selection)

        try: #_Check if a psf file has been loaded
            atomMasses = self.getAtomsMasses(selection)

        except AttributeError:
            print("No .psf file was loaded, please import one before using this method.")
            return

        atomMasses = atomMasses.reshape(atomMasses.size, 1)

        centerOfMass = py_getCenterOfMass(self.dcdData[selection,begin:end:step], atomMasses)


        return centerOfMass




    def getAlignedData(self, selection, begin=0, end=None, step=1):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Returns a similar array as the initial dataSet but with aligned coordinates."""

        if type(selection) == str:
            selection = self.getSelection(selection)

        alignData = self.getAlignedCenterOfMass(selection, begin, end, step)

        alignData = molFit_q.alignAllMol(alignData)
        
        return alignData




    def getAlignedCenterOfMass(self, selection='all', begin=0, end=None, step=1):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be used for center of mass
            drift corrections if no global angular momentum is present. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)


        dataSet = np.copy( self.dcdData[selection, begin:end:step] )

        centerOfMass = self.getCenterOfMass(selection, begin, end, step)[np.newaxis]

        #_Substract the center of mass coordinates to each atom for each frame
        dataSet = dataSet - centerOfMass

        return dataSet





    def setCenterOfMassAligned(self, selection='all', begin=0, end=None, step=1):
        """ Modifies the dcd data by aligning center of mass of all atoms between given frames delimited by
            begin and end keywords. """

        print("\nAligning center of mass of all molecules...\n")

        centerOfMass = self.getCenterOfMass(selection, begin, end, step)

        #_Substract the center of mass coordinates to each atom for each frame
        py_setCenterOfMassAligned(self.dcdData[selection,begin:end:step], centerOfMass)


        self.COMAligned = True


        print("Done\n")




    def getCOMRadialDistribution(self, selection='protH', dr=1, maxR=60, frame=-1):
        """ Computes the radial density distribution from center of mass of selected atoms 
            using the given dr interval.

            Input:  selection   ->  atom selection, can be string
                    dr          ->  radius interval, density is computed as the number of atoms between r and
                                    r + dr divided by the volume integral in spherical coordinates for unit r
                                    times the total number of atoms within maximum r
                    maxR        ->  maximum radius to be used
                    frame       ->  frame to be used for atom coordinates """


        if type(selection) == str:
            selection = self.getSelection(selection)


        COM = self.getCenterOfMass(selection, frame, frame+1)
        dist = np.arange(dr, maxR, dr)
        totalAtoms = self.getWithinCOM(maxR, COM, selection, frame).size 

        nbrAtoms = []
        for r in dist:
            nbrAtoms.append( self.getWithinCOM(r, COM, selection, frame).size )

        nbrAtoms = np.array( nbrAtoms )
        density  = 1 / (4*np.pi*dist**2*dr) * np.insert( (nbrAtoms[1:] - nbrAtoms[:-1]), 0, nbrAtoms[0] )

        return dist, density



#---------------------------------------------
#_Plotting methods
#---------------------------------------------

    def plotSTDperAtom(self, selection="all", align=False, begin=0, end=None, mergeXYZ=True):
        """ Plot the standard deviation along the axis 0 of dataSet.
        nbrAtoms = 
        nbrAtoms = 
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




    def plotCOMRadialDistribution(self, selection='protH', dr=1, maxR=60, frame=-1):
        """ Plot the radial distribution of selected atoms from their center of mass.
            
            Calls the self.getRadialDistribution method to obtain data to plot """

        X, density = self.getCOMRadialDistribution(selection, dr, maxR, frame)


        fig, ax = plt.subplots()

        ax.plot(X, density)
        ax.set_xlabel(r'r [$\AA$]')
        ax.set_ylabel(r'Density $\rho(r)$')

        plt.tight_layout()

        return plt.show(block=False)

