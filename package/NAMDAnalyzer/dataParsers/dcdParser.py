import os, sys
import numpy as np
import re

from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..lib.pygetWithin import py_getWithin
from ..lib.pygetCenterOfMass import py_getCenterOfMass
from ..lib.pysetCenterOfMassAligned import py_setCenterOfMassAligned


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

        DCDReader.__init__(self, stride)

        if dcdFile:
            self.importDCDFile(dcdFile)



    def appendCoordinates(self, coor):
        """ Can be used to append a frame with coordinates from a pdb file.

            coor -> 2D array containing 3D coordinates for each atom. """

        try:
            np.append( self.dcdData, coor[:,np.newaxis,:].astype('float32'), axis=1 )
            np.append( self.dcdFreq, self.dcdFreq[-1] )
            np.append( self.cellDims, self.cellDims[-1] )
            self.nbrFrames += 1
            self.nbrSteps  += 1


        except: #_If no trajectories were loaded, just create one frame and use default values
            self.dcdData    = coor[:,np.newaxis,:].astype('float32')
            self.dcdFreq    = np.array( [1] )
            self.timestep   = 2
            self.nbrFrames  = 1
            self.nbrSteps   = 1
            self.nbrAtoms   = coor.shape[0]
            self.cellDims   = np.array( [[3*np.max(coor[:,0]), 3*np.max(coor[:,1]), 3*np.max(coor[:,2])]] ) 



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

        usrSel      = np.ascontiguousarray(COM, dtype='float32')

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
    def getSTDperAtom(self, selection="all", align=False, frames=None, mergeXYZ=True):
        """ Computes the standard deviation for each atom in selection and for frames between
            begin and end. 
            Returns the standard deviation averaged over time.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   end       -> last frame to be used + 1
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, frames)
        else:
            data = self.dcdData[selection, frames]

        #_Computes the standard deviation
        if mergeXYZ:
            std = np.sqrt(np.sum(data**2, axis=2))
            std = np.std(std, axis=1)

        else:
            std = np.std(data, axis=1)

        return std


    def getRMSDperAtom(self, selection="all", align=False, frames=None, mergeXYZ=True):
        """ Computes the RMSD for each atom in selection and for frames between begin and end.
            Returns the RMSD averaged over time.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   frames      -> either None to select all frames, an int, or a slice object
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, frames)
        else:
            data = self.dcdData[selection, frames]

        #_Computes the standard deviation
        if mergeXYZ:
            rmsd = np.sqrt(np.sum(data**2, axis=2))

        
        rmsd = np.apply_along_axis(lambda arr: (arr - rmsd[:,0])**2, 0, rmsd)
        rmsd = np.mean(rmsd, axis=1)

        return rmsd


    def getRMSDperFrame(self, selection="all", align=False, frames=None, mergeXYZ=True):
        """ Computes the RMSD for each atom in selection and for frames between begin and end.
            Returns the RMSD averaged over all selected atoms.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   frames      -> either None to select all frames, an int, or a slice object
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        #_Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, frames)
        else:
            data = self.dcdData[selection, frames]

        #_Computes the standard deviation
        if mergeXYZ:
            rmsd = np.sqrt(np.sum(data**2, axis=2))

        
        rmsd = np.apply_along_axis(lambda arr: (arr - rmsd[:,0])**2, 0, rmsd)
        rmsd = np.mean(rmsd, axis=0)

        return rmsd
 



    def getCenterOfMass(self, selection, frames=None):
        """ Computes the center of mass for selected atoms and frames. 
        
            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object """

        if type(selection) == str:
            selection = self.getSelection(selection)

        try: #_Check if a psf file has been loaded
            atomMasses = self.getAtomsMasses(selection)

        except AttributeError:
            print("No .psf file was loaded, please import one before using this method.")
            return

        atomMasses = atomMasses.reshape(atomMasses.size, 1)

        dcdData = self.dcdData[selection, frames]
        if dcdData.ndim == 2:
            dcdData = dcdData[:,np.newaxis,:]

        centerOfMass = py_getCenterOfMass(dcdData, atomMasses)


        return centerOfMass




    def getAlignedData(self, selection, frames=None):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object

            Returns a similar array as the initial dataSet but with aligned coordinates."""

        if type(selection) == str:
            selection = self.getSelection(selection)

        alignData = self.getAlignedCenterOfMass(selection, frames)

        alignData = molFit_q.alignAllMol(alignData)
        
        return alignData




    def getAlignedCenterOfMass(self, selection='all', frames=None):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be used for center of mass
            drift corrections if no global angular momentum is present. 

            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object
            """


        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)


        dataSet = np.copy( self.dcdData[selection, frames] )

        centerOfMass = self.getCenterOfMass(selection, frames)

        #_Substract the center of mass coordinates to each atom for each frame
        dataSet = dataSet - centerOfMass

        return dataSet





    def setCenterOfMassAligned(self, selection='all', frames=None):
        """ Modifies the dcd data by aligning center of mass of all atoms between given frames. 

            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object
        
        """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        print("\nAligning center of mass of selected atoms...\n")

        centerOfMass = self.getCenterOfMass(selection, frames)

        dcdData = self.dcdData[selection, frames]
        if dcdData.ndim == 2:
            dcdData = dcdData[:,np.newaxis,:]

        #_Substract the center of mass coordinates to each atom for each frame
        py_setCenterOfMassAligned(dcdData, centerOfMass)

        self.dcdData[selection, frames] = dcdData


        print("Done\n")




    def getCOMRadialNumberDensity(self, selection='protH', dr=1, maxR=60, frame=-1):
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


        dist = np.arange(0, maxR, dr) #_Gets x-axis values

        #_Set center of mass to the origin and computes distances from origin for all atoms
        coor = self.getAlignedCenterOfMass(selection, frame)
        coor = np.sqrt( np.dot(coor, coor.T) )

        density = []
        for i, r in enumerate(dist):
            density.append( coor[ coor < r ].size )
            density[i] -= np.sum( density[:i] )
        
        density = np.array( density ) / coor.size

        return dist, density



#---------------------------------------------
#_Plotting methods
#---------------------------------------------

    def plotSTDperAtom(self, selection="all", align=False, frames=None, mergeXYZ=True):
        """ Plot the standard deviation along the axis 0 of dataSet.
            This makes use of the 'getSTDperAtom method.

            If mergeXYZ is True, then computes the distance to the origin first. """

        std = self.getSTDperAtom(selection, align, frames, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[frames])

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


    def plotRMSDperAtom(self, selection="all", align=False, frames=None, mergeXYZ=True):
        """ Plot the RMSD along the axis 0 of dataSet.
            This makes use of the 'getRMSDperAtom method.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        rmsd = self.getRMSDperAtom(selection, align, frames, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[frames])

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


    def plotRMSDperFrame(self, selection="all", align=False, frames=None, mergeXYZ=True):
        """ Plot the RMSD along the axis 1 of dataSet.
            This makes use of the 'getRMSDperFrame method.

            If mergeXYZ is True, then it computes the distance to the origin first. """

        rmsd = self.getRMSDperFrame(selection, align, frames, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[frames])

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




    def plotCOMRadialNumberDensity(self, selection='protH', dr=1, maxR=60, frame=-1):
        """ Plot the radial distribution of selected atoms from their center of mass.
            
            Calls the self.getRadialDistribution method to obtain data to plot """

        X, density = self.getCOMRadialNumberDensity(selection, dr, maxR, frame)


        fig, ax = plt.subplots()

        ax.plot(X, density)
        ax.set_xlabel(r'r [$\AA$]')
        ax.set_ylabel(r'Density $\rho(r)$')

        plt.tight_layout()

        return plt.show(block=False)

