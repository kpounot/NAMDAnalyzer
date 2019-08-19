import os, sys
import numpy as np
import re

from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation as R

from ..lib.pylibFuncs import (  py_getWithin, 
                                py_getCenterOfMass,
                                py_setCenterOfMassAligned,
                                py_getDistances )


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

        self.COMAligned = False #_To check if center of mass were aligned


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



#---------------------------------------------
#_Distances and within selections
#---------------------------------------------
    def getDistances(self, sel1=None, sel2=None, frame=-1):
        """ Selects all atoms that within the given distance of the given selection and frame.
    
            Input:  sel1    -> first selection of atoms used for distance calculation with sel2 (default -> all)
                    sel2    -> second selection for distance calculation with sel1 (default -> all)
                    frame   -> frame to be used for computation

            Returns a matrix containing pairwise distances if memory allows it. """


        #_Get the indices corresponding to the selection
        if type(sel1) == str:
            if re.search('within', sel1):
                sel1 + ' frame %i' % frame
            sel1 = self.selection(sel1)

        if sel1 is None:
            sel1 = self.getSelection()

        if type(sel2) == str:
            if re.search('within', sel2):
                sel2 + ' frame %i' % frame
            sel2 = self.selection(sel2)

        if sel2 is None:
            sel2 = self.getSelection()


        #_Gets atom coordinates
        sel1 = self.dcdData[sel1,frame]
        sel2 = self.dcdData[sel2,frame]


        out = np.zeros((sel1.shape[0], sel2.shape[0]), dtype='float32')


        #_Get cell dimensions for given frame and make array contiguous
        cellDims    = np.ascontiguousarray(self.cellDims[frame], dtype='float32')


        py_getDistances(sel1, sel2, cellDims, out)


        return out





    def getWithin(self, distance, usrSel, frame=-1):
        """ Selects all atoms that within the given distance of the given selection and frame.
    
            Input:  distance    -> distance in angstrom, within which to select atoms
                    usrSel      -> initial selection from which distance should be computed
                    frame       -> frame to be used for atom selection

            Returns an array of boolean, set to 1 for each selected atom in simulation in each 
            selected frame. If the second dimension of the array is one, the output is flattened, and
            atom indices are returned directly. """


        #_Get the indices corresponding to the selection
        if type(usrSel) == str:
            usrSel = self.selection(usrSel)

        if type(frame) == int:
            frame = [frame]
            nbrFrames = 1
        elif isinstance(frame, slice):
            step = 1 if frame.step == None else frame.step
            nbrFrames = int( (frame.stop - frame.start) / step ) #_frame.stop is not included
        else:
            nbrFrames = len(frame)

    
        frameAll    = np.ascontiguousarray(self.dcdData[:,frame], dtype='float32') 

        #_Get cell dimensions for given frame and make array contiguous
        cellDims    = np.ascontiguousarray(self.cellDims[frame], dtype='float32')

        #_Initialize boolean array for atom selection
        keepIdx = np.zeros( (self.nbrAtoms, nbrFrames), dtype=int)

        py_getWithin(frameAll, usrSel.astype('int32'), cellDims, keepIdx, distance)

        if keepIdx.shape[1] == 1:
            keepIdx = keepIdx.flatten()
            keepIdx = np.argwhere( keepIdx )[:,0]


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
        cellDims    = np.ascontiguousarray(self.cellDims[frame], dtype='float32')

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



    def setCenterOfMassAligned(self, selection='all', frames=slice(0,None)):
        """ Modifies the dcd data by aligning center of mass of all atoms between given frames. 

            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object
        
        """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        print("\nAligning center of mass of selected atoms...\n")

        centerOfMass = self.getCenterOfMass(selection, frames)

        dcdData = self.dcdData[selection, frames]
        if dcdData.ndim == 2:
            dcdData = dcdData[:,np.newaxis,:]

        #_Substract the center of mass coordinates to each atom for each frame
        py_setCenterOfMassAligned(dcdData, centerOfMass)

        self.dcdData[selection, frames] = dcdData


        print("Done\n")




    def setAlignedData(self, selection, outSel='all', frames=slice(0, None)):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Input:  selection   -> either string or array of indices, will be used for fitting
                    outSel      -> either string or array of indices, will be used to apply rotation
                    frames      -> either None to select all frames, an int, or a slice object

            Returns a similar array as the initial dataSet but with aligned coordinates."""

        if type(selection) == str:
            selection = self.selection(selection)

        if type(outSel) == str:
            outSel = self.selection(outSel)


        print("\nAligning selected atoms and frames...\n")


        self.setAlignedCenterOfMass(selection, frames)

        q = molFit_q.alignAllMol(self.dcdData[selection, frames])
        
        self.dcdData[outSel, frames] = molFit_q.applyRotation(self.dcdData[outSel, frames], q)

        print("Done\n")



    def rotate(self, rotVec, selection, frames=slice(0, None)):
        """ This method allows to rotate the given selection using the angle/axis representation
            given by rotVec, whose coordinates represent the axis of rotation and norm gives
            the rotation magnitude in radians. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        norm    = np.sqrt( np.sum(rotVec**2) )
        x       = rotVec[0] / norm
        y       = rotVec[1] / norm
        z       = rotVec[2] / norm
        print("\nRotating selection along axis (%f, %f, %f) with angle %f rad...\n" % (x, y, z, norm))

        r = R.from_rotvec(rotVec)

        q = r.as_quat()

        self.dcdData[selection, frames] = molFit_q.applyRotation(self.dcdData[selection, frames], q)


        print("Done\n")




        
#---------------------------------------------
#_Data analysis methods
#---------------------------------------------
    def getSTDperAtom(self, selection="all", align=False, frames=slice(0, None), mergeXYZ=True):
        """ Computes the standard deviation for each atom in selection and for frames between
            begin and end. 
            Returns the standard deviation averaged over time.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   end       -> last frame to be used + 1
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

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


    def getRMSDperAtom(self, selection="all", align=False, frames=slice(0, None), mergeXYZ=True):
        """ Computes the RMSD for each atom in selection and for frames between begin and end.
            Returns the RMSD averaged over time.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   frames      -> either None to select all frames, an int, or a slice object
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

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


    def getRMSDperFrame(self, selection="all", align=False, frames=slice(0, None), mergeXYZ=True):
        """ Computes the RMSD for each atom in selection and for frames between begin and end.
            Returns the RMSD averaged over all selected atoms.

            Input: selection -> selected atom, can be a single string or a member of parent's selList
                   align     -> if True, will try to align all atoms to the ones on the first frame
                   frames      -> either None to select all frames, an int, or a slice object
                   mergeXYZ  -> if True, uses the vector from the origin instead of each projections """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

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
 



    def getCenterOfMass(self, selection, frames=slice(0,None)):
        """ Computes the center of mass for selected atoms and frames. 
        
            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object """

        if type(selection) == str:
            selection = self.selection(selection)

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




    def getAlignedData(self, selection, frames=slice(0, None)):
        """ This method will fit all atoms between firstAtom and lastAtom for each frame between
            begin and end, using the first frame for the others to be fitted on. 
            
            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object

            Returns a similar array as the initial dataSet but with aligned coordinates."""

        if type(selection) == str:
            selection = self.selection(selection)

        alignData = self.getAlignedCenterOfMass(selection, frames)

        alignData = molFit_q.alignAllMol(alignData)
        
        return alignData




    def getAlignedCenterOfMass(self, selection='all', frames=slice(0,None)):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be used for center of mass
            drift corrections if no global angular momentum is present. 

            Input:  selection   -> either string or array of indices for selected atoms
                    frames      -> either None to select all frames, an int, or a slice object
            """


        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)


        dataSet = np.copy( self.dcdData[selection, frames] )

        centerOfMass = self.getCenterOfMass(selection, frames)

        #_Substract the center of mass coordinates to each atom for each frame
        dataSet = dataSet - centerOfMass

        return dataSet







    def getRadialNumberDensity(self, sel1, sel2, dr=0.1, maxR=10, frames=slice(0, None)):
        """ Computes the radial density distribution from center of mass of selected atoms 
            using the given dr interval.

            Input:  sel1    -> first atom selection from which spherical zone will be computed
                    sel2    -> second selection, only atoms within the spherical zone and corresponding
                                to this selection will be considered
                    dr      ->  radius interval, density is computed as the number of atoms between r and
                                r + dr divided by the total number of sel2 atoms within maxR
                    maxR    ->  maximum radius to be used
                    frames  ->  frames to be averaged on, should be a range 
                                                                (default None, every 10 frames are used) """


        radii  = np.arange(dr, maxR, dr) #_Gets x-axis values

        if not frames:
            frames = np.arange(0, self.nbrFrames, 10)

        frames = np.array(frames)

        density = np.zeros( radii.size, dtype='float32' )
        
        
        for frameId, frame in enumerate(frames):
            print('Processing frame %i of %i...' % (frameId+1, len(frames)), end='\r')

            dist = self.getDistances(sel1, sel2, frame).flatten()
            dist = dist[dist > 0]

            for rIdx, r in enumerate(radii[::-1]):
                dist = dist[dist < r]
                density[-(rIdx+1)] += dist.size / len(frames)
                
            
        density /= ( 4 * np.pi * radii**2 * dr )


        density[1:] = density[1:] - density[:-1]

        return radii, density / np.sum(density)




    def getCOMRadialNumberDensity(self, selection='protH', dr=0.5, maxR=60, frame=-1):
        """ Computes the radial density distribution from center of mass of selected atoms 
            using the given dr interval.

            Input:  selection   ->  atom selection, can be string
                    dr          ->  radius interval, density is computed as the number of atoms between r and
                                    r + dr divided by the volume integral in spherical coordinates for unit r
                                    times the total number of atoms within maximum r
                    maxR        ->  maximum radius to be used
                    frame       ->  frame to be used for atom coordinates """


        if type(selection) == str:
            selection = self.selection(selection)


        radii = np.arange(0, maxR, dr) #_Gets x-axis values
        density = np.zeros( radii.size, dtype='float32' )

        #_Set center of mass to the origin and computes distances from origin for all atoms
        dist = self.getAlignedCenterOfMass(selection, frame)
        dist = np.sqrt( np.dot(dist, dist.T) ).flatten()
        dist = dist[dist > 0]

        for rIdx, r in enumerate(radii[::-1]):
            dist = dist[dist < r]
            density[-(rIdx+1)] += dist.size 
        
        density /= ( 4 * np.pi * radii**2 * dr )
        density[1:] = density[1:] - density[:-1]

        return radii, density



#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotSTDperAtom(self, selection="all", align=False, frames=slice(0, None), mergeXYZ=True):
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




    def plotRMSDperAtom(self, selection="all", align=False, frames=slice(0, None), mergeXYZ=True):
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




    def plotRMSDperFrame(self, selection="all", align=False, frames=slice(0, None), mergeXYZ=True):
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

