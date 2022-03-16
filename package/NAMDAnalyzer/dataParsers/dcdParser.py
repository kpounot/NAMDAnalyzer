"""

Classes
^^^^^^^

"""

import os
import sys
import numpy as np
import re

from collections import namedtuple

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation as R

try:
    from NAMDAnalyzer.lib.pylibFuncs import (
        py_getWithin,
        py_getDistances,
        py_cdf,
        py_getParallelBackend)
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")



from NAMDAnalyzer.dataManipulation import molFit_quaternions as molFit_q
from NAMDAnalyzer.dataParsers.dcdReader import DCDReader
from NAMDAnalyzer.dataParsers.psfParser import NAMDPSF
from NAMDAnalyzer.dataParsers.pdbParser import NAMDPDB

from NAMDAnalyzer.helpersFunctions.DistanceChordDiagram import ChordDiag

from NAMDAnalyzer.kdTree.getWithin_kdTree import getWithin_kdTree



class NAMDDCD(DCDReader, NAMDPSF):
    """ This class contains methods for trajectory file analysis.

        It's the second class to be called, after NAMDPSF.
        Here a dcd file is optional and can be added after initialization

        :arg psfFile: NAMD .psf file to be loaded
        :arg dcdFile: NAMD .dcd file to be used

    """

    def __init__(self, psfFile, dcdFile=None):

        NAMDPSF.__init__(self, psfFile)

        DCDReader.__init__(self)

        if dcdFile:
            self.importDCDFile(dcdFile)


        try:
            self.parallelBackend = py_getParallelBackend()
        except NameError:
            self.parallelBackend = 0





    def appendPDB(self, pdbFile):
        """ Can be used to append a frame with coordinates from a pdb file.

            Can be used even if no .dcd file was loaded.

            :arg pdbFile: pdb file path

        """

        pdb = NAMDPDB(pdbFile)

        self.dcdFiles.append(pdb.getCoor()[:, np.newaxis, :])

        if len(self.dcdFiles) == 1:
            self.nbrFrames += 1
            self.nbrSteps  += 1
            self.nbrAtoms   = pdb.atomList[0].shape[0]
            self.timestep   = 1
            self.dcdFreq    = np.append(self.dcdFreq, 1)
            self.initFrame.append(0)
            self.stopFrame.append(1)

        else:
            self.nbrFrames += 1
            self.nbrSteps  += 1
            self.dcdFreq    = np.append(self.dcdFreq, 1)
            self.initFrame.append(self.stopFrame[-1])
            self.stopFrame.append(self.stopFrame[-1] + 1)




# --------------------------------------------
# Distances and within selections
# --------------------------------------------
    def getDistances(self, sel1, sel2=None, frame=0):
        """ Computes pair-wise distances between sel1 and sel2.

            :arg sel1:  first selection of atoms used for distance
                        calculation with sel2 (default -> all)
            :arg sel2:  second selection for distance calculation
                        with sel1 (default -> all)
            :arg frame: frame to be used for computation, if more than one
                        frame is used, the returned array contains distances
                        averaged over all selected frames.

            :returns: a matrix containing pairwise distances, if memory
                      allows it, with sel1 being arranged row-wise and
                      sel2 column-wise.

        """


        sameSel = 0

        if isinstance(sel1, str) and isinstance(sel2, str):
            if set(sel1.split(' ')) == set(sel2.split(' ')):
                sameSel = 1

        # Get the indices corresponding to the selection
        if type(sel1) == str:
            if re.search('within', sel1):
                sel1 + ' frame %i' % frame

            sel1 = self.selection(sel1)

        if type(sel2) == str:
            if sameSel:
                sel2 = sel1
            else:
                if re.search('within', sel2):
                    sel2 + ' frame %i' % frame

                sel2 = self.selection(sel2)

        if sel2 is None:
            sel2 = sel1
            sameSel = 1

        out = np.zeros((sel1.shape[0], sel2.shape[0]), dtype='float32')

        cellDims = np.ascontiguousarray(self.cellDims[frame])
        sel1 = self.dcdData[sel1, frame]
        sel2 = self.dcdData[sel2, frame]

        py_getDistances(sel1, sel2, out, cellDims, sameSel)


        return out




    def getWithin(self, distance, refSel, outSel='all', frame=0):
        """ Selects all atoms that within the given distance of the
            given selection and frame.

            :arg distance: distance in angstrom, within which to select atoms
            :arg refSel:   initial selection from which distance should
                           be computed
            :arg outSel:   atoms to keep in output, all others won't
                           be considered for computation
            :arg frame:    frame to be used for atom selection, can be
                           str, int, range or slice

            :returns: an array of boolean, set to 1 for each selected atom in
                      simulation in each selected frame. If the second
                      dimension of the array is one, the output is flattened,
                      and atom indices are returned directly.

        """


        # Get the coordinates corresponding to the selection
        if type(refSel) == str:
            refSel = self.selection(refSel)

        # Gets coordinates for atoms to be used as reference
        if type(outSel) == str:
            outSel = self.selection(outSel)

        cellDims = self.cellDims[frame]

        allAtoms = self.dcdData[:, frame]

        # Initialize boolean array for atom selection
        keepIdx = np.zeros((allAtoms.shape[0], allAtoms.shape[1]),
                           dtype='int32')


        if self.parallelBackend == 2 or outSel.size <= 10000:
            py_getWithin(allAtoms, refSel.astype('int32'),
                         outSel.astype('int32'),
                         keepIdx, cellDims, distance)
        else:
            getWithin_kdTree(allAtoms, refSel.astype('int32'),
                             outSel.astype('int32'),
                             keepIdx, cellDims, distance)


        if keepIdx.shape[1] == 1:
            keepIdx = keepIdx.flatten()
            keepIdx = np.argwhere(keepIdx)[:, 0]


        return keepIdx





    def getWithinCOM(self, distance, COM, outSel=None, frame=0,
                     getSameResid=False):
        """ Selects all atoms that within the given distance of the given
            selection and frame.

            :arg distance:     distance in angstrom, within which to
                               select atoms
            :arg COM:          Center of mass for the desired frame
            :arg outSel:       specific selection for output. If not None,
                               after all atoms within the given distance have
                               been selected, the selected can be restricted
                               further using a keyword or a list of indices.
                               Only atoms that are present in the 'within'
                               list and in the 'outSel' list are returned.
            :arg frame:        frame number to be used for atom selection
            :arg getSameResid: if True, select all atoms in the same residue
                               before returning the list

            :returns: list of selected atom indices.

        """


        # Get the indices corresponding to the selection
        if type(outSel) == str:
            outSel = self.getSelection(outSel)


        frameAll = np.ascontiguousarray(
            self.dcdData[:, frame], dtype='float32')

        usrSel   = np.ascontiguousarray(COM, dtype='float32')

        # Initialize boolean array for atom selection
        keepIdx  = np.zeros(self.dcdData.shape[0], dtype='int32')


        py_getWithin(frameAll, usrSel, keepIdx, distance)


        if outSel is not None:
            # Creates an array of boolean for logical_and
            outSelBool = np.zeros(self.dcdData.shape[0], dtype=bool)
            outSelBool[outSel] = 1  # Sets selected atoms indices to 1

            keepIdx = np.logical_and(outSelBool, keepIdx.astype(bool))


        keepIdx = np.argwhere(keepIdx)[:, 0]

        if getSameResid:
            keepIdx = self.getSameResidueAs(keepIdx)


        return keepIdx




# --------------------------------------------
# Special accessors and data modifiers
# --------------------------------------------
    def rotate(self, rotVec, selection, frames=slice(0, None)):
        """ This method allows to rotate the given selection using the
            angle/axis representation given by rotVec, whose coordinates
            represent the axis of rotation and norm gives the rotation
            magnitude in radians.

            :arg rotVec:    a rotation vector in 3D cartesian coordinates
                            as described above
            :type rotVec:   np.ndarray
            :arg selection: either string or array of indices, will
                            be used for fitting
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        norm = np.sqrt(np.sum(rotVec**2))
        x    = rotVec[0] / norm
        y    = rotVec[1] / norm
        z    = rotVec[2] / norm
        print("\nRotating selection along axis (%f, %f, %f) "
              "with angle %f rad...\n" % (x, y, z, norm))

        r = R.from_rotvec(rotVec)

        q = molFit_q.get_bestMatrix(r.as_quat())

        return molFit_q.applyRotation(self.dcdData[selection, frames], q)



    def getCenterOfMass(self, selection, frames=slice(0, None)):
        """ Computes the center of mass for selected atoms and frames.

            :arg selection: selected atom, can be a single string or a
                            list of atom indices
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

        """

        if type(selection) == str:
            selection = self.selection(selection)

        try:  # Check if a psf file has been loaded
            atomMasses = selection.getMasses()

        except AttributeError:
            print("No .psf file was loaded, please import one "
                  "before using this method.")
            return

        atomMasses = atomMasses.reshape((atomMasses.size, 1, 1))

        dcdData = self.dcdData[selection, frames]

        centerOfMass = np.sum(dcdData * atomMasses, axis=0) / atomMasses.sum()


        return centerOfMass.astype('float32')




    def getAlignedCenterOfMass(self, selection='all', outSel=None,
                               frames=slice(0, None)):
        """ This method aligns the center of mass of each frame to the origin.
            It does not perform any fitting for rotations, so that it can be
            used for center of mass drift corrections if no global angular
            momentum is present.

            :arg selection: selected atom, can be a single string or a
                            list of atom indices
            :arg outSel:    if not None, move atoms from outSel based on
                            center of mass of selection
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)


        if outSel is None:
            outSel = selection
        if type(outSel) == str:
            outSel = self.selection(outSel)


        dataset = self.dcdData[outSel, frames]

        centerOfMass = self.getCenterOfMass(selection, frames)

        # Substract the center of mass coordinates to each atom for each frame
        dataset = dataset - centerOfMass

        return dataset




    def getAlignedData(self, selection, outSel=None, frames=slice(0, None)):
        """ This method will fit all atoms between firstAtom and lastAtom for
            each frame between begin and end, using the first frame for the
            others to be fitted on.

            :arg selection: selected atom, can be a single string or a list
                            of atom indices
            :arg outSel:    if not None, move atoms from this selection based
                            transformation matrix obtained from selection
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

            :returns: a similar array as the initial dataset but with
                      aligned coordinates.

        """

        if type(selection) == str:
            selection = self.selection(selection)

        refData = self.getAlignedCenterOfMass(selection, None, frames)
        q       = molFit_q.alignAllMol(refData)

        if outSel is not None:
            alignData = self.getAlignedCenterOfMass(selection, outSel, frames)
        else:
            alignData = refData


        alignData = molFit_q.applyRotation(alignData, q)

        return alignData






    def getPBC(self, selection='all', frames=slice(0, None)):
        """ This method applies periodic boundary conditions on all
            selected atom coordinates for each frame selected.

        """

        if isinstance(selection, str):
            selection = self.selection(selection)


        dcdData = np.copy(self.dcdData[selection, frames])

        dcdData -= (self.cellDims[frames]
                    * np.floor(dcdData / self.cellDims[frames]))


        return dcdData




    def getProtVolume(self, selection='protein', frame=0):
        """ Computes an approximate volume of the protein by summing the
            volume of each amino acids in the provided selection.

            :arg selection: selection to be used. Should be obviously
                            a protein.
            :arg frame:     frame to be used to compute the volume
            :arg volType:   type of volume to be used, either 'IMGT'
                            (from imgt.org),
                            'hard-sphere' or 'vdW'
                            (the latter standing for van der Waals volume)

            :returns: the volume in :math:`\\AA^3`

            References:

            .. [#] http://proteinsandproteomics.org/\
content/free/tables_1/table08.pdf
            .. [#] http://www.imgt.org/IMGTeducation/Aide-memoire/\
_UK/aminoacids/abbreviation.html#refs

        """

        resVol1 = {'ALA': 92, 'ARG': 225, 'ASN': 135, 'ASP': 125, 'CYS': 106,
                   'GLN': 161, 'GLU': 155, 'GLY': 66, 'HIS': 167, 'HSE': 167,
                   'HSD': 167, 'HSP': 167, 'ILE': 169, 'LEU': 168, 'LYS': 171,
                   'MET': 171, 'PHE': 203, 'PRO': 129, 'SER': 99, 'THR': 122,
                   'TRP': 240, 'TYR': 203, 'VAL': 142}


        resVol2 = {'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1,
                   'CYS': 108.5, 'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1,
                   'HIS': 153.2, 'HSE': 153.2, 'HSD': 153.2, 'HSP': 153.2,
                   'ILE': 166.7, 'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9,
                   'PHE': 189.9, 'PRO': 112.7, 'SER': 89.0, 'THR': 116.1,
                   'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0}



        sel = self.selection(selection)
        sel.frame = frame

        nbrRes = sel.getSubSelection(selection + ' and name CA').size

        vol1 = 0
        vol2 = 0
        for resID in range(nbrRes):
            tmpSel = sel.getSubSelection('resid %s' % str(resID + 1))

            vol1 += resVol1[tmpSel.getUniqueResName()[0]]
            vol2 += resVol2[tmpSel.getUniqueResName()[0]]


        return (vol1 + vol2) / 2



    def getSpecVolume(self, selection='protein', frame=0):
        """ Estimates protein specific volume based on volume estimation.

            :arg selection: selection to be used.
                            Should be obviously a protein.
            :arg frame:     frame to be used to compute the volume
            :arg volType:   type of volume to be used, either
                            'hard-sphere' or 'vdW'
                            (the latter standing for van der Waals volume)

            :returns: the specific volume in :math:`\\cm^3 / g`

        """

        massP = np.sum(self.getAtomsMasses(selection))

        protVol = self.getProtVolume(selection, frame) * 1e-24

        specVol = protVol / massP * 6.02214076e23

        return specVol




    def getGyrationRadius(self, selection='protein', frames=slice(0, None)):
        """ Computes the radius of gyration for the given selection and frames.

            :arg selection: selected atom, can be a single string or a
                            list of atom indices
            :arg frames:    if True, will try to align all atoms to the ones
                            on the first frame

        """

        sel = self.selection(selection)

        com = self.getCenterOfMass(selection, frames)

        gyrR = sel.getMasses()[:, np.newaxis] * np.sum(
            (self.dcdData[sel, frames] - com)**2, 2)
        gyrR = np.sqrt(gyrR.sum(0) / sel.getMasses().sum())

        return gyrR







# --------------------------------------------
# Data analysis methods
# --------------------------------------------
    def getSTDperAtom(self, selection="all", align=False,
                      frames=slice(0, None), mergeXYZ=True):
        """ Computes the standard deviation for each atom in selection
            and averaged over selected frames.

            :arg selection: selected atom, can be a single string or a
                            list of atom indices
            :arg align:     if True, will try to align all atoms to the
                            ones on the first frame
            :arg end:       last frame to be used + 1
            :arg mergeXYZ:  if True, uses the vector from the origin
                            instead of each projections

            :returns: the standard deviation averaged over time.

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, frames)
        else:
            data = self.dcdData[selection, frames]

        # Computes the standard deviation
        if mergeXYZ:
            std = np.sqrt(np.sum(data**2, axis=2))
            std = np.std(std, axis=1)

        else:
            std = np.std(data, axis=1)

        return std




    def getRMSDperAtom(self, selection="all", alignCOM=True, align=False,
                       frames=slice(0, None)):
        """ Computes the RMSD for each atom in selection and for frames
            between begin and end.

            :arg selection: selected atom, can be a single string or a
                            list of atom indices
            :arg alignCOM:  if True, will try to align the center of mass of
                            the selection to the first frame
            :arg align:     if True, will try to align all atoms to the ones
                            on the first frame
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

            :returns: the RMSD averaged over time.

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, frames)
        elif alignCOM:
            data = self.getAlignedCenterOfMass(selection, frames)
        else:
            data = self.dcdData[selection, frames]

        rmsd = data - data[:, 0][:, np.newaxis, :]
        rmsd = np.sqrt(np.sum(rmsd**2, axis=2))

        rmsd = np.mean(rmsd, axis=1)

        return rmsd



    def getRMSDperResidue(self, selection="protein", alignCOM=True,
                          align=False, frames=slice(0, None)):
        """ Computes the RMSD for each residue in selection
            and for selected frames.

            :arg selection: selected atom, can be a single string
                            or a list of atom indices
            :arg alignCOM:  if True, will try to align the center of
                            mass of the selection to the first frame
            :arg align:     if True, will try to align all atoms to
                            the ones on the first frame
            :arg frames:    either not given to select all frames,
                            an int, or a slice object

            :returns: the RMSD averaged over time.

        """

        atoms = self.selection(selection)

        residues = atoms.getUniqueResidues().astype(int)
        resRMSD  = np.zeros_like(residues, dtype='float32')


        atomRMSD = self.getRMSDperAtom(
            selection, alignCOM, align, frames, mergeXYZ)


        for idx, val in enumerate(residues):
            resRMSD[idx] = np.mean(
                atomRMSD[np.argwhere(
                    atoms.getResidues().astype(int) == val)[:, 0]])


        return resRMSD





    def getRMSDperFrame(self, selection="all", alignCOM=True,
                        align=False, frames=slice(0, None)):
        """ Computes the RMSD for each atom in selection and for
            frames between begin and end.

            :arg selection: selected atom, can be a single string or
                            a list of atom indices
            :arg alignCOM:  if True, will try to align the center of mass of
                            the selection to the first frame
            :arg align:     if True, will try to align all atoms to the ones
                            on the first frame
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

            :returns: the RMSD averaged over all selected atoms.

        """

        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            data = self.getAlignedData(selection, frames)
        elif alignCOM:
            data = self.getAlignedCenterOfMass(selection, frames)
        else:
            data = self.dcdData[selection, frames]

        rmsd = data - data[:, 0][:, np.newaxis, :]
        rmsd = np.sqrt(np.sum(rmsd**2, axis=2))
        rmsd = np.mean(rmsd, axis=0)

        return rmsd








# --------------------------------------------
# Plotting methods
# --------------------------------------------
    def plotSTDperAtom(self, selection="all", align=False,
                       frames=slice(0, None), mergeXYZ=True):
        """ Plot the standard deviation along the axis 0 of dataset.
            This makes use of the :func:`getSTDperAtom` method.

            If mergeXYZ is True, then computes the distance
            to the origin first.

        """

        std = self.getSTDperAtom(selection, align, frames, mergeXYZ)
        xRange = self.timestep * np.cumsum(self.dcdFreq[frames])

        if mergeXYZ:
            plt.plot(xRange, std)
            plt.ylabel(r'$STD \ (\AA)$')

        else:
            # In case of three columns for (x, y, z) coordinates,
            # generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, std[:, 0])
            ax[0].set_ylabel(r'$STD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, std[:, 1])
            ax[1].set_ylabel(r'$STD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, std[:, 2])
            ax[2].set_ylabel(r'$STD \ along \ Z \ (\AA)$')

        plt.xlabel(r'$Atom \ index$')

        plt.tight_layout()
        return plt.show(block=False)




    def plotRMSDperAtom(self, selection="all", alignCOM=True, align=False,
                        frames=slice(0, None)):
        """ Plot the RMSD along the axis 0 of dataset.
            This makes use of the :func:`getRMSDperAtom` method.

        """

        rmsd = self.getRMSDperAtom(selection, alignCOM, align, frames)
        xRange = np.arange(rmsd.size)

        if mergeXYZ:
            plt.plot(xRange, rmsd)
            plt.ylabel(r'$RMSD \ (\AA)$')

        else:
            # In case of three columns for (x, y, z) coordinates,
            # generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, rmsd[:, 0])
            ax[0].set_ylabel(r'$RMSD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, rmsd[:, 1])
            ax[1].set_ylabel(r'$RMSD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, rmsd[:, 2])
            ax[2].set_ylabel(r'$RMSD \ along \ Z \ (\AA)$')

        plt.xlabel(r'$Atom \ index$')

        plt.tight_layout()
        return plt.show(block=False)



    def plotRMSDperResidue(self, selection="all", alignCOM=True, align=False,
                           frames=slice(0, None)):
        """ Plot the RMSD along the axis 0 of dataset.
            This makes use of the :func:`getRMSDperAtom` method.

        """

        rmsd = self.getRMSDperResidue(selection, alignCOM, align, frames)
        xRange = np.arange(rmsd.size)

        if mergeXYZ:
            plt.plot(xRange, rmsd)
            plt.ylabel(r'$RMSD \ (\AA)$')

        else:
            # In case of three columns for (x, y, z) coordinates,
            # generate three plot for each.
            fig, ax = plt.subplots(3, 1, sharex=True)

            ax[0].plot(xRange, rmsd[:, 0])
            ax[0].set_ylabel(r'$RMSD \ along \ X \ (\AA)$')

            ax[1].plot(xRange, rmsd[:, 1])
            ax[1].set_ylabel(r'$RMSD \ along \ Y \ (\AA)$')

            ax[2].plot(xRange, rmsd[:, 2])
            ax[2].set_ylabel(r'$RMSD \ along \ Z \ (\AA)$')

        plt.xlabel('Residue')

        plt.tight_layout()
        return plt.show(block=False)






    def plotRMSDperFrame(self, selection="all", alignCOM=True, align=False,
                         frames=slice(0, None)):
        """ Plot the RMSD along the axis 1 of dataset.
            This makes use of the :func:`getRMSDperFrame` method.

        """

        rmsd = self.getRMSDperFrame(selection, alignCOM, align, frames)
        xRange = self.timestep * np.cumsum(self.dcdFreq[frames])

        plt.plot(xRange, rmsd)
        plt.ylabel(r'$RMSD \ (\AA)$')

        plt.xlabel(r'Time (s)')

        plt.tight_layout()
        return plt.show(block=False)





    def plotAveragedDistances_parallelPlot(self, sel1, sel2=None, frames=None,
                                           startDist=None,
                                           maxDist=10, step=2, lwStep=0.8):
        """ Computes averaged distances between sel1 and sel2,
            then plot the result on a parallel plot in a residue-wise manner.

            Both selections need to be the same for all frames used,
            so 'within' keyword cannot be used here.

            :arg sel1:    first selection of atoms for ditance computation
            :arg sel2:    second selection of atoms
                          (optional, if None, sel1 is used)
            :arg frames:  frames to be used for averaging
            :arg maxDist: maximum distance to use for the plot
            :arg step:    step between each distance bin, each of them will be
                          plotted on a color and line width scale.
            :arg lwStep:  line width step for plotting, each bin will
                          be plotted with a
                          linewidth being (maxDist / bin max edge) * lwStep

        """

        dist = self.getDistances(sel1, sel2, frames)


        if isinstance(sel1, str):
            sel1 = self.selection(sel1)

        if sel2 is None:
            sel2 = np.copy(sel1)
        elif isinstance(sel2, str):
            sel2 = self.selection(sel2)


        if startDist is None:
            startDist = step

        rList  = np.arange(maxDist, startDist, -step)

        cmap = cm.get_cmap('hot')
        norm = colors.Normalize(startDist, maxDist)

        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 25]})


        for idx, r in enumerate(rList):
            keep = np.argwhere(dist < r)
            keep = np.column_stack((sel1[keep[:, 0]], sel2[keep[:, 1]]))

            if keep.ndim == 2:
                # Keeps only on index per residue
                resPairs = np.unique(
                    self.psfData.atoms[keep][:, :, 2], axis=0).astype(int)

                ax[1].plot([0, 1], resPairs.T,
                           lw=(maxDist / r) * lwStep, color=cmap(norm(r)))

        ax[1].set_ylabel('Residue number')
        ax[1].set_xlim(0, 1)
        ax[1].xaxis.set_ticks([], [])
        ax[1].tick_params(labelright=True, labelleft=True)

        cb = colorbar.ColorbarBase(ax[0], cmap=cmap, norm=norm)
        ax[0].yaxis.set_ticks_position('left')
        ax[0].yaxis.set_label_position('left')
        ax[0].set_ylabel('Distance [$\AA$]')


        return fig.show()



    def plotAveragedDistances_chordDiagram(self, sel1, sel2=None, frames=None,
                                           startDist=None, maxDist=10, step=2,
                                           lwStep=1.2, resList=None,
                                           labelStep=5):
        """ Computes averaged distances between sel1 and sel2,
            then plot the result on a chord diagram in a residue-wise manner.

            Both selections need to be the same for all frames used,
            so 'within' keyword cannot be used here.

            :arg sel1:      first selection of atoms for ditance computation
            :arg sel2:      second selection of atoms
                            (optional, if None, sel1 is used)
            :arg frames:    frames to be used for averaging
            :arg maxDist:   maximum distance to use for the plot
            :arg step:      step between each distance bin, each of them will
                            be plotted on a color and line width scale.
            :arg lwStep:    line width step for plotting, each bin will be
                            plotted with a linewidth being
                            ( maxDist / bin max edge ) * lwStep
            :arg resList:   list of residue indices
                            (optional, if None, will be guessed from file)
            :arg labelStep: step for residue labels
                            (optional, default 5, each 5 residue are indicated)

        """

        matplotlib.interactive(False)

        chord = ChordDiag(self, sel1, sel2, frames, startDist,
                          maxDist, step, lwStep, resList, labelStep)

        chord.process()

        return chord
