"""Module for structural analysis, mostly based on mean square deviations.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg import eig


class Structure:
    """Structural analysis based on deviation from reference structure.

    Attributes
    ----------
    residueWiseMSD : np.ndarray
        Contains the mean-square displacements computed using 
        :py:meth:`getMSDperResidue`.

    """
    def __init__(self, data):
        """Structural analysis based on deviation from reference structure.

        Parameters
        ----------
        data : NAMDAnalyzer.Dataset.Dataset class
            A dataset containing topology and trajectory data.

        """
        self.data = data

        self.msdPerAtom = None
        self.msdPerResidue = None
        self.rmsdPerFrame = None
        self.rmsdPerAtom = None
        self.rmsdPerResidue = None
        self.stdPerClusters = None
        self.rmsdMatrix = None

    def getMSDperAtom(self, selection, align=False, alignCOM=True, 
                      dt=20, frames=slice(0, None)):
        """Compute the mean-square displacement for each atom in the selection.

        Parameters
        ----------
        selection : np.ndarray or str
            Selection of atoms.
        align : bool
            If True, will align the molecules prior to MSD computation.
        alignCOM : bool
            If True, will align the center-of-mass in each frame.
        dt : int
            Number of frames for the time interval to compute the MSD.
        frames : slice, range, np.ndarray
            Frames to be used for MSD computation, the step will define the 
            time interval for the MSD.

        Returns
        -------
        msd : np.ndarray
            A list containing the mean-square displacement for each residue.
        msdStd : np.ndarray
            The standard deviation, time-origin wise, for the MSD.

        """
        out = []
        outStd = []
        selAtoms = self.data.selection(selection)

        if align:
            coor = self.data.getAlignedData(selAtoms, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(
                selAtoms, frames=frames
            )
        else:
            coor = self.data.dcdData[selAtoms, frames]

        msd = ((coor[:, dt:] - coor[:, :-dt]) ** 2).sum(-1)

        self.msdPerAtom = msd.mean(1)

        return self.msdPerAtom

    def getMSDperResidue(self, selection, align=False, alignCOM=True, 
                         dt=20, frames=slice(0, None)):
        """Compute the mean-square displacement for each residue in the selection.

        Parameters
        ----------
        selection : np.ndarray or str
            Selection of atoms.
        align : bool
            If True, will align the molecules prior to MSD computation.
        alignCOM : bool
            If True, will align the center-of-mass in each frame.
        dt : int
            Number of frames for the time interval to compute the MSD.
        frames : slice, range, np.ndarray
            Frames to be used for MSD computation, the step will define the 
            time interval for the MSD.

        Returns
        -------
        msd : np.ndarray
            A list containing the mean-square displacement for each residue.

        """
        out = []
        selAtoms = self.data.selection(selection)

        if align:
            coor = self.data.getAlignedData(selAtoms, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(
                selAtoms, frames=frames
            )
        else:
            coor = self.data.dcdData[selAtoms, frames]

        msd = ((coor[:, dt:] - coor[:, :-dt]) ** 2).sum(-1)

        residues = selAtoms.getUniqueResidues()
        for idx, resid in enumerate(residues):
            print("Processing resid %i of %i" % (idx + 1, residues.size), end='\r')
            sel = self.data.selection(selection + ' and resid %s' % resid)
            vals, id1, id2 = np.intersect1d(sel, selAtoms, return_indices=True)
            out.append(msd[id2].mean())

        self.msdPerResidue = out

        return self.msdPerResidue

    def getRMSDperAtom(self, selection="all", ref=0, alignCOM=True, 
                       align=False, frames=slice(0, None)):
        """ Computes the RMSD for each atom in selection and for frames
            between begin and end.

            :arg selection: selected atom, can be a single string or a
                            list of atom indices
            :arg ref:       reference frame to compute the RMSD.
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
            selection = self.data.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            coor = self.data.getAlignedData(selection, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(selection, frames=frames)
        else:
            coor = self.data.dcdData[selection, frames]

        rmsd = coor - coor[:, [int(ref)]]
        rmsd = np.sum(rmsd**2, axis=2)
        rmsd = np.sqrt(np.mean(rmsd, axis=1))

        self.rmsdPerAtom = rmsd

        return self.rmsdPerAtom

    def getRMSDperResidue(self, selection="protein", ref=0, alignCOM=True,
                          align=False, frames=slice(0, None)):
        """ Computes the RMSD for each residue in selection
            and for selected frames.

            :arg selection: selected atom, can be a single string
                            or a list of atom indices
            :arg ref:       reference frame to compute the RMSD.
            :arg alignCOM:  if True, will try to align the center of
                            mass of the selection to the first frame
            :arg align:     if True, will try to align all atoms to
                            the ones on the first frame
            :arg frames:    either not given to select all frames,
                            an int, or a slice object

            :returns: the RMSD averaged over time.

        """
        allAtoms = self.data.selection(selection)
        residues = allAtoms.getUniqueResidues()
        resRMSD  = np.zeros_like(residues, dtype='float32')
        atomRMSD = self.getRMSDperAtom(
            allAtoms, ref, alignCOM, align, frames)

        for idx, val in enumerate(residues):
            atoms = self.data.selection(selection + " and resid %s" % val)
            resRMSD[idx] = np.mean(
                atomRMSD[np.isin(allAtoms.getIndices(), atoms.getIndices())])

        self.rmsdPerResidue = resRMSD[:residues.size]

        return self.rmsdPerResidue

    def getRMSDperFrame(self, selection="all", alignCOM=True,
                        align=False, ref=0, frames=slice(0, None)):
        """ Computes the RMSD for each atom in selection and for
            frames between begin and end.

            :arg selection: selected atom, can be a single string or
                            a list of atom indices
            :arg alignCOM:  if True, will try to align the center of mass of
                            the selection to the first frame
            :arg align:     if True, will try to align all atoms to the ones
                            on the first frame
            :arg ref:       Reference frame for the RMSD computation
            :arg frames:    either not given to select all frames, an int,
                            or a slice object

            :returns: the RMSD averaged over all selected atoms.

        """
        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.data.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            coor = self.data.getAlignedData(selection, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(selection, frames=frames)
        else:
            coor = self.data.dcdData[selection, frames]


        rmsd = coor - coor[:, [int(ref)]]
        rmsd = np.sum(rmsd**2, axis=2)
        rmsd = np.sqrt(np.mean(rmsd, axis=0))

        self.rmsdPerFrame = rmsd

        return self.rmsdPerFrame

    def getSTDClusters(self, selection="all", alignCOM=True, align=False,
                       frames=slice(0, None)):
        """Principal components of coordinates standard deviation.

        The matrix :math:`C_{ij} = \langle x_i - \\bar{x_i} \\rangle
        \langle x_j - \\bar{x_j} \\rangle`, where :math:`x_i` is a specific
        degree of freedom (a coordinate of a specific atom) is computed [#]_.

        Parameters
        ----------
        selection : str, np.ndarray
            A selection of atoms.
            (default "all")
        alignCOM : bool
            Whether centers of mass should be aligned. 
            (default True)
        align : bool
            Whether the molecules should be aligned.
            Override *alignCOM* if True.
            (default False)
        frames : list, slice, np.ndarray
            A selection of frames to perform the computation
            (default slice(0, None))

        Returns
        -------
        eigVal : np.ndarray
            The eigenvalues of the matrix :math:`C_{ij}`.
        eigVec : np.ndarray
            The eigenvectors of the matrix :math:`C_{ij}`.

        References
        ----------

        .. [#] https://doi.org/10.1016/S1574-1400(09)00502-7

        """
        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.data.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            coor = self.data.getAlignedData(selection, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(selection, frames=frames)
        else:
            coor = self.data.dcdData[selection, frames]

        std = np.std(coor, 1).flatten()
        matrix = std[:, np.newaxis] @ std[np.newaxis, :]
        eigVal, eigVec = eig(matrix)

        self.stdPerClusters = (np.abs(eigVal), np.abs(eigVec))

        return self.stdPerClusters

    def getRMSDMatrix(self, selection="all", alignCOM=True, align=False,
                      frames=slice(0, None)):
        """Matrix of RMSD for each pair of frames.

        Parameters
        ----------
        selection : str, np.ndarray
            A selection of atoms.
            (default "all")
        alignCOM : bool
            Whether centers of mass should be aligned. 
            (default True)
        align : bool
            Whether the molecules should be aligned.
            Override *alignCOM* if True.
            (default False)
        frames : list, slice, np.ndarray
            A selection of frames to perform the computation
            (default slice(0, None))

        Returns
        -------
        rmsd : np.ndarray
            The matrix of RMSD for each pair of frames.

        """
        # Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.data.selection(selection)

        # Align selected atoms for each selected frames
        if align:
            coor = self.data.getAlignedData(selection, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(selection, frames=frames)
        else:
            coor = self.data.dcdData[selection, frames]

        out = np.zeros((coor.shape[1], coor.shape[1]), dtype='float32')
        for i in range(coor.shape[1]):
            rmsd = np.sum((coor[:, [i]] - coor) ** 2, -1)
            out[i] = np.sqrt(np.mean(rmsd, 0))

        self.rmsdMatrix = out

        return self.rmsdMatrix

# --------------------------------------------
# Plotting methods
# --------------------------------------------
    def plotRMSDperAtom(self):
        """ Plot the RMSD along the axis 0 of dataset.
            This makes use of the :func:`getRMSDperAtom` method.

        """
        rmsd = self.rmsdPerAtom
        xRange = np.arange(rmsd.size)

        plt.plot(xRange, rmsd)
        plt.ylabel(r'$RMSD \ (\AA)$')
        plt.xlabel(r'$Atom \ index$')

        plt.tight_layout()
        return plt.show(block=False)

    def plotRMSDperResidue(self):
        """ Plot the RMSD along the axis 0 of dataset.
            This makes use of the :func:`getRMSDperAtom` method.

        """
        rmsd = self.rmsdPerResidue
        xRange = np.arange(rmsd.size)

        plt.plot(xRange, rmsd)
        plt.ylabel(r'$RMSD \ (\AA)$')
        plt.xlabel('Residue')

        plt.tight_layout()
        return plt.show(block=False)

    def plotRMSDperFrame(self):
        """ Plot the RMSD along the axis 1 of dataset.
            This makes use of the :func:`getRMSDperFrame` method.

        """
        rmsd = self.rmsdPerFrame
        xRange = self.data.timestep * np.cumsum(self.data.dcdFreq[frames])

        plt.plot(xRange, rmsd)
        plt.ylabel(r'$RMSD \ (\AA)$')

        plt.xlabel(r'Time (s)')

        plt.tight_layout()
        return plt.show(block=False)

    def plotSTDClusters(self):
        """Plot the three main components of :py:meth:`getSTDClusters`."""
        eigVal, eigVec = self.stdPerClusters

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        ax.scatter(eigVec[:, 0], eigVec[:, 1], zs=eigVec[:, 2], s=5)
        ax.set_xlabel('PC 1', labelpad=12)
        ax.set_ylabel('PC 2', labelpad=12)
        ax.set_zlabel('PC 3', labelpad=12)

    def plotRMSDMatrix(self):
        """Plot the RMSD matrix generated by :py:meth:`getRMSDMatrix`."""
        plt.imshow(self.rmsdMatrix)
        plt.axes().invert_yaxis()
        plt.colorbar()
