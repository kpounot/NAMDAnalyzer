"""Module for essential dynamics analysis.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from NAMDAnalyzer.helpersFunctions.objectConverters import fromSliceToArange


class EssentialDynamics:
    """Essential dynamics analysis of the trajectory.

    Attributes
    ----------

    """
    def __init__(self, data):
        """Essential dynamics analysis of the trajectory.

        Parameters
        ----------
        data : NAMDAnalyzer.Dataset.Dataset class
            A dataset containing topology and trajectory data.

        """
        self.data = data
        self.coor = None

    def computeCartesianCorrMatrix(
            self, 
            selection='name CA', 
            align=True,
            alignCOM=False,
            frames=slice(0, None)
    ): 
        """Correlation matrix from cartesian coordinates.

        The 3D matrix of shape (atoms, frames, 3 (positions)) is reshaped in a
        2D matrix of shape (atoms * 3, frames). The correlation
        coefficients are then computes by considering the rows as variables 
        and the columns as observations of the variables.

        Parameters
        ----------
        selection : np.ndarray or str
            Selection of atoms.
        align : bool
            If True, will align the molecules prior to MSD computation.
        alignCOM : bool
            If True, will align the center-of-mass in each frame.
        frames : slice, range, np.ndarray
            Frames to be used for the computation.

        Returns
        -------
        corr : np.ndarray
            A 2D matrix containing the correlation coefficients.
            For N atoms in the selection, the returned matrix is of size
            3N x 3N.

        """
        sel = self.data.selection(selection)

        if align:
            coor = self.data.getAlignedData(sel, frames=frames)
        elif alignCOM:
            coor = self.data.getAlignedCenterOfMass(
                sel, frames=frames
            )
        else:
            coor = self.data.dcdData[sel, frames]

        nbrFrames = coor.shape[1]
        coor = coor.transpose(1, 0, 2).flatten()
        coor = coor.reshape(nbrFrames, sel.size * 3).T

        self.coor = coor

        return np.corrcoef(coor)

    def computeInternalCorrMatrix(
            self, 
            selection='name CA', 
            frames=slice(0, None)
    ): 
        """Correlation matrix from internal pairwise distances.

        Parameters
        ----------
        selection : np.ndarray or str
            Selection of atoms.
        frames : slice, range, np.ndarray
            Frames to be used for the computation.

        Returns
        -------
        corr : np.ndarray
            A 2D matrix containing the correlation coefficients.
            For N atoms in the selection, the returned matrix is of size
            N * N with N * (N - 1) / 2 independent components.

        """
        sel = self.data.selection(selection)

        if isinstance(frames, slice):
            frames = fromSliceToArange(frames, self.data.nbrFrames)

        dist = []
        for frame in frames:
            matrix = self.data.getDistances(sel, frame=frame)
            dist.append(matrix[np.where(np.triu(matrix))].flatten())

        return np.corrcoef(np.asarray(dist).T)