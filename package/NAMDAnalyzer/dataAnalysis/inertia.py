"""Module that provides methods related to the moment of inertia.

"""

import sys

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap

from scipy.linalg import eig

from NAMDAnalyzer.helpersFunctions.objectConverters import fromSliceToArange


class Inertia:
    """Provides tools for the moment of inertia of a set of atoms.

    Parameters
    ----------
    data : :class:`NAMDAnalyzer.Dataset`
        A 'Dataset' class containing loaded topology and coordinates.
    sel : str or :class:`NAMDAnalyzer.selection.selText.SelText`
        A selection of atoms.
    frames : range or list
        Frames to be used for analysis (default all)
    align : bool
        Whether to align the structures before computing the inertia tensor.

    """

    def __init__(self, data, sel, maxR=15, dr=0.1, frames=None, align=False):
        self.data = data

        # Parses selection
        if isinstance(sel, str):
            sel = self.data.selection(sel)
        self.sel = sel

        if frames is None:
            self.frames = np.arange(0, self.data.nbrFrames, 1)
        else:
            if isinstance(frames, slice):
                frames = fromSliceToArange(frames, self.data.nbrFrames)
            self.frames = frames

        self.align = align

        self.inertiaTensor = []
        self.principalMoments = []
        self.correlationTimes = []
        self.autoCorr = []

    def compInertiaTensor(self):
        """Computes the tensor of inertia for all selected frames.

        The result is stored in the *inertiaTensor* attribute.

        """
        self.inertiaTensor = []
        masses = self.sel.getMasses()
        for idx, val in enumerate(self.frames):
            print(
                "Processing frame %i of %i..." % (idx + 1, self.frames.size),
                end='\r'
            )
            tensor = np.zeros((3, 3))

            if self.align:
                coor = self.data.getAlignedData(
                    self.sel, 
                    frames=val, 
                    ref=self.frames[0]
                ).squeeze()
            else:
                coor = self.data[self.sel, val].squeeze()

            coorSquare = (coor ** 2).sum(1)
            for i, j in product((0, 1, 2), (0, 1, 2)):
                tensor[i, j] = np.sum(
                        masses * (
                            coorSquare * int(i == j) - 
                            coor[:, i] * coor[:, j]
                        )
                )
            self.inertiaTensor.append(tensor)

    def compPrincipalMoments(self):
        """Principal moments of inertia from the diagonalized inertia tensor.

        The result for each frame is stored in *principalMoments* 
        list attribute. Each entry is a tuple containing the eigenvalues and
        the eigenvectors.

        """
        if len(self.inertiaTensor) == 0:
            self.compInertiaTensor()

        self.principalMoments = []
        for val in self.inertiaTensor:
            self.principalMoments.append(eig(val))

    def compDynamics(self, maxInterval=None, nbrPoints=100, nbrTimeOri=20):
        """Comuted the dynamics of principal moments.

        This methods basically computes the ratio of eigenvalues to the one
        a initial time and the autocorrelation of the eigenvectors.
            
        Parameters
        ----------
        maxInterval : int, optional
            Maximum time interval in number of frames to compute 
            the dynamics. The interval will be 
            (maxInterval - nbrTimeOri).
        nbrPoint : int, optional
            Number of time steps to compute the dynamics.
        nbrTimeOri : int, optional
            Number of time origins to average over.

        The result is stored in the *autoCorr* attribute along
        with the times in seconds in the *correlationTimes* attribute.
        The *autoCorr* attribute thus contains two lists, one for the ratios
        of the eigenvalues, and one for the autocorrelation of the 
        eigenvectors.

        """
        if len(self.principalMoments) == 0:
            self.compPrincipalMoments()

        if maxInterval is None:
            maxInterval = self.frames.size

        step = int(maxInterval / nbrPoints)
        self.correlationTimes = (
            self.frames * self.data.dcdFreq[self.frames] * self.data.timestep
        )[:maxInterval:step]

        oriStep = int((self.frames.size - maxInterval) / nbrTimeOri)
        origins = np.arange(nbrTimeOri) * oriStep

        outVal = []
        outVec = []
        for ori in origins:
            vals = np.array(
                [
                    val[0].real for val in 
                    self.principalMoments[ori:ori+maxInterval:step]
                ]
            )
            vecs = np.array(
                [
                    val[1] for val in 
                    self.principalMoments[ori:ori+maxInterval:step]
                ]
            )

            ratios = vals / vals[0] - 1

            correl = (vecs[0].T @ vecs).diagonal(0, 1, 2)
            correl = correl / correl[0]

        self.autoCorr = [ratios, correl]
