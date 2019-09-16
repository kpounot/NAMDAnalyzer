import numpy as np

from .periodic_kdtree import PeriodicCKDTree



def getWithin_kdTree(allAtoms, usrSel, keepIdx, cellDims, distance):
    """ This method uses the periodicCKDTree class to find all atoms within given distance
        from user selection.

        The periodic KDTree class was written by Patrick Varilly.
        See https://github.com/patvarilly/periodic_kdtree

        Input:  allAtoms    -> atom coordinates for all seleted frames (3D numpy array)
                usrSel      -> atom indices that will be used in query_ball_point method
                keepIdx     -> array of shape (# atoms, # frames) to store ones where
                               atoms should be kept within given distance and selection
                cellDims    -> cell dimensions for all selected frames
                distance    -> distance in angstrom for query_ball_point method 

        Returns the keepIdx array with ones where atoms are within given distance and 0 otherwise. """


    for frame in range(keepIdx.shape[1]):
        bounds  = cellDims[frame]
        allPos  = allAtoms[:,frame]

        selPos  = allAtoms[usrSel, frame]

        T = PeriodicCKDTree(bounds, allPos)

        toKeep = T.query_ball_point(selPos, r=distance)

        toKeep = np.unique( np.concatenate( toKeep ) )

        keepIdx[toKeep, frame] = 1



