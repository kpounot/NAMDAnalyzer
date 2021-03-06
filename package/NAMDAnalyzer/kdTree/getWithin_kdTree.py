import numpy as np

from NAMDAnalyzer.kdTree.periodic_kdtree import PeriodicCKDTree



def getWithin_kdTree(allAtoms, refSel, outSel, keepIdx, cellDims, distance):
    """ This method uses the periodicCKDTree class to find all atoms within
        given distance from user selection.

        The periodic KDTree class was written by Patrick Varilly.
        See https://github.com/patvarilly/periodic_kdtree

        :arg allAtoms: atom coordinates for all seleted frames
                       (3D numpy array)
        :arg usrSel:   atom indices that will be used in
                       query_ball_point method
        :arg keepIdx:  array of shape (# atoms, # frames) to store ones where
                       atoms should be kept within given
                       distance and selection
        :arg cellDims: cell dimensions for all selected frames
        :arg distance: distance in angstrom for query_ball_point method

        :returns: the keepIdx array with ones where atoms are within
                  given distance and 0 otherwise.

    """


    for frame in range(keepIdx.shape[1]):
        bounds  = cellDims[frame]
        outPos  = allAtoms[outSel, frame]

        refPos  = allAtoms[refSel, frame]

        T = PeriodicCKDTree(bounds, outPos)

        toKeep = T.query_ball_point(refPos, r=distance)

        toKeep = np.unique(np.concatenate(toKeep)).astype(int)

        keepIdx[outSel[toKeep], frame] = 1
