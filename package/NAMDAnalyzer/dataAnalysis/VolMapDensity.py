"""

Classes
^^^^^^^

"""

import sys

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import entropy

try:
    from NAMDAnalyzer.lib.pylibFuncs import (py_getWaterDensityVolMap,
                                             py_setWaterDistPBC,
                                             py_waterNumberDensityHist)
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")





class WaterVolNumberDensity:
    """ This class defines methods to compute volumetric map of number density
        of selected atoms. Additional methods to plot a distribution and
        related entropy are available.

        :arg data:    a :class:`Dataset` class instance containing
                      trajectories data
        :arg protSel: selection of protein atoms to be used for
                      calculations
        :arg watSel:  water selection for which number of atoms in each voxel
                      will be computed for each selected frame. If *within*
                      keyword is used, it should be preceded by a
                      *bound to* keyword to avoid errors in computations.
        :arg frames:  frames to be used to compute number density
        :arg nbrVox:  number of voxel in each dimension to be used

    """


    def __init__(self, data, protSel='protein', watSel='water',
                 frames=None, nbrVox=32):

        self.data = data

        if isinstance(protSel, str):
            self.protSel = self.data.selection(protSel)
        else:
            self.protSel = protSel


        if isinstance(watSel, str):
            self.watSel = self.data.selection(watSel)
        else:
            self.watSel = watSel



        self.frames  = frames
        self.nbrVox  = nbrVox

        if self.frames is None:
            self.frames = np.arange(0, self.data.nbrFrames)




    def generateVolMap(self, align=False):
        """ This method computes, for each frame the number density of
            selected atom in each voxel of the volumetric map.

            :arg nbrBins: number of bins to be used to compute the histograms
            :arg align:   if True, all atoms will be aligned for each frame,
                          taking the first frame
                          as reference set. Else, only center of mass of
                          selected atoms are aligned.

        """

        nbrWAtoms = self.watSel.getUniqueName().size

        fullSel = self.protSel + self.watSel

        if align:
            coor = self.data.getAlignedData(self.protSel, fullSel,
                                            frames=self.frames)
        else:
            coor = self.data.getAlignedCenterOfMass(self.protSel,
                                                    fullSel,
                                                    frames=self.frames)


        prot  = coor[fullSel.getSubSelection(self.protSel)]
        water = coor[fullSel.getSubSelection('water')]


        cellDims = self.data.cellDims[self.frames]


        # Moves water molecules to their nearest atom
        py_setWaterDistPBC(water, prot, cellDims, nbrWAtoms)


        # Getting minimum and maximum of all coordinates
        min_x = np.concatenate((prot, water))[:, :, 0].min()
        min_y = np.concatenate((prot, water))[:, :, 1].min()
        min_z = np.concatenate((prot, water))[:, :, 2].min()
        minCoor = np.array([min_x, min_y, min_z])

        # Moving coordinates close to origin
        prot  -= minCoor
        water -= minCoor

        max_x = np.concatenate((prot, water))[:, :, 0].max()
        max_y = np.concatenate((prot, water))[:, :, 1].max()
        max_z = np.concatenate((prot, water))[:, :, 2].max()
        maxCoor = np.array([max_x, max_y, max_z]) * 1.001

        self.volOri      = np.array([0.0, 0.0, 0.0])
        self.volDeltas   = maxCoor / self.nbrVox

        # Get water indices on volMap based on coordinates
        indices = (water[::nbrWAtoms] / maxCoor * self.nbrVox).astype('int32')

        self.histList = []
        self.volMap = np.zeros(
            (self.nbrVox, self.nbrVox, self.nbrVox), dtype='float32')

        py_getWaterDensityVolMap(indices, self.volMap)

        self.histList = np.array(self.histList)
        self.pCoor    = prot
        self.wCoor    = water




    def writeVolMap(self, fileName=None, frame=0, pFrames=None):
        """ Write the volumetric map containing water density.

            The file is in the APBS .dx format style, so that it can
            be imported directly into VMD.
            Moreover, a pdb file is also generated containing frame
            averaged coordinates for aligned protein.

            :arg fileName: file name for .pdb and .dx files. If none,
                           the name of the loaded .psf file is used.
            :arg frame:    frame to be used to generate the pdb file
            :arg pFrames:  if not None, this will be used instead for
                           protein frame selection and the resulting
                           coordinates will be an average over all
                           selected frames

        """

        if fileName is None:
            fileName = self.data.psfFile[:-4]

        volMap = self.volMap.flatten()

        wSel = self.data.selection('water')

        # Gets water and protein coordinates for selected frame
        wCoor = self.wCoor[:, frame]

        if pFrames is None:
            pCoor = self.pCoor[:, frame]
        else:
            pCoor = np.mean(self.pCoor[:, pFrames], axis=1)

        coor  = np.concatenate((pCoor, wCoor)).squeeze()


        # Write the volumetric map file
        with open(fileName + '.dx', 'w') as f:
            f.write('object 1 class gridpositions counts ' + 3 * '%i '
                    % (self.nbrVox,
                       self.nbrVox,
                       self.nbrVox) + '\n')
            f.write('origin %f %f %f\n'
                    % (self.volOri[0], self.volOri[1], self.volOri[2]))
            f.write('delta %f 0.000000 0.000000\n' % self.volDeltas[0])
            f.write('delta 0.000000 %f 0.000000\n' % self.volDeltas[1])
            f.write('delta 0.000000 0.000000 %f\n' % self.volDeltas[2])
            f.write('object 2 class gridconnections counts ' + 3 * '%i '
                    % (self.nbrVox,
                       self.nbrVox,
                       self.nbrVox) + '\n')
            f.write('object 3 class array type float rank 0 '
                    'items %i data follows\n' % self.nbrVox**3)

            for idx in range(int(np.ceil(volMap.size / 3))):
                batch = volMap[3 * idx:3 * idx + 3]
                for val in batch:
                    f.write('%f ' % val)

                f.write('\n')


            f.write('attribute "dep" string "positions"\n')
            f.write('object "regular positions regular connections" '
                    'class field\n')
            f.write('component "positions" value 1\n')
            f.write('component "connections" value 2\n')
            f.write('component "data" value 3\n')



        # Write the PDB file
        wSel = self.data.selection(wSel._indices)

        sel = self.protSel + wSel

        sel.writePDB(fileName, coor=coor)




    def getDistribution(self, nbrBins=100, minN=0):
        """ Computes the ditribution of water molecule number
            in volumetric map.

            :arg nbrBins: number of bins to be used to computes histogram
            :arg minN:    minimum limit on volumetric map values, only voxels
                          containing values higher than given one will
                          be used.

        """

        volMap = self.volMap[self.volMap > minN]

        edges = np.arange(volMap.min(), volMap.max(),
                          (volMap.max() - volMap.min()) / nbrBins,
                          dtype='float32')

        dist  = np.zeros(nbrBins, dtype='float32')

        py_waterNumberDensityHist(volMap, edges, dist)


        return edges, dist / dist.sum()



# --------------------------------------------
# Plotting methods
# --------------------------------------------
    def plotDistribution(self, nbrBins=100, minN=0, kwargs={'lw': 1}):
        """ Plots the distribution of density for all voxel having a value
            higher than 0.

            :arg frames: frames to be used in *histList* attribute for
                         plotting, if multiple frames are selected, the
                         average is returned. This argument should be an
                         integer, a list or a range.
            :arg minN:   minimum limit on volumetric map values, only voxels
                         containing values higher than given one will be used.
            :arg kwargs: additional keywords arguments to give to matplotlib
                         plot and fill_between functions

        """

        edges, dist = self.getDistribution(nbrBins, minN)


        plt.plot(edges, dist, **kwargs)
        plt.fill_between(edges, dist, alpha=0.5, **kwargs)

        plt.xlabel('Density d')
        plt.ylabel('P[d]')

        plt.axes().spines['top'].set_visible(False)
        plt.axes().spines['right'].set_visible(False)

        plt.show()
