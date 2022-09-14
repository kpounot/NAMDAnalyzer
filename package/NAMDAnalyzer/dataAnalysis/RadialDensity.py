"""

Classes
^^^^^^^

"""

import sys

import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap


from scipy.signal import medfilt2d


try:
    from NAMDAnalyzer.lib.pylibFuncs import py_cdf, py_getRadialNbrDensity
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
          "Please compile it before using it.\n")



class ResidueWiseWaterDensity:
    """ This class allows to compute radial number density for water
        around each residue.

        :arg data:       a :class:`Dataset` class instance containing
                         trajectories data
        :arg sel:        selection to be used for analysis, can be
                         ``protein`` for all residues or
                         ``protein and segname A B C and resid 20:80``
                         for specific segment/chain
                         name(s) and residues.
        :arg maxR:       maximum radius to be used for radial density
                         computation (default 10)
        :arg dr:         radius step, determines the number of bins
                         (default 0.1)
        :arg frames:     frames to be used for analysis (default all)

    """

    def __init__(self, data, sel, maxR=15, dr=0.1, frames=None):

        self.data = data
        self.sel  = sel

        # Parses selection
        if isinstance(sel, str):
            sel = self.data.selection(sel)

        self.maxR = maxR
        self.dr   = dr

        self.radii = np.arange(self.dr, self.maxR, self.dr)


        if not frames:
            self.frames = np.arange(0, self.data.nbrFrames, 1)
        else:
            self.frames = frames


        self.residues = np.sort(np.array(sel.getUniqueResidues()).astype(int))

        self.density = np.zeros((self.residues.size, self.radii.size))

    def compDensity(self):
        """ Computes the density given class attributes.

            Results is stored in *density* attribute
            (radii are in *radii* attribute).

        """

        waters   = self.data.selection('name OH2').coordinates(self.frames)
        cellDims = self.data.cellDims[self.frames]

        for resId, residue in enumerate(self.residues):
            # Prints state, leaving space for verbosity
            # from py_getRadialNbrDensity
            print(50 * '  ' + "[Residue %i of %i]"
                  % (resId + 1, len(self.residues)), end='\r')

            sel = self.sel + ' and resid %s' % residue
            sel = self.data.selection(sel).coordinates(self.frames)

            density = np.zeros(self.radii.size, dtype='float32')

            py_getRadialNbrDensity(
                np.ascontiguousarray(waters), 
                np.ascontiguousarray(sel), 
                density, 
                np.ascontiguousarray(cellDims),
                0, 
                self.maxR, 
                self.dr
            )

            density[0] -= density[0]
            density /= (4 * np.pi * self.radii**2 * self.dr)

            self.density[resId] = density



    def plotDensity(self, medianFiltSize=[13, 3]):
        """ Plots the computed density as a 3D surface.

            :arg medianFiltSize: density can be filtered using
                                 *Scipy.signal medfilt2d* method.
                                 This might help getting better
                                 looking results.
                                 Set it to [1, 1] to get original data.

        """


        cmap = get_cmap('jet')

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        X, Y = np.meshgrid(self.radii, self.residues.astype(int))

        filteredDensity = medfilt2d(self.density, medianFiltSize)

        ax.plot_surface(X, Y, filteredDensity, cmap=cmap)

        ax.set_xlabel('Radius [$\AA$]')
        ax.set_ylabel('Residue')
        ax.set_zlabel('Density')


        fig.show()


class RadialNumberDensity:
    """ Computes the radial density distribution from center of mass
        of selected atoms using the given dr interval.

        :arg data:   a :class:`Dataset` class instance containing
                     trajectories data
        :arg sel1:   first atom selection from which spherical zone
                     will be computed
        :arg sel2:   second selection, only atoms within the spherical zone
                     and corresponding to this selection will be considered
        :arg dr:     radius interval, density is computed as the number
                     of atoms between r and r + dr divided by the total number
                     of sel2 atoms within maxR
        :arg maxR:   maximum radius to be used
        :arg frames: frames to be averaged on, should be a range
                     (default None, all frames are used)

        :returns:
            - **radii** array of radius edges (minimum)
            - **density** radial number density

    """


    def __init__(self, data, sel1, sel2=None, dr=0.1, maxR=15, frames=None):

        self.data = data

        # Parsing selections
        self.sameSel = 0

        if frames is None:
            self.frames = np.arange(0, self.data.nbrFrames, 1)
        else:
            self.frames = frames

        if isinstance(sel1, str) and isinstance(sel2, str):
            if set(sel1.split(' ')) == set(sel2.split(' ')):
                self.sameSel = 1

        # Get the indices corresponding to the selection
        if type(sel1) == str:
            if re.search('within', sel1):
                sel1 = sel1 + ' frame %s' % ' '.join(frames)

            self.sel1 = self.data.selection(sel1)

        if type(sel2) == str:
            if self.sameSel:
                self.sel2 = self.sel1
            else:
                if re.search('within', sel2):
                    sel2 + ' frame %s' % ' '.join(frames)

                self.sel2 = self.data.selection(sel2)

        if sel2 is None:
            self.sel2 = self.sel1
            self.sameSel = 1


        self.dr   = dr
        self.maxR = maxR



    def compDensity(self):
        """ Computes density given class attributes. """

        radii  = np.arange(self.dr, self.maxR, self.dr)

        density = np.zeros(radii.size, dtype='float32')


        sel1 = self.data[self.sel1, self.frames]
        if self.sameSel:
            sel2 = sel1
        else:
            sel2 = self.data[self.sel2, self.frames]

        cellDims = self.data.cellDims[self.frames]

        py_getRadialNbrDensity(sel1, sel2, density, cellDims,
                               self.sameSel, self.maxR, self.dr)

        density[0] -= density[0]
        density /= (4 * np.pi * radii**2 * self.dr)

        self.radii = radii
        self.density = density



    def plotDensity(self):
        """ Plots the computed density. """


        plt.plot(self.radii, self.density)

        plt.xlabel('Radius [$\AA$]')
        plt.ylabel('Density')

        plt.show()
