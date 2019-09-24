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
    from NAMDAnalyzer.lib.pylibFuncs import py_cdf

except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
            + "Please compile it before using it.\n")






class ResidueWiseWaterDensity:
    """ This class allows to compute radial number density for water around each residue.

        :arg data:       a :class:`Dataset` class instance containing trajectories data 
        :arg sel:        selection to be used for analysis, can be ``protein`` for all residues
                         or ``protein and segname A B C and resid 20:80`` for specific segment/chain 
                         name(s) and residues. 
        :arg maxR:       maximum radius to be used for radial density computation (default 10) 
        :arg dr:         radius step, determines the number of bins (default 0.1)
        :arg frames:     frames to be used for analysis (default all)

    """

    def __init__(self, data, sel, maxR=15, dr=0.1, frames=None):

        self.data    = data
        self.sel     = sel

        #_Parses selection
        if isinstance(sel, str):
            sel = self.data.selection(sel)


        self.maxR    = maxR
        self.dr      = dr 

        self.radii  = np.arange(self.dr, self.maxR, self.dr) #_Gets x-axis values


        if not frames:
            self.frames = np.arange(0, self.data.nbrFrames, 1)
        else:
            self.frames = frames


        self.residues = sel.getUniqueResidues()

        self.density = np.zeros( (self.residues.size, self.radii.size) )



    def compDensity(self):
        """ Computes the density given class attributes. 

            Results is stored in *density* attribute (radii are in *radii* attribute).

        """

        for resId, residue in enumerate(self.residues):

            sel = self.sel + ' and resid %s' % residue

            density = np.zeros( self.radii.size, dtype='float32' )

            for frameId, frame in enumerate(self.frames):

                print('Computes frame %i of %i for residue %i of %i   ' % (frameId+1,
                                                                        len(self.frames),
                                                                        resId+1,
                                                                        self.residues.size),
                      end='\r' )

                dist = self.data.getDistances('name OH2', sel, frame ).flatten()

                py_cdf(dist, density, self.maxR, self.dr, len(self.frames)) 

            density[0] -= density[0]
            density /= (4 * np.pi * self.radii**2 * self.dr)

            self.density[resId] = density



    def plotDensity(self, medianFiltSize=[13,3]):
        """ Plots the computed density as a 3D surface. 
    
            :arg medianFiltSize: density can be filtered using *Scipy.signal medfilt2d*
                                 method. This might help getting better looking results.
                                 Set it to [1, 1] to get original data.

        """


        cmap = get_cmap('jet')

        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

        X, Y = np.meshgrid(self.radii, self.residues.astype(int))


        filteredDensity = medfilt2d(self.density, medianFiltSize) #_For better visualization


        ax.surface(X, Y, filteredDensity, cmap=cmap)

        ax.set_xlabel('Radius [$\AA$]')
        ax.set_ylabel('Residue')
        ax.set_zlabel('Density')


        fig.show()








class RadialNumberDensity:
    """ Computes the radial density distribution from center of mass of selected atoms 
        using the given dr interval.

        :arg data:   a :class:`Dataset` class instance containing trajectories data 
        :arg sel1:   first atom selection from which spherical zone will be computed
        :arg sel2:   second selection, only atoms within the spherical zone and corresponding
                        to this selection will be considered
        :arg dr:     radius interval, density is computed as the number of atoms between r and
                        r + dr divided by the total number of sel2 atoms within maxR
        :arg maxR:   maximum radius to be used
        :arg frames: frames to be averaged on, should be a range 
                        (default None, every 10 frames are used) 

        :returns:
            - **radii** array of radius edges (minimum)
            - **density** radial number density

    """


    def __init__(self, data, sel1, sel2=None, dr=0.1, maxR=10, frames=None):

        self.data = data

        #_Parses selection
        if isinstance(sel1, str):
            self.sel1 = self.data.selection(sel1)
        else:
            self.sel1 = sel1


        if sel2 is None:
            self.sel2 = self.data.selection('all')
        elif isinstance(sel2, str):
            self.sel2 = self.data.selection(sel2)
        else:
            self.sel2 = sel2

        
        self.dr   = dr
        self.maxR = maxR


        if not frames:
            self.frames = np.arange(0, self.data.nbrFrames, 1)
        else:
            self.frames = frames


        

    def compDensity(self):
        """ Computes density given class attributes. """

        radii  = np.arange(dr, maxR, dr) #_Gets x-axis values

        density = np.zeros( radii.size, dtype='float32' )

        
        for frameId, frame in enumerate(self.frames):
            print('Processing frame %i of %i...' % (frameId+1, len(self.frames)), end='\r')

            dist = self.getDistances(sel1, sel2, frame).flatten()

            py_cdf(dist, density, maxR, dr, len(self.frames)) 

            
        density[0] -= density[0]

        density /= (4 * np.pi * radii**2 * dr)


        self.radii = radii 
        self.density = density



    def plotDensity(self):
        """ Plots the computed density. """


        plt.plot(self.radii, self.density)

        plt.xlabel('Radius [$\AA$]')
        plt.ylabel('Density')

        plt.show()








class COMRadialNumberDensity:
    """ Computes the radial density distribution from center of mass of selected atoms 
        using the given dr interval.

        :arg data:      a :class:`Dataset` class instance containing trajectories data 
        :arg sel:       atom selection, can be string or array of atom indices
        :arg dr:        radius interval, density is computed as the number of atoms between r and
                            r + dr divided by the volume integral in spherical coordinates for unit r
                            times the total number of atoms within maximum r
        :arg maxR:      maximum radius to be used
        :arg frame:     frame to be used for atom coordinates 

        :returns:
            - **radii** array of radius edges (minimum)
            - **density** radial number density

    """


    def __init__(self, data, sel='protH', dr=0.5, maxR=60, frame=-1):

        
        self.data = data

        #_Parses selection
        if isinstance(sel, str):
            self.sel = self.data.selection(sel)
        else:
            self.sel = sel


        self.dr   = dr
        self.maxR = maxR



    def compDensity(self):
        """ Computes density given class attributes. """

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

        self.radii   = radii 
        self.density = density


    def plotDensity(self):
        """ Plots the computed density. """


        plt.plot(self.radii, self.density)

        plt.xlabel('Radius [$\AA$]')
        plt.ylabel('Density')

        plt.show()




