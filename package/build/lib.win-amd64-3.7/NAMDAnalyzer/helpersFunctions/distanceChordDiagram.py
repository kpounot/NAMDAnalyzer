import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
from matplotlib.path import Path as mpath
import matplotlib.patches as mpatches


class ChordDiag:
    """ This class provides methods to draw a chord diagram from pairwise
        distance matrix. Distances are binned and they form teh edges that
        link the nodes that represent the residue in the protein.

        Input:  data    -> Dataset class instance
                sel1    -> first selection of atoms for ditance computation
                sel2    -> second selection of atoms (optional, if None, sel1 is used)
                frames  -> frames to be used for averaging
                maxDist -> maximum distance to use for the plot
                step    -> step between each distance bin, each of them will be plotted on a color
                           and line width scale
                lwStep  -> line width step for plotting, each bin will be plotted with a 
                           linewidth being ( maxDist / bin max edge ) * lwStep 
                resList -> list of residue indices (optional, if None, will be guessed from file) """



    def __init__(self, data, sel1, sel2=None, frames=None, startDist=None,
                 maxDist=10, step=2, lwStep=1.2, resList=None, labelStep=5):

        self.data = data

        self.sel1   = sel1
        self.sel2   = sel2
        self.frames = frames

        #_Parses selection arguments
        if isinstance(self.sel1, str):
            self.sel1 = self.data.selection(sel1)

        if self.sel2 is None:
            self.sel2 = np.copy(sel1)
        elif isinstance(self.sel2, str):
            self.sel2 = self.data.selection(sel2)


        self.startDist  = startDist
        self.maxDist    = maxDist
        self.step       = step
        if startDist is None:
            self.startDist = step



        #_Define some variables
        self.dist         = None
        self.rList        = np.arange( self.maxDist, self.startDist, -self.step )
        self.resPairsList = []


        if resList is None:
            self.resList = self.data.selection('protein')
            self.resList = np.unique( self.data.psfData.atoms[self.resList][:,2].astype(int) )

        if isinstance(resList, list):
            self.resList = np.array(resList)



        self.lwStep = lwStep
        self.labelStep = labelStep
        self.cmap = cm.get_cmap('summer')
        self.norm = colors.Normalize(self.startDist, self.maxDist)






    def _getDistMatrixAndPairs(self):
        """ Computes the distance matrix given the selection(s) and extract the residues
            pairs that are stored for each distance bin in self.resPairsList. """

        self.dist = self.data.getAveragedDistances(self.sel1, self.sel2, self.frames)

        for idx, r in enumerate(self.rList):
            keep = np.argwhere( self.dist < r )
            keep = np.column_stack( (self.sel1[keep[:,0]], self.sel2[keep[:,1]]) )
    
            if keep.ndim == 2:
                #_Keeps only on index per residue
                keep = np.unique( self.data.psfData.atoms[keep][:,:,2], axis=0 ).astype(int)

                self.resPairsList.append( keep )
                





    def _drawNodes(self):


        allNodes = 2*np.pi*(self.resList-1) / self.resList.size
        self.ax.scatter( allNodes, np.ones_like(allNodes), color=(1,1,1,1), ec=(0,0,0,0.5))
        
        for idx, r in enumerate(self.rList):
            pts = 2 * np.pi * (self.resPairsList[idx].flatten() - 1) / self.resList.size
            self.ax.scatter( pts, np.ones_like(pts), color=self.cmap(self.norm(r)), 
                             alpha=(1 - r / (2*self.maxDist)), ec=None)






    def _drawEdges(self):

        for idx, r in enumerate(self.rList):
            pts = 2 * np.pi * (self.resPairsList[idx]-1) / self.resList.size

            for i, pt in enumerate(pts):

                halfD = np.sqrt( (pt[1] - pt[0])**2 ) / np.pi


                path = mpath( [(pt[0], 1), 
                               (pt[0], 0.3 * (3 - int( halfD * 4 / np.pi )) ), 
                               (pt[1], 0.3 * (3 - int( halfD * 4 / np.pi )) ), 
                               (pt[1], 1)],
                              [mpath.MOVETO, mpath.CURVE4, mpath.CURVE4, mpath.CURVE4] )

                patch = mpatches.PathPatch( path, fill=False, ec=self.cmap(self.norm(r)),
                                            alpha=(1 - r / (2*self.maxDist)),
                                            lw=(self.maxDist/r)*self.lwStep )

                self.ax.add_patch(patch)






    def process(self):

        if self.dist is None:
            self._getDistMatrixAndPairs()

        self.figure = plt.figure()
        self.grid   = self.figure.add_gridspec(1, 2, width_ratios=[25,1])
        self.ax     = self.figure.add_subplot(self.grid[0], projection='polar')

        cb = self.figure.add_subplot(self.grid[1])
        self.cb = colorbar.ColorbarBase(cb, cmap=self.cmap, norm=self.norm)
        cb.set_ylabel('Distance [$\AA$]')

        self.ax.spines['polar'].set_visible(False) 
        self.ax.set_yticks([])

        self.ax.set_xticks(2*np.pi*np.arange(0, self.resList.size, self.labelStep) / self.resList.size)
        self.ax.set_xticklabels(np.arange(1, self.resList.size+1, self.labelStep).astype(str))
        self.ax.grid(False)


        self._drawNodes()
        self._drawEdges()






    def show(self, xlim=None, ylim=(0., 1.05)):

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.figure.show()
