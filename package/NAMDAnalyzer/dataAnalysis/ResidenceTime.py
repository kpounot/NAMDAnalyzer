"""

Classes
^^^^^^^

"""

import sys

import numpy as np

import matplotlib.pyplot as plt


class ResidenceTime:
    """ This class defines methods to compute retention time of atoms in a certain region.
        This determines how fast atoms in sel2 can leave the vicinity of sel1.
        
        Some plotting methods are also available to quicly check the results. 
        
        :arg data:       a :class:`Dataset` class instance containing trajectories data 
        :arg sel:        selection corresponding to analysis region (with 'within' keyword')
        :arg tMax:       maximum number of frames to be used 
        :arg step:       time interval between each computed frame
        :arg nbrTimeOri: number of time origins to be averaged over (optional, default 20) 

    """


    def __init__(self, data, sel, tMax=25, step=1, nbrTimeOri=20):
                                                                                                            
        self.data       = data
        self.sel        = sel 
        self.tMax       = tMax
        self.step       = step
        self.nbrTimeOri = nbrTimeOri

        self.resTime     = None
        self.times       = None
        self.residueWise = None
        self.residues    = None



#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compResidenceTime(self):
        """ For each frame in the range of tMax with given step, the number of atoms in selected region
            is computed and divided by the number of atoms at time origin. The result is averaged over
            multiple time origins. 

            The result is stored in *resTime* attribute.

        """


        self.times  = ( np.arange(0, self.tMax, self.step, dtype=int) 
                        * self.data.dcdFreq[0] * self.data.timestep * 1e12 )
        corr        = np.zeros_like(self.times)


        oriList = ( (self.data.nbrFrames - self.tMax) * np.random.random(self.nbrTimeOri) ).astype(int)
        for idx, frame in enumerate(oriList):

            print("Processing time origin %i of %i with %i frames..." 
                                % (idx+1, oriList.size, self.tMax/self.step), end='\r')

            sel = self.data.selection(self.sel + " frame %i:%i:%i" % (frame, frame+self.tMax, self.step))
            
            for tIdx, keepIdx in enumerate(sel):
                corr[tIdx] += np.intersect1d(sel[0], keepIdx).size


        self.resTime = corr / corr[0]




    def compResidueWiseResidenceTime(self, dt=25):
        """ Computes, for a given time dt, the residence time for 
            selected atoms around each residue in the protein. 

            By default all atoms pertaining to protein are selected.
            If different proteins are present, the *segName* argument can be used in selection text.

            :arg dt:      time step to compute residence time. Basically, the number of selected molecules
                          wihtin the given region at initial time divides the number that stayed within the 
                          region after a time dt (in number of frames).

            The result is stored in *residueWise* attribute

        """

        #_Gets residue number list
        resSel = self.sel[self.sel.find('of ')+3:]
        resSel = self.data.selection(resSel).getUniqueResidues()

        corr = np.zeros_like(resSel, dtype=float)

        oriList = ( (self.data.nbrFrames - dt) * np.random.random(self.nbrTimeOri) ).astype(int)


        for resId, residue in enumerate(resSel):

            for idx, frame in enumerate(oriList):

                print("Processing time origin %i of %i for residue %i of %i...        " 
                                    % (idx+1, oriList.size, resId+1, int(resSel.size)), end='\r')

                #_Processing selection text
                selText = self.sel

                selText += ' and resid %s' % residue


                sel = self.data.selection(selText + " frame %i %i" % (frame, frame+dt))
                
                corr[resId] += np.intersect1d(sel[0], sel[1]).size


        self.residueWise    = corr / np.max(corr)
        self.residues       = resSel.astype(int)
        self.residueWise_dt = dt

        


#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotResidenceTime(self):
        """ Used to quickly plot residence time. """


        fig, ax = plt.subplots()

        ax.plot(self.times, self.resTime, marker='o')
        ax.set_xlabel('Time [ps]')
        ax.set_ylabel('P(t)')

        fig.show()


    def plotResidueWiseResidenceTime(self):
        """ Used to quickly plot residue wise residence time. """

        fig, ax = plt.subplots()

        ax.bar(self.residues, self.residueWise)
        ax.set_xlabel('Residue')
        ax.set_ylabel('P(dt=%i ps)' % 
                            np.ceil(self.residueWise_dt*self.data.timestep*self.data.dcdFreq[0]*1e12))

        fig.show()



