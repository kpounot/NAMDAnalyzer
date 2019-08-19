import sys

import numpy as np

import matplotlib.pyplot as plt


class RetentionTime:

    def __init__(self, data, sel, tMax=25, step=1, nbrTimeOri=20):
        """ This class defines methods to compute retention time of atoms in a certain region.
            This determines how fast atoms in sel2 can leave the vicinity of sel1.
            
            Some plotting methods are also available to quicly check the results. 
            
            Input:  data        -> a Dataset class instance containing trajectories data 
                    sel         -> selection corresponding to analysis region (with 'within' keyword')
                    tMax        -> maximum number of frames to be used 
                    step        -> time interval between each computed frame
                    nbrTimeOri  -> number of time origins to be averaged over (optional, default 20) """

                                                                                                            
        self.data       = data
        self.sel        = sel 
        self.tMax       = tMax
        self.step       = step
        self.nbrTimeOri = nbrTimeOri

        self.retTime    = None
        self.times      = None



#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compRetentionTime(self):
        """ For each frame in the range of tMax with given step, the number of atoms in selected region
            is computed and divided by the number of atoms at time origin. The result is averaged over
            multiple time origins. """


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
                corr[tIdx] /= self.nbrTimeOri


        self.retTime = corr / corr[0]



        


#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotRetentionTime(self):
        """ Used to quickly plot retention time. """


        fig, ax = plt.subplots()

        ax.plot(self.times, self.retTime, marker='o')
        ax.set_xlabel('Time [ps]')
        ax.set_ylabel('P(t)')

        fig.show()

