"""

Classes
^^^^^^^

"""

import sys

import numpy as np

import matplotlib.pyplot as plt


class Rotations:
    """ This class defines methods to compute rotational relaxation and orientation probabilities. 
        
        Some plotting methods are also available to quicly check the results. 
        
        :arg data:        a Dataset class instance containing trajectories data 
        :arg sel1:        first selection corresponding to one end of each vector
        :arg sel2:        second selection for vectors, should be of same size as sel1
        :arg tMax:        maximum number of frames to be used 
        :arg step:        time interval between each computed vectors
        :arg dPhi:        angular bin size for orientational probability (in degrees)
        :arg axis:        reference axis for orientation probabilities
        :arg nbrTimeOri:  number of time origins to be averaged over (optional, default 25) """


    def __init__(self, data, sel1, sel2, tMax=100, step=1, dPhi=0.5, axis='z', nbrTimeOri=20):
                                                                                                            
        self.data       = data
        self.sel1       = sel1
        self.sel2       = sel2
        self.tMax       = tMax
        self.step       = step
        self.dPhi       = dPhi / 180 * np.pi
        self.axis       = axis
        self.nbrTimeOri = nbrTimeOri

        self.rotCorr    = None
        self.rotDensity = None
        self.times      = None
        self.angles     = None



#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compRotationalRelaxation(self):
        """ For each frame in the range of tMax with gieven step, computes the distance vector between 
            sel1 and sel2 and performs scalar product between the first one and all other one. 
            This is averaged over multiple time origins and the obtained correlation is stored
            in self.rotRelax variable. 
            
            References: 
                
                - Yu-ling Yeh and Chung-Yuan Mou (1999). Orientational Relaxation 
                  Dynamics of Liquid Water Studied by Molecular Dynamics Simulation, 
                  J. Phys. Chem. B 1999, 103, 3699-3705. 

        """


        self.times  = ( np.arange(0, self.tMax, self.step, dtype=int) 
                        * self.data.dcdFreq[0] * self.data.timestep * 1e12 )
        corr        = np.zeros_like(self.times)

        oriList = ( (self.data.nbrFrames - self.tMax) * np.random.random(self.nbrTimeOri) ).astype(int)
        for idx, frame in enumerate(oriList):

            sel1 = self.sel1 + ' frame %i' % frame
            sel2 = self.sel2 + ' frame %i' % frame

            sel1 = self.data.dcdData[self.data.selection(sel1), frame:frame+self.tMax:self.step]
            sel2 = self.data.dcdData[self.data.selection(sel2), frame:frame+self.tMax:self.step]

            angles  = sel2 - sel1
            angles  = angles / np.sqrt( np.sum( angles**2, axis=2 ) )[:,:,np.newaxis]

            angles  = np.sum(angles[:,[0]] * angles, axis=2)

            corr += np.mean( (3*angles**2 - 1) / 2, axis=0) / self.nbrTimeOri


        self.rotCorr    = corr


        



    def compOrientationalProb(self):
        """ Compute the probability for the vector between sel1 and sel2 to be in a 
            particular orientation.
            The angle is computed with respect to a given axis. 
            Averaging is performed for each frame between 0 and tMax with given step. 

        """

        if self.axis=='x':
            ref = np.array( [[[1, 0, 0]]] )
        elif self.axis=='y':
            ref = np.array( [[[0, 1, 0]]] )
        elif self.axis=='z':
            ref = np.array( [[[0, 0, 1]]] )


        self.angles     = np.arange(0, np.pi, self.dPhi)
        self.rotDensity = np.zeros_like( self.angles )
        

        sel1 = self.data.dcdData[self.data.selection(self.sel1)]
        sel2 = self.data.dcdData[self.data.selection(self.sel2)]

        angles  = sel2 - sel1
        angles  = angles / np.sqrt( np.sum( angles**2, axis=2 ) )[:,:,np.newaxis]

        angles  = np.arccos( np.sum(ref * angles, axis=2) )
        angles = angles.flatten() 

        normF = angles.size
        for i, val in enumerate( self.angles ):
            self.rotDensity[i] = angles[angles < val].size / normF
            angles = angles[angles >= val]




#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotRotationalRelaxation(self):
        """ Used to quickly plot rotational relaxation function """


        fig, ax = plt.subplots()

        ax.plot(self.times, self.rotCorr, marker='o')
        ax.set_xlabel('Time [ps]')
        ax.set_ylabel('$C_{\hat{u}}(t)$')

        fig.show()



    def plotOrientationalProb(self):
        """ Used to quickly plot orientational probability """


        fig, ax = plt.subplots()

        ax.plot(self.angles, self.rotDensity)
        ax.set_xlabel('$\\theta \ [rad]$')
        ax.set_ylabel('$P(\\theta)$')

        fig.show()

