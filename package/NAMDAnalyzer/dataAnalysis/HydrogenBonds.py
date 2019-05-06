import sys
import numpy as np

import matplotlib.pyplot as plt



class HydrogenBonds:

    def __init__(self, dataset):
        """ This class defines methods to compute hydrogen bonds auto-correlation functions. 
            
            Some plotting methods are also available to quicly check the results. 
            
            Input: self.dataset -> a self.dataset class instance containing trajectories data """

        self.dataset = dataset

        self.cc = None
        self.ic = None


#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compContinuousAC(self, acceptors, donors, hydrogens, dt=20, maxTime=500, nbrTimeOri=25, 
                                                                                maxR=3.5, maxAngle=35):
        """ Computes the continuous hydrogen bonds autocorrelation function.

            Input:  acceptors   -> selection of acceptors atoms for hydrogen bonds, must be an array of indices
                    donors      -> selection of donors atoms for hydrogen bonds, must be an array of indices
                    hydrogens   -> selection of hydrogens that are bound to donors, must be of same size
                                   as donors or twice its size (for water donors for instance), 
                                   must be an array of indices
                    dt          -> time interval for which to compute correlations in ps 
                                    (optional, default 20)
                    maxTime     -> maximum time interval to be used for correlation in ps
                                    (optional, default 500)
                    nbrTimeOri  -> number of time origins to be averaged over (optional, default 25)
                    maxR        -> maximum distance to allow for hydrogen bonding, angström 
                                    (optional, default 3.5)
                    maxAngle    -> maximum angle to allow for hydrogen bonding (optional, default 35)

            The result, an 1D array containing correlations for all time intervals is stored 
            in self.cc variable. """

        angleFactor = np.cos(maxAngle) #_To quickly check angle using dot product

        hbonds = np.zeros( (acceptors.size, donors.size) )

        for tIdx, timeIncr in enumerate( range(0, maxTime, dt) ):
            frameIncr = int( timeIncr / ( self.dataset.timestep * 

            for oriIdx, ori in range(0, self.dataset.nbrFrames, self.dataset.nbrFrames / 25):
                pass

