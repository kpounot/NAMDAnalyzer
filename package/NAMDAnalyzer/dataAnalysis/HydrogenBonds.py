import sys
import numpy as np

import matplotlib.pyplot as plt



class HydrogenBonds:

    def __init__(self, data):
        """ This class defines methods to compute hydrogen bonds auto-correlation functions. 
            
            Some plotting methods are also available to quicly check the results. 
            
            Input: self.data -> a Dataset class instance containing trajectories data """

        self.data = data

        self.cc = None
        self.ic = None



    def processDonors(self, donors):
        """ This function takes a list of indices corresponding to selected hydrogen bond donor atoms.
            Then, bound hydrogens are found and two lists of same size are returned. One containing 
            all hydrogens, and the other their associated donors. """


        allH    = self.data.selection('hbhydrogens')

        keepIdx = np.zeros(allH.shape[0], dtype=int)


        bonds1  = self.psfData.bonds[:,::2]
        bonds2  = self.psfData.bonds[:,1::2]

        selBonds1   = np.argwhere( np.isin(bonds1, self.psfData.atoms[selection][:,0].astype(int)) )
        selBonds2   = np.argwhere( np.isin(bonds2, self.psfData.atoms[selection][:,0].astype(int)) )

        keepIdx[ bonds2[selBonds1[:,0], selBonds1[:,1]] - 1 ] = 1
        keepIdx[ bonds1[selBonds2[:,0], selBonds2[:,1]] - 1 ] = 1


        return outDonors, hydrogens
        


#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compContinuousAC(self, acceptors, donors, dt=20, maxTime=500, nbrTimeOri=25, 
                                                                        maxR=2.5, maxAngle=130):
        """ Computes the continuous hydrogen bonds autocorrelation function.

            Input:  acceptors   -> selection of acceptors atoms for hydrogen bonds
                    donors      -> selection of donors atoms for hydrogen bonds
                    dt          -> number of time intervals to compute for each origin (optional, default 20)
                    maxTime     -> maximum time interval to be used for correlation in number of frames
                                    (optional, default 100)
                    nbrTimeOri  -> number of time origins to be averaged over (optional, default 25)
                    maxR        -> maximum distance to allow for hydrogen bonding, angström 
                                    (optional, default 2.5 - acceptor-hydrogen distance)
                    minAngle    -> minimum angle to allow for hydrogen bonding (optional, default 130)

            The result, an 1D array containing correlations for all time intervals is stored 
            in self.cc variable. """

        #_Parsing selection arguments
        if type(acceptors) == str:
            acceptors = self.data.selection(acceptors)

        if type(donors) == str:
            donors = self.data.selection(donors)


        donors, hydrogens = self.processDonors(donors)



        angleFactor = np.cos(minAngle) #_To quickly check angle using dot product
        hbonds = np.zeros( (acceptors.size, donors.size), dtype=int ) #_To add 1 for bounded pairs

        for tIdx, timeIncr in enumerate( range(0, maxTime, dt) ):
            frameIncr = int( timeIncr / (self.dataset.timestep * self.dataset.dcdFreq[0]) ) 

            for oriIdx in range(0, self.dataset.nbrFrames, self.dataset.nbrFrames / 25):
                pass



#---------------------------------------------
#_Plotting methods
#---------------------------------------------
