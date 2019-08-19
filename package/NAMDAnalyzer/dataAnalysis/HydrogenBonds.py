import sys

import numpy as np

import matplotlib.pyplot as plt

from ..lib.pygetHydrogenBonds import py_getHydrogenBonds


class HydrogenBonds:

    def __init__(self, data, acceptors='hbacceptors', donors='hbdonors', maxTime=50, nbrTimeOri=20, 
                                                                        step=1, maxR=2.5, minAngle=130):
        """ This class defines methods to compute hydrogen bonds auto-correlation functions. 
            
            Some plotting methods are also available to quicly check the results. 
            
            Input:  data        -> a Dataset class instance containing trajectories data 
                    acceptors   -> selection of acceptors atoms for hydrogen bonds (string)
                    donors      -> selection of donors atoms for hydrogen bonds (string)
                    maxTime     -> maximum time interval to be used for correlation in number of frames
                                    (optional, default 100)
                    step        -> number of frames between each time interval points (optional, default 1) 
                    nbrTimeOri  -> number of time origins to be averaged over (optional, default 25)
                    maxR        -> maximum distance to allow for hydrogen bonding, angström 
                                    (optional, default 2.5 - acceptor-hydrogen distance)
                    minAngle    -> minimum angle to allow for hydrogen bonding (optional, default 130) 
                    
            References: - D.C. Rapaport (1983): Hydrogen bonds in water, 
                          Molecular Physics: An International Journal at the 
                          Interface Between Chemistry and Physics, 50:5, 1151-1162. """


        self.data       = data

        self.acceptors  = acceptors
        self.donors     = donors

        self.step       = step
        self.maxTime    = maxTime
        self.nbrTimeOri = nbrTimeOri
        self.maxR       = maxR
        self.minAngle   = minAngle

        self.times = None

        self.cc = np.ascontiguousarray( np.zeros( self.times.size ), dtype='float32' )
        self.ic = np.ascontiguousarray( np.zeros( self.times.size ), dtype='float32' )


    def _processDonors(self, donors):
        """ This function takes a list of indices corresponding to selected hydrogen bond donor atoms.
            Then, bound hydrogens are found and two lists of same size are returned. One containing 
            all hydrogens, and the other their associated donors. """

        donors = self.data.selection(donors)

        #_Gets hydrogens bound to donors
        allH    = self.data.selection('hbhydrogens')
        boundH  = np.argwhere( self.data.getBoundAtoms(donors) )[:,0]

        boundH = np.intersect1d(boundH, allH)
        boundH.sort()


        #_Finds donor atom linked to each selected hydrogen. Generates a donor list of same siez as boundH
        bonds1  = self.data.psfData.bonds[:,::2]
        bonds2  = self.data.psfData.bonds[:,1::2]

        selBonds1   = np.argwhere( np.isin(bonds1, self.data.psfData.atoms[boundH][:,0].astype(int)) )
        selBonds2   = np.argwhere( np.isin(bonds2, self.data.psfData.atoms[boundH][:,0].astype(int)) )

        outDonors = np.concatenate( (bonds2[selBonds1[:,0], selBonds1[:,1]],
                                     bonds1[selBonds2[:,0], selBonds2[:,1]]) ) - 1
        outDonors.sort()

        #_Keeps the donors only
        outDonors = outDonors[ np.argwhere(np.isin(outDonors, donors))[:,0] ]



        return outDonors, boundH
        


#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compAutoCorrel(self, continuous=1):
        """ Computes the hydrogen bonds autocorrelation function.

            Both distances and angles are computed exactly, without any approximation to single-point
            precision.

            The result, a 1D array containing correlations for all time intervals is stored 
            in self.cc or self.ic variable for continuous or intermittent types respectively. """

        if continuous==1:
            self.cc *= 0
        else:
            self.ic *= 0


        self.times = ( np.arange(0, self.maxTime, self.step, dtype=int) 
                                * self.data.dcdFreq[0] * self.data.timestep * 1e12 )

        #_To store final result for each time interval
        corr = np.ascontiguousarray( np.zeros( self.times.size ), dtype='float32' )
        
        oriList = ( (self.data.nbrFrames - self.maxTime) * np.random.random(self.nbrTimeOri) ).astype(int)
        for idx, frame in enumerate(oriList):

            print( "Computing time origin %i of %i with %i frames..." % 
                                        (idx+1, self.nbrTimeOri, self.times.size), end='\r' )

            corr *= 0 #_Reset corr to its initial state

            #_Updating selection for each time origin
            acceptors           = self.data.selection( self.acceptors + ' frame %i' % frame)
            donors, hydrogens   = self._processDonors( self.donors + ' frame %i' % frame)


            #_Get cell dimensions for given frame and make array contiguous
            cellDims = np.ascontiguousarray(self.data.cellDims[frame:frame+self.maxTime:self.step], 
                                                                                        dtype='float32')

            py_getHydrogenBonds(self.data.dcdData[acceptors, frame:frame+self.maxTime:self.step], 
                                self.times.size,
                                self.data.dcdData[donors, frame:frame+self.maxTime:self.step], 
                                self.data.dcdData[hydrogens, frame:frame+self.maxTime:self.step], 
                                cellDims, corr, self.maxTime,
                                self.step, self.nbrTimeOri, self.maxR, self.minAngle, continuous)



            if continuous==1:
                self.cc += corr / (corr[0] * self.nbrTimeOri)
            else:
                self.ic += corr / (corr[0] * self.nbrTimeOri)



                     



#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotAutoCorrel(self, corrType='continuous'):
        """ Used to quickly plot autocorrelation function, either continuous or intermittent depending on the
            value of corrType parameter. """


        if corrType=='continuous':
            data = self.cc
            yLabel = '$C_{continuous}(t)$'
        else:
            data = self.ic
            yLabel = '$C_{intermittent}(t)$'

        plt.plot(self.times, data, marker='o')
        plt.xlabel('Time [ps]')
        plt.ylabel(yLabel)

        plt.show(block=False)