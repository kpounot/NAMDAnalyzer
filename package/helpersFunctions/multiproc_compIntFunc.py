import numpy as np

from multiprocessing import Process, Lock



def compIntermediateFunc(qValList, minFrames, maxFrames, nbrBins=60, selection='protNonExchH', 
                                                                                    begin=0, end=None):
        """ This method computes intermediate function for all q-value (related to scattering angle)

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    selection   -> atom selection
                    begin       -> first frame to be used
                    end         -> last frame to be used 
                    
            Returns an (nbr of q-values, timesteps) shaped array. """

        print("Computing intermediate scattering function...")

        qValList = np.array(qValList)
        self.qVals = qValList

        #_Computes atoms positions
        atomPos = self.alignCenterOfMass(selection, begin, end)

        #_Computes random q vectors
        qArray = []
        for qIdx, qVal in enumerate(qValList):
            qList = [CM.getRandomVec(qVal) for i in range(20)] 
            qArray.append( np.array(qList).T )

        qArray = np.array(qArray)

        corr = np.zeros( (qValList.size, nbrBins), dtype='c16') 
        timestep = []

        for it in range(nbrBins):
            print("Computing bin: %i/%i" % (it+1, nbrBins), end='\r')
            nbrFrames   = minFrames + int(it * (maxFrames - minFrames) / nbrBins )

            #_Compute time step
            timestep.append(self.timestep * nbrFrames * self.dcdFreq[0])

            #_Defines the number of time origins to be averaged on
            #_Speeds up computation and helps to prevent MemoryError for large arrays
            incr = int(atomPos.shape[1] / 25) 

            #_Computes intermediate scattering function for one timestep, averaged over time origins
            displacement = atomPos[:,nbrFrames::incr] - atomPos[:,:-nbrFrames:incr]

            pList = []
            for qIdx, qVecs in enumerate(qArray):
                pList.append( Thread(target=self.threadedInterFunc, 
                                            args=(displacement, qIdx, qVecs, corr, it)) )
        
                pList[qIdx].start()
        
            for p in pList:
                p.join()
            

        self.interFunc = corr, np.array(timestep)
        print("\nDone\n")




    def threadedInterFunc(self, displacement, qIdx, qVecs, corr, it):
        with RLock():
            #_Dotting with random q vectors -> shape (nbr atoms, nbr time origins, nbr vectors)
            temp = 1j * np.dot( displacement, qVecs )
            np.exp( temp, out=temp )

            temp = temp.mean() #_Average over time origins, q vectors and atoms

            corr[qIdx,it] += temp


