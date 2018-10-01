import os, sys
import numpy as np
import re

from collections import namedtuple

class NAMDPSF:
    """ This class is used for .psf file reading. """

    def __init__(self, psfFile, parent=None):

        self.parent = parent
        
        with open(psfFile, 'r') as f:
            try:
                data = f.read().splitlines()
            except:
                print("Error while reading the file.\n"
                      + "Please check the file path given in argument.")
                return 

        dataTuple = namedtuple('dataTuple', 'atoms bonds angles dihedrals impropers donors acceptors '
                                          + 'nonBonded xterms')

        #_Initialize the temporary lists (avoid errors in case they're not created below
        atomList        = []
        bondList        = []
        thetaList       = []
        phiList         = []
        imprpList       = []
        donorsList      = []
        acceptorsList   = []
        nonBondedList   = []
        xTermsList      = []

        pattern = re.compile('\s+[0-9]+\s![A-Z]+') #_Entry category identifier
        for i, line in enumerate(data):
            if pattern.match(line):
                nbrEntries = int(line.split()[0]) #_Get the number of lines to read for the category
                #_Each category data is stored in a corresponding temporary array, which will be
                #_then copied and protected from modifications in the namedtuple.
                if re.search('ATOM', line):
                    atomList = np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('BOND', line):
                    bondList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('THETA', line):
                    thetaList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('PHI', line):
                    phiList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('IMPHI', line):
                    imprpList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('DON', line):
                    donorsList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('ACC', line):
                    acceptorsList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('NNB', line):
                    nonBondedList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])
                elif re.search('CRTERM', line):
                    xTermsList =  np.array([ entry.split() for entry in data[i+1:i+1+nbrEntries] ])

        self.dataSet = dataTuple( atomList, bondList, thetaList, phiList, imprpList, 
                                donorsList, acceptorsList, nonBondedList, xTermsList)


    #_____________________________________________
    #_Data accession methods
    #_____________________________________________
    def getAtomsMasses(self, begin=0, end=None):
        """ Return the column corresponding to atoms masses as an 1D array of float.

            Input:  begin   -> starting index
                    end     -> last index """ 

        return self.dataSet.atoms[begin:end,7].astype(float)

 

    #_____________________________________________
    #_Plotting methods
    #_____________________________________________

