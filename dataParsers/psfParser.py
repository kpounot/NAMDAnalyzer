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

        #_Defining some useful attributes
        self.protSel        = ["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TRY", "PRO"
                              "SER", "THR", "CYS", "TYR", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS"]

        self.backboneSel    = ["C", "O", "N", "HN", "CA", "HA"]


    #---------------------------------------------
    #_Data accession methods
    #---------------------------------------------
    def getAtomsMasses(self, selection):
        """ Return the column corresponding to atoms masses as an 1D array of float.

            Input:  firstAtom   -> starting index
                    lastAtom    -> last index """ 

        return self.dataSet.atoms[selection,7].astype(float)


    def getSelection(self, selText="all", segList=None, resList=None, nameList=None, index=None):
        """ This method returns an list of index corresponding to the ones that have been selected 
            using the 'selText' argument and the indices list.

            Possible selText are:   - all
                                    - protein
                                    - backbone
                                    - water

            For segList, resList, nameList: - segment id, or list of segment id
                                            - any residue number, or list of residue number
                                            - any atom name, or list of atom names 

            Argument index should be a list, which can be generated with range in case of a range.
                                    
            In case the user wants to the protein and a given segment id, the following argument can
            be entered ['protein', 'segID_name']. Then, for every selection, the index lists are 
            compared and only the indices that appear in all lists are kept. """


        #_Converting to lists on case of single string
        if type(segList) == str:
            segList = [segList]
        if type(resList) == str:
            resList = [resList]
        if type(nameList) == str:
            nameList = [nameList]

        keepIdxList = []

        #_Getting the different index lists corresponding to the given selection(s)
        if selText == "all":
            keepIdxList.append( np.ones_like(self.dataSet.atoms[:,0]).astype(bool) )
        if selText == "protein":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,3], self.protSel) )
        if selText == "backbone":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,4], self.backboneSel) )
        if selText == "water":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,3], "TIP3") )

        #_Parsing the segment list if not None
        if segList:
            segList = np.array(segList).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,1], segList) )

        #_Parsing the residue list if not None
        if resList:
            resList = np.array(resList).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,2], resList) )

        #_Parsing the name list if not None
        if nameList:
            keepIdxList.append( np.isin(self.dataSet.atoms[:,4], nameList) )

        #_Parsing the index list if not None
        if index:
            index   = np.array(index).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,0], index) )


        #_Using bitwise AND to keep only the indices that are true everywhere
        if len(keepIdxList) > 1:
            for i in range(1, len(keepIdxList)):
                keepIdxList[0] = np.bitwise_and(keepIdxList[0], keepIdxList[i])

        #_Using argwhere to return the indices corresponding to the True values
        return np.argwhere(keepIdxList[0])

            
