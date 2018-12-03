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

        self.protH          = ["HN", "HA", "HB", "HA1", "HA2", "HB1", "HB2", "HB3", "HD1", "HD2", "HD11", 
                               "HD12", "HD13", "HD21", "HD22", "HD23", "HD2", "HE1", "HE2", "HG1", "HG2",
                               "HG11", "HG12", "HG13", "HG21", "HG22", "HG23", "HZ1", "HZ2", "HZ3", "HH", 
                               "OH", "HH11", "HH12", "HH21", "HH22", "HE", "HE3", "HE21", "HE22", "HD3", "HZ",
                               "HT1", "HT2", "HT3"]

        self.protExchH   = {'ALA': ['HN'],
                            'ARG': ['HN', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
                            'ASN': ['HN', 'HD21', 'HD22'],
                            'ASP': ['HN'],
                            'CYS': ['HN', 'HG1'],
                            'GLN': ['HN', 'HE21', 'HE22'],
                            'GLU': ['HN'],
                            'GLY': ['HN'],
                            'HSD': ['HN', 'HD1', 'HD2'],
                            'HSE': ['HN', 'HE2'],
                            'HSP': ['HN', 'HD1', 'HE2'],
                            'ILE': ['HN'],
                            'LEU': ['HN'],
                            'LYS': ['HN', 'HZ1', 'HZ2', 'HZ3'],
                            'MET': ['HN'],
                            'PHE': ['HN'],
                            'PRO': ['None'],
                            'SER': ['HN', 'HG1'],
                            'THR': ['HN', 'HG1'],
                            'TRP': ['HN', 'HE1'],
                            'TYR': ['HN', 'HH'],
                            'VAL': ['HN']}

        self.waterH         = ["H1", "H2"]


    #---------------------------------------------
    #_Data accession methods
    #---------------------------------------------
    def getAtomsMasses(self, selection):
        """ Return the column corresponding to atoms masses as an 1D array of float.

            Input:  firstAtom   -> starting index
                    lastAtom    -> last index """ 

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.parent.psfData.getSelection(selection)

        return self.dataSet.atoms[selection,7].astype(float)


    def getSelection(self, selText="all", segName=None, NOTsegName=None, resNbr=None, resName=None, 
                            NOTresNbr=None, NOTresName=None, atom=None, NOTatom=None, index=None):
        """ This method returns an list of index corresponding to the ones that have been selected 
            using the 'selText' argument and the indices list.

            Possible selText are:   - all
                                    - protein
                                    - backbone
                                    - protH
                                    - protNonExchH
                                    - water
                                    - waterH

            For segName, resNbr, resName, atom:  - segment id, or list of segment id
                                                 - any residue number, or list of residue number
                                                 - any residue name, or list of residue names
                                                 - any atom name, or list of atom names 

            Argument index should be a list, which can be generated with range in case of a range.

            Some NOT... arguments can be provided as well to eliminate some entries. 
                                    
            In case the user wants to the protein and a given segment id, the following argument can
            be entered ['protein', 'segID_name']. Then, for every selection, the index lists are 
            compared and only the indices that appear in all lists are kept. """


        #_Converting to lists on case of single string
        if type(segName) == str:
            segName = [segName]
        if type(resNbr) == str:
            resNbr = [resNbr]
        if type(atom) == str:
            atom = [atom]

        keepIdxList = []

        #_Getting the different index lists corresponding to the given selection(s)
        if selText == "all":
            keepIdxList.append( np.ones_like(self.dataSet.atoms[:,0]).astype(bool) )
        if selText == "protein":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,3], self.protSel) )
        if selText == "protH":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,4], self.protH) )
        if selText == "backbone":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,4], self.backboneSel) )
        if selText == "water":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,3], "TIP3") )
        if selText == "waterH":
            keepIdxList.append( np.isin(self.dataSet.atoms[:,4], self.waterH) )

        if selText == "protNonExchH":
            keepIdxList = np.ones(self.dataSet.atoms.shape[0]).astype(bool)
            keepIdxList = np.bitwise_and(keepIdxList, np.isin(self.dataSet.atoms[:,3], self.protSel))
            for key, value in self.protExchH.items():
                resArray        = np.isin(self.dataSet.atoms[:,3], key)
                exchHArray      = np.isin(self.dataSet.atoms[:,4], value)
                nonExchHArray   = np.invert(np.bitwise_and(resArray, exchHArray))
                keepIdxList     = np.bitwise_and(keepIdxList, nonExchHArray)

            return np.argwhere(keepIdxList)[:,0]


        #_Parsing the segment list if not None
        if segName:
            segName = np.array(segName).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,1], segName) )

        #_Parsing the not segment list if not None
        if NOTsegName:
            NOTsegName = np.array(NOTsegName).astype(str)
            keepIdxList.append( np.invert(np.isin(self.dataSet.atoms[:,1], NOTsegName)) )

        #_Parsing the residue number list if not None
        if resNbr:
            resNbr = np.array(resNbr).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,2], resNbr) )

        #_Parsing the not residue number list if not None
        if NOTresNbr:
            NOTresNbr = np.array(NOTresNbr).astype(str)
            keepIdxList.append( np.invert(np.isin(self.dataSet.atoms[:,2], NOTresNbr)) )

        #_Parsing the residue name list if not None
        if resName:
            resName = np.array(resName).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,3], resNbr) )

        #_Parsing the not residue name list if not None
        if NOTresName:
            NOTresName = np.array(NOTresName).astype(str)
            keepIdxList.append( np.invert(np.isin(self.dataSet.atoms[:,3], NOTresName)) )

        #_Parsing the name list if not None
        if atom:
            atom = np.array(atom).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,4], atom) )

        #_Parsing the not atom list if not None
        if NOTatom:
            NOTatom = np.array(NOTatom).astype(str)
            keepIdxList.append( np.invert(np.isin(self.dataSet.atoms[:,4], NOTatom)) )

        #_Parsing the index list if not None
        if index:
            index   = np.array(index).astype(str)
            keepIdxList.append( np.isin(self.dataSet.atoms[:,0], index) )


        #_Using bitwise AND to keep only the indices that are true everywhere
        if len(keepIdxList) > 1:
            for i in range(1, len(keepIdxList)):
                keepIdxList[0] = np.bitwise_and(keepIdxList[0], keepIdxList[i])

        #_Using argwhere to return the indices corresponding to the True values
        return np.argwhere(keepIdxList[0])[:,0]

