import numpy as np
import re

from collections import namedtuple

class PSFReader:

    def __init__(self):

        self.psfData = None

        #_Defining some useful attributes
        self.protSel        = ["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TRY", "PRO",
                              "SER", "THR", "CYS", "TYR", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS",
                              "HSE", "HSP"," HSD"]

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

        self.H              = re.compile('[A-Za-z0-9]*H[A-Za-z0-9]*')


    def importPSFFile(self, psfFile):
        """ Imports a new file and store the result in self.psfData.
            If something already exists in self.psfData, it will be deleted. """

        self.psfFile = psfFile

        with open(psfFile, 'r') as f:
            data = f.read().splitlines()

        dataTuple = namedtuple('dataTuple', 'atoms bonds angles dihedrals impropers donors acceptors '
                                          + 'nonBonded xterms')

        #_Initialize the temporary lists        
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

        self.psfData = dataTuple( atomList, bondList, thetaList, phiList, imprpList, 
                                donorsList, acceptorsList, nonBondedList, xTermsList)


