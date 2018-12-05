import numpy as np
import re


class PDBReader:
    
    def __init__(self):

        #_Initializing the different entry types.
        self.headerList     = []
        self.titleList      = []
        self.atomList       = []
        self.hetatomList    = []
        self.anisouList     = []
        self.helixList      = []
        self.sheetList      = []
        self.seqresList     = []


    def importPDBFile(pdbFile):
        """ Imports a new file and store the result in self.pdbData.
            If something already exists in self.pdbData, it will be deleted. """

        #_Open the file and get the lines
        with open(pdbFile, 'r') as fileToRead:
            try:
                raw_data = fileToRead.read().splitlines()
            except:
                print("Error while reading the file.\n"
                      + "Please check the file path given in argument.")
                return 


        #_Creates two temporary lists to separate the different models/chains.
        tempAtomList = []
        tempHetAtomList = []
        tempAnisouList = []
        for line in raw_data:
            if re.search('^HEADER', line): 
                self.headerList.append(line)
            elif re.search('^TITLE', line):
                self.titleList.append(line)
            elif re.search('^SEQRES', line):
                self.seqresList.append(line)
            elif re.search('^HELIX', line):
                self.helixList.append(line)
            elif re.search('^SHEET', line):
                self.sheetList.append(line)
            elif re.search('^ATOM', line):
                tempAtomList.append(line)
            elif re.search('^HETATOM', line):
                tempHetAtomList.append(line)
            elif re.search('^ANISOU', line):
                tempAnisouList.append(line)

            #_If a 'TER' or a 'ENDMDL' is encountered, copy temporary lists into main ones, 
            #_and clear the former to start a new model/chain
            elif re.search('^TER', line):
                self.atomList.append(tempAtomList)
                self.hetatomList.append(tempHetAtomList)
                self.anisouList.append(tempAnisouList)
                tempAtomList.clear()
                tempHetAtomList.clear()
                tempAnisouList.clear()
            elif re.search('^ENDMDL', line):
                self.atomList.append(tempAtomList)
                self.hetatomList.append(tempHetAtomList)
                self.anisouList.append(tempAnisouList)
                tempAtomList.clear()
                tempHetAtomList.clear()
                tempAnisouList.clear()
 
