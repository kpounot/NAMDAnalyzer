import os, sys
import numpy as np
import re


class NAMDPDB:
    """ This class takes a PDB file as input.
        The different types of entries (ATOM, HETATOM,...) are stored in separate lists of lists.
        In case of an 'TER' or a water, the new chain's atoms list is simply append to the main list. """

    def __init__(self, pdbFile, parent=None):

        self.parent = parent

        #_Open the file and get the lines
        with open(pdbFile, 'r') as fileToRead:
            try:
                raw_data = fileToRead.read().splitlines()
            except:
                print("Error while reading the file.\n"
                      + "Please check the file path given in argument.")
                return 

        #_Initializing the different entry types.
        self.headerList     = []
        self.titleList      = []
        self.atomList       = []
        self.hetatomList    = []
        self.anisouList     = []
        self.helixList      = []
        self.sheetList      = []
        self.seqresList     = []

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
            

    #_____________________________________________
    #_Data accession methods
    #_____________________________________________

    #_____________________________________________
    #_Plotting methods
    #_____________________________________________


