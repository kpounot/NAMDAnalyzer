import os, sys
import numpy as np
import re

from collections import namedtuple

from .psfReader import PSFReader

class NAMDPSF(PSFReader):
    """ This class is used for .psf file reading. """

    def __init__(self, psfFile=None):

        PSFReader.__init__(self)

        if psfFile:
            self.importPSFFile(psfFile)

        #_Defining some useful attributes
        self.protSel        = ["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TRY", "PRO",
                              "SER", "THR", "CYS", "TYR", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS",
                              "HSE", "HSP"," HSD"]

        self.backboneSel    = ["C", "O", "N", "HN", "CA", "HA"]

        self.protH          = ["HN", "HA", "HB", "HA1", "HA2", "HB1", "HB2", "HB3", "HD1", "HD2", "HD11", 
                               "HD12", "HD13", "HD21", "HD22", "HD23", "HD2", "HE1", "HE2", "HG1", "HG2",
                               "HG11", "HG12", "HG13", "HG21", "HG22", "HG23", "HZ1", "HZ2", "HZ3", "HH", 
                               "HH11", "HH12", "HH21", "HH22", "HE", "HE3", "HE21", "HE22", "HD3", "HZ",
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

        self.H              = np.concatenate( (self.protH, self.waterH) )


        self.HDonors        = ["N", "OH2", "OW", "NE", "NH1", "NH2",
                               "ND2", "SG", "NE2", "ND1", "NE2", "NZ",
                               "OG", "OG1", "NE1", "OH"]

        self.HAcceptors     = ["O", "OC1", "OC2", "OH2", "OW", "OD1", "OD2", "SG", "OE1",
                               "OE2", "ND1", "NE2", "SD", "OG", "OG1", "OH"]



#---------------------------------------------
#_Data accession methods
#---------------------------------------------
    def getAtomsMasses(self, selection):
        """ Return the column corresponding to atoms masses as an 1D array of float.

            Input:  selection   -> a selection of atom indices """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.selection(selection)

        return self.psfData.atoms[selection,7].astype(float)




    def getSelection(self, selText="all", segName=None, resID=None, resName=None, atom=None, 
                                                                        index=None, invert=False):
        """ This method returns an list of index corresponding to the ones that have been selected 
            using the 'selText' argument and the indices list.

            Possible selText are:   - all
                                    - protein
                                    - backbone
                                    - protH or proteinH
                                    - protNonExchH
                                    - water
                                    - waterH
                                    - hydrogen
                                    - hbdonors
                                    - hbacceptors

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
        if type(resID) == str:
            resID = [resID]
        if type(atom) == str:
            atom = [atom]

        keepIdxList = []

        #_Getting the different index lists corresponding to the given selection(s)
        if selText == "all":
            keepIdxList.append( np.ones_like(self.psfData.atoms[:,0]).astype(bool) )

        if selText == "hydrogen":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.H, invert=invert) )

        if selText == "hbdonors":
            keepList = np.isin(self.psfData.atoms[:,4], self.HDonors, invert=invert) 
            keepIdxList.append( self.getHBDonors(keepList) )

        if selText == "hbacceptors":
            keepList = np.isin(self.psfData.atoms[:,4], self.HAcceptors, invert=invert) 
            keepIdxList.append( self.getHBAcceptors(keepList) )

        if selText == "hbhydrogens":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.H) )

            keepList = np.isin(self.psfData.atoms[:,4], self.HDonors) 
            keepList = self.getHBDonors(keepList) 
            keepList = np.argwhere(keepList)[:,0]
            keepIdxList.append( self.getBoundAtoms(keepList) )

        if selText == "protein":
            keepIdxList.append( np.isin(self.psfData.atoms[:,3], self.protSel, invert=invert) )

        if selText == "protH" or selText == "proteinH":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.protH, invert=invert) )

        if selText == "backbone":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.backboneSel, invert=invert) )

        if selText == "water":
            keepIdxList.append( np.isin(self.psfData.atoms[:,3], ["TIP3", "TIP4", "TIP5", "SPC", "SPCE",
                                                                    "TP3B", "TP3F", "TP4E", "TP45", "TP5E"],
                                                                invert=invert) )
        if selText == "waterH":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.waterH, invert=invert) )

        if selText == "protNonExchH":
            keepList = np.isin(self.psfData.atoms[:,4], self.protH)
            for key, value in self.protExchH.items():
                resArray        = np.isin(self.psfData.atoms[:,3], key)
                exchHArray      = np.isin(self.psfData.atoms[:,4], value)
                nonExchHArray   = np.invert(np.bitwise_and(resArray, exchHArray))
                keepList        = np.bitwise_and(keepList, nonExchHArray)

            keepIdxList.append(keepList)



        #_Parsing the segment list if not None
        if segName:
            segName = np.array(segName).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,1], segName, invert=invert) )

        #_Parsing the residue number list if not None
        if resID:
            resID = np.array(resID).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,2], resID, invert=invert) )


        #_Parsing the residue name list if not None
        if resName:
            resName = np.array(resName).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,3], resName, invert=invert) )


        #_Parsing the name list if not None
        if atom:
            atom = np.array(atom).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], atom, invert=invert) )


        #_Parsing the index list if not None
        if index:
            index   = np.array(index).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,0], index, invert=invert) )


        #_Using bitwise AND to keep only the indices that are true everywhere
        if len(keepIdxList) > 1:
            for i in range(1, len(keepIdxList)):
                keepIdxList[0] = np.bitwise_and(keepIdxList[0], keepIdxList[i])

        #_Using argwhere to return the indices corresponding to the True values
        return np.argwhere(keepIdxList[0])[:,0]





    def getSameResidueAs(self, selection):
        """ Given the provided selection, selects all others atoms that are present in the same residues and
            returns an updated selection. """


        if type(selection) == int:
            sel = self.psfData.atoms[[selection]]
        elif selection.shape[0] == 1:
            sel = self.psfData.atoms[selection]
        else:
            sel = self.psfData.atoms[selection]

        segList = np.isin(self.psfData.atoms[:,1], sel[:,1])
        resList = np.isin(self.psfData.atoms[:,2], sel[:,2])

        keepList = np.bitwise_and(segList, resList)

        return np.argwhere(keepList)[:,0]





    def getHBDonors(self, keepList):
        """ Identifies all possible hydrogen bond donors in the given inde list, and returns only those that
            correspond to atoms bound to an hydrogen. """

        idxList = np.argwhere(keepList)[:,0]

        for idx in idxList:
            if self.psfData.atoms[idx,3] == "CYH":
                idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "HIS":
                if "HE2" not in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)
                if "HD1" not in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "HSE" and self.psfData.atoms[idx,4] == "ND1":
                idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "HSD" and self.psfData.atoms[idx,4] == "NE2":
                idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "SER":
                if "HG1" not in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "THR":
                if "HG1" not in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "TYR":
                if "HH" not in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)


            keepIdx = np.zeros(self.psfData.atoms.shape[0])
            keepIdx[idxList] = 1

            return keepIdx.astype(bool)




    def getHBAcceptors(self, keepList):
        """ Identifies all possible hydrogen bond acceptors in the given index list, and returns only 
            those that correspond to atoms not bound to an hydrogen. """

        idxList = np.argwhere(keepList)[:,0]

        for idx in idxList:
            if self.psfData.atoms[idx,3] == "CYS":
                idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "HIS":
                if "HE2" in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)
                if "HD1" in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "HSE" and self.psfData.atoms[idx,4] == "NE2":
                idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "HSD" and self.psfData.atoms[idx,4] == "ND1":
                idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "SER":
                if "HG1" in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "THR":
                if "HG1" in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)

            if self.psfData.atoms[idx,3] == "TYR":
                if "HH" in data.psfData.atoms[self.getSameResidueAs(idx)]:
                    idxList = np.delete(idxList, idx)


            keepIdx = np.zeros(self.psfData.atoms.shape[0])
            keepIdx[idxList] = 1

            return keepIdx.astype(bool)



    def getBoundAtoms(self, selection):
        """ Returns the bound atoms for each atom in the given selection. """

        if type(selection) == int:
            selection = [selection] #_To keep it 2 dimensional

        #_Get the indices corresponding to the selection string
        if type(selection) == str:
            selection = self.selection(selection)

        keepIdx = np.zeros(self.psfData.atoms.shape[0])
        keepIdx[selection] = 1

        bonds1  = self.psfData.bonds[:,::2]
        bonds2  = self.psfData.bonds[:,1::2]

        selBonds1   = np.argwhere( np.isin(bonds1, self.psfData.atoms[selection][:,0].astype(int)) )
        selBonds2   = np.argwhere( np.isin(bonds2, self.psfData.atoms[selection][:,0].astype(int)) )

        keepIdx[ bonds2[selBonds1[:,0], selBonds1[:,1]] - 1 ] = 1
        keepIdx[ bonds1[selBonds2[:,0], selBonds2[:,1]] - 1 ] = 1

        return keepIdx.astype(bool)