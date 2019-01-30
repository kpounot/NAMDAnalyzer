import os, sys
import numpy as np

from collections import namedtuple

from .psfReader import PSFReader

class NAMDPSF(PSFReader):
    """ This class is used for .psf file reading. """

    def __init__(self, psfFile=None):

        PSFReader.__init__(self)

        if psfFile:
            self.importPSFFile(psfFile)



#---------------------------------------------
#_Data accession methods
#---------------------------------------------
    def getAtomsMasses(self, selection):
        """ Return the column corresponding to atoms masses as an 1D array of float.

            Input:  selection   -> a selection of atom indices """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.getSelection(selection)

        return self.psfData.atoms[selection,7].astype(float)




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
            keepIdxList.append( np.ones_like(self.psfData.atoms[:,0]).astype(bool) )
        if selText == "protein":
            keepIdxList.append( np.isin(self.psfData.atoms[:,3], self.protSel) )
        if selText == "protH":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.protH) )
        if selText == "backbone":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.backboneSel) )
        if selText == "water":
            keepIdxList.append( np.isin(self.psfData.atoms[:,3], "TIP3") )
        if selText == "waterH":
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], self.waterH) )

        if selText == "protNonExchH":
            keepIdxList = np.array([bool( self.H.match(val) ) for val in self.psfData.atoms[:,4]])
            protList    = np.isin(self.psfData.atoms[:,4], self.protH)
            keepIdxList = np.bitwise_and(keepIdxList, protList)
            for key, value in self.protExchH.items():
                resArray        = np.isin(self.psfData.atoms[:,3], key)
                exchHArray      = np.isin(self.psfData.atoms[:,4], value)
                nonExchHArray   = np.invert(np.bitwise_and(resArray, exchHArray))
                keepIdxList     = np.bitwise_and(keepIdxList, nonExchHArray)

            return np.argwhere(keepIdxList)[:,0]


        #_Parsing the segment list if not None
        if segName:
            segName = np.array(segName).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,1], segName) )

        #_Parsing the not segment list if not None
        if NOTsegName:
            NOTsegName = np.array(NOTsegName).astype(str)
            keepIdxList.append( np.invert(np.isin(self.psfData.atoms[:,1], NOTsegName)) )

        #_Parsing the residue number list if not None
        if resNbr:
            resNbr = np.array(resNbr).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,2], resNbr) )

        #_Parsing the not residue number list if not None
        if NOTresNbr:
            NOTresNbr = np.array(NOTresNbr).astype(str)
            keepIdxList.append( np.invert(np.isin(self.psfData.atoms[:,2], NOTresNbr)) )

        #_Parsing the residue name list if not None
        if resName:
            resName = np.array(resName).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,3], resNbr) )

        #_Parsing the not residue name list if not None
        if NOTresName:
            NOTresName = np.array(NOTresName).astype(str)
            keepIdxList.append( np.invert(np.isin(self.psfData.atoms[:,3], NOTresName)) )

        #_Parsing the name list if not None
        if atom:
            atom = np.array(atom).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,4], atom) )

        #_Parsing the not atom list if not None
        if NOTatom:
            NOTatom = np.array(NOTatom).astype(str)
            keepIdxList.append( np.invert(np.isin(self.psfData.atoms[:,4], NOTatom)) )

        #_Parsing the index list if not None
        if index:
            index   = np.array(index).astype(str)
            keepIdxList.append( np.isin(self.psfData.atoms[:,0], index) )


        #_Using bitwise AND to keep only the indices that are true everywhere
        if len(keepIdxList) > 1:
            for i in range(1, len(keepIdxList)):
                keepIdxList[0] = np.bitwise_and(keepIdxList[0], keepIdxList[i])

        #_Using argwhere to return the indices corresponding to the True values
        return np.argwhere(keepIdxList[0])[:,0]



