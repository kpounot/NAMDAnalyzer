"""

Classes
^^^^^^^

"""

import os, sys
import numpy as np
import re


class SelParser:
    """ This class is used to parse a simple selection text, calls the appropriate functions from 
        :class:`.NAMDPSF` and :class:`.NAMDDCD`.
    
        A dataContext is provided to the class, that is simply the :class:`.Dataset` class, in which psf
        and dcd files are loaded. 

        Atom indices are stored for each frame in *selection* attribute.

        Examples:

        To select hydrogens bound to water oxygens within 3 angstroms of a protein region 
        for a given frame range, use from class :class:`.Dataset` instance d: 

        .. code-block:: python

            d.selection('name H1 H2 and bound to name OH2 and within 3 of protein' 
                         + ' and resid 40:80 frame 0:100:2')

    """


    def __init__(self, dataContext, selT=None):

        self.selection = None

        self.dataContext = dataContext
        self.init_selT   = selT
        self.selT        = selT
        self.frame       = 0
        self.withinList  = None

        self.and_selList = [] #_Selections that will be compared using 'and' operator


        self.keywords   = [ 'protein', 'water', 'backbone', 'waterH', 'hbhydrogens', 'hydrogen',
                            'protNonExchH', 'protH', 'proteinH', 'all', 'hbdonors', 'hbacceptors']


        self.selTxtDict =   {
                                '^index'         : 'index', 
                                '^atom'          : 'atom',
                                '^name'          : 'atom',
                                '^resID'         : 'resID',
                                '^resName'       : 'resName',
                                '^segName'       : 'segName',
                                '^segID'         : 'segName'
                            }


        self.selKwdDict =  {
                                'index'         : [], 
                                'atom'          : [],
                                'resID'         : [],
                                'resName'       : [],
                                'segName'       : []
                            }


        if selT is not None:
            self.process()




    def process(self):
        """ Parse the selection string 'selText' by identifying keywords. 

        """

        if type(self.selT) != str:
            print('Selection text should be a string instance, the given argument cannot be parsed.')
            return


        #_Parses frame keyword
        if re.search('frame', self.selT):
            frameTxt = self.selT[self.selT.find('frame')+6:]

            if re.match('[0-9]+:[0-9]+$', frameTxt):
                frameTxt    = np.array(frameTxt.split(':')).astype(int) 
                self.frame  = slice(*frameTxt)
            elif re.match('[0-9]+:[0-9]+:[0-9]+$', frameTxt):
                frameTxt    = np.array(frameTxt.split(':')).astype(int) 
                self.frame  = slice(*frameTxt)
            else:
                self.frame = np.array( frameTxt.split() ).astype(int)


            self.selT = re.sub(' frame [0-9]+(:[0-9]+)*(\s[0-9]+)*', '', self.selT)


        else:
            self.frame = 1



        #_Parses within keyword
        if re.search('within', self.selT):
            self.withinList = self._getWithin(self.selT) 

        if self.withinList is not None:
            if self.withinList.ndim > 1:

                self.selection = []
                selT = self.selT

                for fId in range(self.withinList.shape[1]):
                    withinList = np.argwhere(self.withinList[:,fId])[:,0]
                    self.selection.append( self._parseText(withinList) )
                    self.selT = selT
                    

            else:
                self.selection  = self._parseText(self.withinList)

        else:
            self.selection = self._parseText()





    def _parseText(self, withinList=None):

        sel = []

        if re.search('same resid as', self.selT):
            sel.append( self._getSameResid(self.selT[self.selT.find('same resid as'):], withinList) )
            withinList = None

        if re.search('bound to', self.selT):
            sel.append( self._getBoundTo(self.selT[self.selT.find('bound to'):], withinList) )
            withinList = None

        if self.selT != '':
            sel.append( self._getSelection(self.selT) )


        if withinList is not None:
            sel.append( withinList )
 

        for selArray in sel:
            sel[0] = np.intersect1d(sel[0], selArray)


        return sel[0]





    def _getSelection(self, partialText):
        """ This method is used by parseText to generate the different independent selections from
            selection text before these are compared using 'and' operators to generate the 
            final selection. 

        """


        selList = []


        partialText = partialText.strip().split('and')

        for selCmd in partialText:
            #_Reinitialize selection dict
            for key, val in self.selKwdDict.items():
                self.selKwdDict[key] = []

            invert=False

            if selCmd != '':
                selCmd = re.sub('[\(\)\{\}]', '', selCmd).strip() #_Performs some cleaning

                if 'not' in selCmd:
                    invert=True
                    selCmd = re.sub('not ', '', selCmd)

                if np.isin(selCmd, self.keywords):
                    selList.append(self.dataContext.getSelection(selCmd, invert=invert))
                    continue

                for key in self.selTxtDict.keys():
                    matchRes = re.match(key, selCmd, re.I)
                    if matchRes:
                        sel = selCmd[matchRes.end():].strip()
                        
                        if re.search('[0-9]+:[0-9]+', sel):
                            sel = range(int(sel.split(':')[0]), int(sel.split(':')[1]) + 1)
                        else:
                            sel = sel.split()

                        self.selKwdDict[self.selTxtDict[key]] = sel

                selList.append( self.dataContext.getSelection(**self.selKwdDict, invert=invert) )

        for sel in selList:
            selList[0] = np.intersect1d(selList[0], sel)

        return selList[0]

            


    def _getSameResid(self, partialSel, withinList=None):
        """ Used to parse 'same resid as' selection text.
            Everything after 'same resid as' will be considered.
            Processes it and returns the rest of the selection. 

        """
    
        sel = []

        partialSel = re.sub('same resid as ', '', partialSel).strip()

        
        if re.search('bound to', partialSel):
            sel.append( self._getBoundTo(partialSel[partialSel.find('bound to'):], self.withinList) )
            partialSel  = partialSel[:partialSel.find('bound to')] 

            if withinList is not None:
                withinList = None

            if partialSel != '':
                sel.append( np.intersect1d(sel[0], self._getSelection(partialSel)) )

        else:
            sel.append( self._getSelection(partialSel) )



        if sel != []:
            for selArray in sel:
                sel[0] = np.intersect1d(sel[0], selArray)

            if withinList is not None:
                sel[0] = np.intersect1d(sel[0], withinList)

        elif sel == [] and withinList is not None:
            sel.append( withinList )

        else:
            return np.array([])


        self.selT = self.selT[:self.selT.find('same resid as ')]


        return self.dataContext.getSameResidueAs(sel[0])




    def _getWithin(self, partialSel):
        """ Used specifically to parse within keyword selection. 
            The 'within' part is processed and the rest of the selection text is returned. 

        """

        outSel = self.selT[:self.selT.find('within')]

        partialSel = partialSel[partialSel.find('within'):]
        partialSel = re.sub('within ', '', partialSel).strip()

        distance    = float(partialSel.split(' of ')[0])
        partialSel  = partialSel.split(' of ')[1]

        if re.search('same resid as', partialSel):
            sel         = self._getSameResid(partialSel[partialSel.find('same resid as'):])
            partialSel  = partialSel[:partialSel.find('same resid as')] 

            if partialSel != '':
                sel = np.intersect1d(sel[0], self._getSelection(partialSel))

        elif re.search('bound to', partialSel):
            sel         = self._getBoundTo(partialSel[partialSel.find('bound to'):])
            partialSel  = partialSel[:partialSel.find('bound to')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))


        else:
            sel = self._getSelection(partialSel)


        #_Process outSel 
        if outSel != '':
            outSel = list(filter(None, outSel.strip().split('and')))

            if outSel[-1] == '':
                sel = self.dataContext.getWithin(distance, sel, frame=self.frame)

            elif re.search('same resid as', outSel[-1]):
                if outSel[-1].split(' as ')[-1] != '':
                    sel = self.dataContext.getWithin(distance, sel, outSel[-1].split('as')[-1], 
                                                     frame=self.frame)
                else:
                    sel = self.dataContext.getWithin(distance, sel, frame=self.frame)

            elif re.search('bound to', outSel[-1]):
                if outSel[-1].split(' to ')[-1] != '':
                    sel = self.dataContext.getWithin(distance, sel, outSel[-1].split('to')[-1], 
                                                     frame=self.frame)
                else:
                    sel = self.dataContext.getWithin(distance, sel, frame=self.frame)

            else:
                sel = self.dataContext.getWithin(distance, sel, outSel[-1], frame=self.frame)

        else:
            sel = self.dataContext.getWithin(distance, sel, frame=self.frame)




        self.selT = self.selT[:self.selT.find('within ')]


        return sel




    def _getBoundTo(self, partialSel, withinList=None):
        """ Used specifically to parse within keyword selection. 
            The 'within' part is processed and the rest of the selection text is returned. 

        """

        sel = []

        partialSel = partialSel[partialSel.find('bound to'):]
        partialSel = re.sub('bound to ', '', partialSel).strip()

        if re.search('same resid as', partialSel):
            sel.append( self._getSameResid(partialSel[partialSel.find('same resid as'):], withinList) )
            partialSel  = partialSel[:partialSel.find('same resid as')] 

            if withinList is not None:
                withinList = None

            if partialSel != '':
                sel.append( np.intersect1d(sel[0], self._getSelection(partialSel)) )

        else:
            sel.append(self._getSelection(partialSel))


        if sel != []:
            for selArray in sel:
                sel[0] = np.intersect1d(sel[0], selArray)


            if withinList is not None:
                sel[0] = np.intersect1d(sel[0], withinList)

        elif sel == [] and withinList is not None:
            sel.append( withinList )

        else:
            return np.array([])

        self.selT = self.selT[:self.selT.find('bound to ')]


        return np.argwhere( self.dataContext.getBoundAtoms(sel[0]) )[:,0]


