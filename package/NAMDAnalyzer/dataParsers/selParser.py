import os, sys
import numpy as np
import re


class SelParser:
    """ This class is used to parse a simple selection text, calls the appropriate functions from psdParser
        and dcdParser, and eventually, returns a list of atom indices.
        
        A dataContext is provided to the class, that is simply the Dataset class, in which psf
        and dcd files are loaded. """


    def __init__(self, dataContext, selT=None, frame=-1):

        self.selection = None

        self.dataContext = dataContext
        self.init_selT   = selT
        self.selT        = selT
        self.frame       = frame

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
            self.parseText()




    def parseText(self):
        """ Parse the selection string 'selText' by identifying keywords. """

        if type(self.selT) != str:
            print('Selection text should be a string instance, the given argument cannot be parsed.')
            return

        if re.search('frame', self.selT):
            self.frame = int(self.selT[self.selT.find('frame'):].split()[1])
            self.selT = re.sub(' frame [0-9]+', '', self.selT)

        if re.search('same resid as', self.selT):
            self.and_selList.append( self._getSameResid(self.selT[self.selT.find('same resid as'):]) )

        if re.search('within', self.selT):
            self.and_selList.append( self._getWithin(self.selT[self.selT.find('within'):]) )

        if re.search('bound to', self.selT):
            self.and_selList.append( self._getBoundTo(self.selT[self.selT.find('bound to'):]) )


        if self.selT != '':
            self.and_selList.append( self._getSelection(self.selT) )


        for sel in self.and_selList:
            self.and_selList[0] = np.intersect1d(self.and_selList[0], sel)


        self.selection = self.and_selList[0]
        
        self.and_selList = []

        


    def _getSelection(self, partialText):
        """ This method is used by parseText to generate the different independent selections from
            selection text before these are compared using 'and'/'or' operators to generate the 
            final selection. """


        selList = []

        partialText = partialText.strip().split('and')

        for selCmd in partialText:
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
                        sel = selCmd[matchRes.end():]
                        
                        if re.search('[0-9]+:[0-9]+', sel):
                            sel = range(int(sel.split(':')[0]), int(sel.split(':')[1]))
                        else:
                            sel = sel.split()

                        self.selKwdDict[self.selTxtDict[key]] = sel

                selList.append( self.dataContext.getSelection(**self.selKwdDict, invert=invert) )


        for sel in selList:
            selList[0] = np.intersect1d(selList[0], sel)

        return selList[0]

            


    def _getSameResid(self, partialSel):
        """ Used to parse 'same resid as' selection text.
            Everything after 'same resid as' will be considered.
            Processes it and returns the rest of the selection. """
    
        partialSel = re.sub('same resid as ', '', partialSel).strip()

        
        if re.search('within', partialSel):
            sel         = self._getWithin(partialSel)
            partialSel  = partialSel[:partialSel.find('within')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))


        elif re.search('bound to', partialSel):
            sel         = self._getBoundTo(partialSel[partialSel.find('bound to'):])
            partialSel  = partialSel[:partialSel.find('bound to')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))

        else:
            sel = self._getSelection(partialSel)


        sel = self.dataContext.getSameResidueAs(sel)


        self.selT = self.selT[:self.selT.find('same resid as ')]


        return sel




    def _getWithin(self, partialSel):
        """ Used specifically to parse within keyword selection. 
            The 'within' part is processed and the rest of the selection text is returned. """

        partialSel = partialSel[partialSel.find('within'):]
        partialSel = re.sub('within ', '', partialSel).strip()

        distance    = float(partialSel.split(' of ')[0])
        partialSel  = partialSel.split(' of ')[1]

        if re.search('same resid as', partialSel):
            sel         = self._getSameResid(partialSel[partialSel.find('same resid as'):])
            partialSel  = partialSel[:partialSel.find('same resid as')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))

        elif re.search('bound to', partialSel):
            sel         = self._getBoundTo(partialSel[partialSel.find('bound to'):])
            partialSel  = partialSel[:partialSel.find('bound to')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))


        else:
            sel = self._getSelection(partialSel)


        sel = self.dataContext.getWithin(distance, sel, frame=self.frame)


        self.selT = self.selT[:self.selT.find('within ')]


        return sel




    def _getBoundTo(self, partialSel):
        """ Used specifically to parse within keyword selection. 
            The 'within' part is processed and the rest of the selection text is returned. """

        partialSel = partialSel[partialSel.find('bound to'):]
        partialSel = re.sub('bound to ', '', partialSel).strip()

        if re.search('same resid as', partialSel):
            sel         = self._getSameResid(partialSel[partialSel.find('same resid as'):])
            partialSel  = partialSel[:partialSel.find('same resid as')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))

        elif re.search('within', partialSel):
            sel         = self._getWithin(partialSel)
            partialSel  = partialSel[:partialSel.find('within')] 

            if partialSel != '':
                sel = np.intersect1d(sel, self._getSelection(partialSel))

        else:
            sel = self._getSelection(partialSel)


        sel = np.argwhere(self.dataContext.getBoundAtoms(sel))[:,0]


        self.selT = self.selT[:self.selT.find('bound to ')]


        return sel


