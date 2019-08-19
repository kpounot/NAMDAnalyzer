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
        """ Parse the selection string 'selText' by identifying keywords. """

        if type(self.selT) != str:
            print('Selection text should be a string instance, the given argument cannot be parsed.')
            return

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


        if re.search('within', self.selT):
            self.withinList = self._getWithin(self.selT[self.selT.find('within'):]) 

        if self.withinList is not None:
            if self.withinList.ndim > 1:

                self.selection = []
                selT = self.selT

                for fId in range(self.withinList.shape[1]):
                    withinList = np.argwhere(self.withinList[:,fId])[:,0]
                    self.selection.append( self._parseText(self.selT, withinList) )
                    self.selT = selT
                    

            else:
                self.selection  = self._parseText(self.selT, self.withinList)

        else:
            self.selection = self._parseText(self.selT)





    def _parseText(self, frameSel, withinList=None):

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
            Processes it and returns the rest of the selection. """
    
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



        for selArray in sel:
            sel[0] = np.intersect1d(sel[0], selArray)

        if withinList is not None:
            sel[0] = np.intersect1d(sel[0], withinList)


        self.selT = self.selT[:self.selT.find('same resid as ')]


        return self.dataContext.getSameResidueAs(sel[0])




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
                sel = np.intersect1d(sel[0], self._getSelection(partialSel))

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




    def _getBoundTo(self, partialSel, withinList=None):
        """ Used specifically to parse within keyword selection. 
            The 'within' part is processed and the rest of the selection text is returned. """

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


        for selArray in sel:
            sel[0] = np.intersect1d(sel[0], selArray)


        if withinList is not None:
            sel[0] = np.intersect1d(sel[0], withinList)

        self.selT = self.selT[:self.selT.find('bound to ')]


        return np.argwhere( self.dataContext.getBoundAtoms(sel[0]) )[:,0]


