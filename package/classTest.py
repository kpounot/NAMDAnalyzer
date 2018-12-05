import sys, os

import numpy as np

class classA:
    def __init__(self, ATag):
        print("Initializing class A with tag: " + ATag)

        self.atag = ATag

    def printTag(self):
        print("Tag is: " + self.tag)


class classB:
    def __init__(self, BTag):
        print("Initializing class B with tag: " + BTag)

        self.btag = BTag

    def printTag(self):
        print("Tag is: " + self.tag)



class Main(classA, classB):

    def __init__(self, mainTag, ATag="A", BTag="B"):
        classA.__init__(self, ATag)
        classB.__init__(self, BTag)

        self.mtag = mainTag

    def printTag(self):
        print("Tag is: " + self.atag)




