import sys, os

from package.NAMDAnalyzer import NAMDAnalyzer

fileList = sys.argv[1:]
data = NAMDAnalyzer(fileList)
