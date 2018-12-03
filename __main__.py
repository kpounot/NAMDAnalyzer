import sys, os

from package.NAMDAnalyzer import NAMDAnalyzer
from package.Dataset import Dataset

fileList = sys.argv[1:]
data = NAMDAnalyzer(fileList)
