import sys, os

import argparse

from NAMDAnalyzer.Dataset import Dataset
from NAMDAnalyzer.selection.selParser import SelParser
from NAMDAnalyzer.dataAnalysis.BackscatteringDataConvert import BackScatData
from NAMDAnalyzer.dataAnalysis.ScatDiffusion import ScatDiffusion
from NAMDAnalyzer.dataAnalysis.HydrogenBonds import HydrogenBonds
from NAMDAnalyzer.dataAnalysis.Rotations import Rotations
from NAMDAnalyzer.dataAnalysis.ResidenceTime import ResidenceTime
from NAMDAnalyzer.dataAnalysis.RadialDensity import ( ResidueWiseWaterDensity, 
                                                      COMRadialNumberDensity, 
                                                      RadialNumberDensity )



parser = argparse.ArgumentParser()
parser.add_argument("fileList", nargs="*", help="List of data files to be loaded on startup.")


args = parser.parse_args()

data = Dataset(*args.fileList)

