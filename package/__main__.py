import sys, os

import argparse

from NAMDAnalyzer.Dataset import Dataset
from NAMDAnalyzer.selection.selParser import SelParser
from NAMDAnalyzer.dataAnalysis.IncoherentScat import IncoherentScat
from NAMDAnalyzer.dataAnalysis.ScatDiffusion import ScatDiffusion
from NAMDAnalyzer.dataAnalysis.HydrogenBonds import HydrogenBonds
from NAMDAnalyzer.dataAnalysis.Rotations import Rotations, WaterAtProtSurface
from NAMDAnalyzer.dataAnalysis.ResidenceTime import ResidenceTime
from NAMDAnalyzer.dataAnalysis.VolMapDensity import WaterVolNumberDensity
from NAMDAnalyzer.dataAnalysis.RadialDensity import ( ResidueWiseWaterDensity, 
                                                      RadialNumberDensity )



parser = argparse.ArgumentParser()
parser.add_argument("fileList", nargs="*", help="List of data files to be loaded on startup.")


args = parser.parse_args()

data = Dataset(*args.fileList)

