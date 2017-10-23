#!/usr/bin/python

from __future__ import print_function
import os, sys
import platform
import fnmatch

try:
    from itertools import izip
except ImportError:
    izip = zip

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from csHelpers import *

CSUPPORT = True
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False

def getPrediction(args, groundTruthFile):
    if not args.predictionPath:
        rootPath = None
        if 'CITYSCAPES_RESULTS' in os.environ:
            rootPath = os.environ['CITYSCAPES_RESULTS']
        elif 'CITYSCAPES_DATASET' in os.environ:
            rootPath = os.path.join(os.environ['CITYSCAPES_DATASET'], "result")
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root foler. Please read the instructions of this method.")

        args.predictionPath = rootPath 
