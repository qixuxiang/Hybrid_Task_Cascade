# from https://github.com/joheras/CLoDSA
# pip install clodsa
from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique


from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

PROBLEM = "instance_segmentation"
ANNOTATION_MODE = "coco"
INPUT_PATH = "jinnan2_round2_train_20190401/restricted"
GENE_MODE = "linear"
OUTPUT_MODE = "coco"
OUTPUT_PATH= "output/"

augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENE_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH})
transformer = transformerGenerator(PROBLEM)

for angle in [90,180]:
    rotate = createTechnique("rotate", {"angle" : angle})
    augmentor.addTransformer(transformer)

flip = createTechnique("flip",{"flip":1})
augmentor.addTransformer(transformer(flip))

augmentor.applyAugmentation()

