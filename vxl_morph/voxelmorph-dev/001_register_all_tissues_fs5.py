from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from my_utils import Utils
import glob
import re
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from my_utils import Utils
import torch
import csv
import pandas as pd
import seaborn as sns
import fnmatch
import matplotlib.pyplot as plt

mask_dir='/nfs2/baos1/rudravg/Retention_Masks'
tissue_path='/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set01/GCA002ACB/Registered'
tissue_name='GCA002ACB'

for file in os.listdir(tissue_path):
    if fnmatch.fnmatch(file, '*{}*'.format(tissue_name)):
        print("File found: ", file)