import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

data_text_file='/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/data_tissues.txt'
Image.MAX_IMAGE_PIXELS = None
data_dir='/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF'

with open(data_text_file, 'r') as f:
    lines = f.readlines()

    for line in lines:
        file_path = os.path.join(data_dir, line.strip())

        img = Image.open(file_path)

        plt.imshow(img, cmap='gray')

        filename = os.path.basename(file_path)

        plt.title(filename)

        plt.show()