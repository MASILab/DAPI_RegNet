from PIL import Image
import numpy as np
import os
import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None


mask = "/nfs2/baos1/rudravg/Retention_Masks/GCA069ACB_TISSUE01_TISSUE_RETENTION.tif"
default_img = "/nfs2/baos1/rudravg/GCA069ACB_TISSUE01_DAPI_DAPI_12ms_ROUND_00_initial_reg.tif"
metric_nmi = []
metric_ncc=[]
metric_sm=[]
for i in tqdm(range(18),desc="Calculating NMI"):
    round_number = str(i).zfill(2)
    new_img="/nfs2/baos1/rudravg/GCA069ACB_TISSUE01_DAPI_DAPI_12ms_ROUND_" + round_number + "_initial_reg.tif"
    metric_obj=metrics.Metrics(default_img,new_img,mask)
    nmi=metric_obj.calculate_normalized_mutual_information()
    #cc=metric_obj.calculate_normalized_cross_correlation()
    
    metric_nmi.append(nmi)
    #metric_ncc.append(cc)
    smi=metric_obj.calculate_ssim()
    metric_sm.append(smi)
#print(metric_ncc)
#print(metric_nmi/max(metric_nmi))
#print(metric_nmi)
print(metric_sm)
x = range(len(metric_sm))


plt.figure(figsize=(10, 6))
plt.plot(x, metric_sm, marker='o', label='SSIM')  # 'o' marker to show each point
plt.plot(x, metric_nmi, marker='o', label='NMI')  # 'o' marker to show each point

# Adding titles and labels
plt.title('SSIM and NMI Values for each Round')
plt.xlabel('Round Number')
plt.ylabel('Metric Value')

# Adding y-values on each point for SSIM
for i, txt in enumerate(metric_sm):
    plt.annotate(f"{txt:.2f}", (x[i], metric_sm[i]))

# Adding y-values on each point for NMI
for i, txt in enumerate(metric_nmi):
    plt.annotate(f"{txt:.2f}", (x[i], metric_nmi[i]))

plt.grid(True)
plt.xticks(x)  # Ensure x-axis ticks correspond to whole numbers
plt.legend()  # Show legend
plt.show()

