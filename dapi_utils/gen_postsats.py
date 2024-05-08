import cv2
from PIL import Image
import numpy as np
import pandas as pd
import pathlib
import sys
Image.MAX_IMAGE_PIXELS = None

# image = Image.open('GCA002ACB_DAPI_masked_deepcell_format_nuc.tif')

# sample_id = sys.argv[1]# 'GCA054TIA'
# x=8016; w=2432; y=192; h=3472; # ROI #1
# x=192; w=7392; y=4624; h=5952; # ROI #2
# x=5552; w=3424; y=8864; h=3492; # ROI #2


# work_dir = 'x%s_w%s_y%s_h%s' % (x,w,y,h)

def gen_deepcell_poststats(sample_id, marker_list,fDeepCellMask,postStatsFname):
#     mask = cv2.imread('GCA054TIA/mask.tif',cv2.IMREAD_ANYDEPTH)
#     work_dir = 'x%s_w%s_y%s_h%s' % (x,w,y,h)
    
    image = Image.open(fDeepCellMask)#'GCA002ACB_DAPI_masked_deepcell_format_nuc.tif')
    mask = np.asarray(image)
#     mask = cv2.imread(fDeepCellMask,cv2.IMREAD_ANYDEPTH)
    all_y = np.where(mask != 0)[0]
    all_x = np.where(mask != 0)[1]

    total_element = len(np.unique(mask))-1
    ind_x = list()
    ind_y = list()
    

    # initialize ind_x, ind_y to save all values' coordinate, and feature_list
    for i in range(0,total_element):
        ind_x.append([])
        ind_y.append([])
    
    # prepare feature list for markers
    feature_list = list()
    c_Nuc_feature = 4
    total_feature = c_Nuc_feature + len(marker_list) * 2
    for i in range(0,total_feature):
        feature_list.append([])

    # iterate all pixel values over whole mask, then add coordinate to relevant ind_x and ind_y
    for i in range(0,len(all_x)):
        cur_y = all_y[i]
        cur_x = all_x[i]
        cur_pixel_val = mask[cur_y,cur_x]
        ind_x[cur_pixel_val-1].append(cur_x)
        ind_y[cur_pixel_val-1].append(cur_y)

    img_marker_list = []
    
    for marker_id in marker_list:
        fMarker = '/home/baos1/GCA/MXIF/pytorch-CycleGAN-and-pix2pix-master_FGH/prepare_datasets/a100_fgh/masked_image/%s/%s_%s_masked.tif' % (sample_id,sample_id, marker_id)
        # print(fMarker)
        tmp_img_marker = cv2.imread(fMarker,cv2.IMREAD_GRAYSCALE)
#         tmp_img_marker = tmp_img_marker[start_pos[0]:start_pos[0]+patch_shape[0],\
#                                         start_pos[1]:start_pos[1]+patch_shape[1]]
        img_marker_list.append(tmp_img_marker)

    # for epithelium percentage, for epithelium percentage
    epi_mask_Marker = cv2.imread('EPI_STR_MASK/%s_EpiMask_masked.tif' % (sample_id),cv2.IMREAD_GRAYSCALE)
    str_mask_Marker = cv2.imread('EPI_STR_MASK/%s_StromaMask_masked.tif' % (sample_id),cv2.IMREAD_GRAYSCALE)


    
    
    for i in range(0,total_element):
#         print(i)
        cell_index = i + 1 # start from index 1
        
        # get centroid coordinate x and y
        cY = round(sum(ind_y[i])/(len(ind_y[i])))
        cX = round(sum(ind_x[i])/(len(ind_x[i])))

#         final_ind_y.append(tmp_arr_y)
#         final_ind_x.append(tmp_arr_x)
        
        feature_list[0].append(cell_index) # cell_index_list.append(cell_index)  
        feature_list[1].append(cX) # Cell_Centroid_X_list.append(cX)
        feature_list[2].append(cY) # Cell_Centroid_Y_list.append(cY)
        feature_list[3].append(len(ind_x[i])) # Cell_Area_list.append(len(rows))

        for img_marker_index in range (0,len(marker_list)):
#             print(img_marker_index)
#             print(marker_id)
            #tmp_marker_intensity_arr = get_intensity_arr(rows,cols,marker_list[img_marker_index], x,w,y,h)

            tmp_marker_intensity_arr = get_intensity_arr(ind_y[i],ind_x[i],img_marker_list[img_marker_index])
                    # again, c_Nuc_feature = 4 is constant 
            tmp_marker_feature_index = 2*img_marker_index+ c_Nuc_feature
                    #Median_Nuc_marker
            feature_list[tmp_marker_feature_index].append(np.median(tmp_marker_intensity_arr))
                    #Mean_Nuc_marker
            feature_list[tmp_marker_feature_index+1].append(np.mean(tmp_marker_intensity_arr))
            
        # get Percent_Epithelium
        tmp_epi_marker_intensity_arr = get_intensity_arr(ind_y[i],ind_x[i],epi_mask_Marker)
        tmp_percentage_epi = np.sum(tmp_epi_marker_intensity_arr)/(255*len(tmp_epi_marker_intensity_arr)) * 100
        #print('percentage epi: %s' % tmp_percentage_epi)
        feature_list.append(tmp_percentage_epi)

        # get Percent_Stroma
        tmp_str_marker_intensity_arr = get_intensity_arr(ind_y[i],ind_x[i],str_mask_Marker)
        tmp_percentage_str = np.sum(tmp_str_marker_intensity_arr)/(255*len(tmp_str_marker_intensity_arr)) * 100
        #print('percentage str: %s' % tmp_percentage_str)
        feature_list.append(tmp_percentage_str)

    
    data = getFeatureFromListToPd(feature_list)

    # Make data frame of above data
    df = pd.DataFrame(data)
    # append data frame to CSV file
    df.to_csv(postStatsFname, mode='a', index=False, header=True)
    print('done')
                 
#         mask[tmp_arr_y,tmp_arr_x] = 10000
def get_intensity_arr(cols,rows,marker_image):
    tmp_intensity_arr = []
    for i in range(0, len(rows)):
        tmp_intensity_arr.append(marker_image[cols[i]][rows[i]])
    return tmp_intensity_arr

def getFeatureFromListToPd(feature_list):
    data = {
            'ID': feature_list[0],#cell_index_list,
            'Cell_Centroid_X': feature_list[1],#Cell_Centroid_X_list,
            'Cell_Centroid_Y': feature_list[2],#Cell_Centroid_Y_list,
            'Cell_Area': feature_list[3],#Cell_Area_list,
            'Median_Nuc_ACTG1': feature_list[4],#Median_Nuc_VIMENTIN_list,
            'Mean_Nuc_ACTG1': feature_list[5],#Mean_Nuc_VIMENTIN_list,
            'Median_Nuc_ACTININ': feature_list[6],#Median_Nuc_VIMENTIN_list,
            'Mean_Nuc_ACTININ': feature_list[7],#Mean_Nuc_VIMENTIN_list,
            'Median_Nuc_BCATENIN': feature_list[8],#Median_Nuc_SMA_list,
            'Mean_Nuc_BCATENIN': feature_list[9],#Mean_Nuc_SMA_list,
            'Median_Nuc_CD11B': feature_list[10],#Median_Nuc_CD3D_list,
            'Mean_Nuc_CD11B': feature_list[11],#Mean_Nuc_CD3D_list,
            'Median_Nuc_CD20': feature_list[12],#Median_Nuc_CD4_list,
            'Mean_Nuc_CD20': feature_list[13],#Mean_Nuc_CD4_list,
            'Median_Nuc_CD3D': feature_list[14],#Median_Nuc_CD8_list,
            'Mean_Nuc_CD3D': feature_list[15],# Mean_Nuc_CD8_list,
            'Median_Nuc_CD45': feature_list[16],# Median_Nuc_CD68_list,
            'Mean_Nuc_CD45': feature_list[17],#Mean_Nuc_CD68_list,
            'Median_Nuc_CD4': feature_list[18],#Median_Nuc_CD11B_list,
            'Mean_Nuc_CD4': feature_list[19],#Mean_Nuc_CD11B_list,
            'Median_Nuc_CD68': feature_list[20],#Median_Nuc_PANCK_list,
            'Mean_Nuc_CD68': feature_list[21],#Mean_Nuc_PANCK_list,
            'Median_Nuc_CD8': feature_list[22],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_CD8': feature_list[23],#Mean_Nuc_NAKATPASE_list,
            'Median_Nuc_CGA': feature_list[24],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_CGA': feature_list[25],
            'Median_Nuc_COLLAGEN': feature_list[26],#Median_Nuc_NAKATPASE_list,
            'Del_Nuc_COLLAGEN': feature_list[27],
            'Median_Nuc_DAPI': feature_list[28],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_DAPI': feature_list[29],
            'Median_Nuc_ERBB2': feature_list[30],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_ERBB2': feature_list[31],
            'Median_Nuc_FOXP3': feature_list[32],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_FOXP3': feature_list[33],
            'Median_Nuc_HLAA': feature_list[34],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_HLAA': feature_list[35],
            'Median_Nuc_LYSOZYME': feature_list[36],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_LYSOZYME': feature_list[37],
            'Median_Nuc_MUC2': feature_list[38],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_MUC2': feature_list[39],
            'Median_Nuc_NAKATPASE': feature_list[40],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_NAKATPASE': feature_list[41],
            'Median_Nuc_OLFM4': feature_list[42],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_OLFM4': feature_list[43],
            'Median_Nuc_PANCK': feature_list[44],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_PANCK': feature_list[45],
            'Median_Nuc_PCNA': feature_list[46],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_PCNA': feature_list[47],
            'Median_Nuc_PEGFR': feature_list[48],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_PEGFR': feature_list[49],
            'Median_Nuc_PSTAT3': feature_list[50],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_PSTAT3': feature_list[51],
            'Median_Nuc_SMA': feature_list[52],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_SMA': feature_list[53],
            'Median_Nuc_SOX9': feature_list[54],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_SOX9': feature_list[55],
            'Median_Nuc_VIMENTIN': feature_list[56],#Median_Nuc_NAKATPASE_list,
            'Mean_Nuc_VIMENTIN': feature_list[57],
            'Percent_Epithelium': feature_list[58],
            'Percent_Stroma':feature_list[59]
        }
    return data

#fDeepCellMask = 'deepcell/GCA002ACB_DAPI_masked_deepcell_format_nuc.tif' # % (sample_id,work_dir,sample_id,x,w,y,h)
marker_list = ['ACTG1','ACTININ','BCATENIN','CD11B','CD20','CD3D','CD45','CD4','CD68','CD8','CGA','COLLAGEN','DAPI','ERBB2','FOXP3','HLAA','LYSOZYME','MUC2','NAKATPASE','OLFM4','PANCK','PCNA','PEGFR','PSTAT3','SMA','SOX9','VIMENTIN']


import sys

sample_id = sys.argv[1]
print(sample_id)

#for sample_id in sample_list:
fDeepCellMask = 'NUC/%s_DAPI_MUC2_MASKED_NUC.tif' % sample_id

postStatsFname = 'POSTSTATS/with_epi_str_percentage/%s_DAPI_MUC2_MASKED_NUC_deepcell_format_PosStats_with_epi_str_percentage.csv' % sample_id
gen_deepcell_poststats(sample_id, marker_list,fDeepCellMask,postStatsFname)
