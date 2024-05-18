import numpy as np
import tifffile as tiff
import cv2

#Define whole sets metadata load from excel.
metadata_cy3=[]
metadata_cy5=[]
metadata_cy2=[]
metadata_cy3.append({'ROUND_ID': '00','Marker':'SNA','ms':50})
metadata_cy3.append({'ROUND_ID': '01','Marker':'Background','ms':50})
metadata_cy3.append({'ROUND_ID': '02','Marker':'ACTG1','ms':100})
metadata_cy3.append({'ROUND_ID': '03','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '04','Marker':'Collagen','ms':100})
metadata_cy3.append({'ROUND_ID': '05','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '06','Marker':'BCATENIN','ms':50})
metadata_cy3.append({'ROUND_ID': '07','Marker':'Background','ms':50})
metadata_cy3.append({'ROUND_ID': '08','Marker':'CgA','ms':150})
metadata_cy3.append({'ROUND_ID': '09','Marker':'Background','ms':150})
metadata_cy3.append({'ROUND_ID': '10','Marker':'CD3d','ms':150})
metadata_cy3.append({'ROUND_ID': '11','Marker':'Background','ms':150})
metadata_cy3.append({'ROUND_ID': '12','Marker':'OLFM4','ms':50})
metadata_cy3.append({'ROUND_ID': '13','Marker':'Background','ms':50})
metadata_cy3.append({'ROUND_ID': '14','Marker':'CD68','ms':100})
metadata_cy3.append({'ROUND_ID': '15','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '16','Marker':'Sox9','ms':150})
metadata_cy3.append({'ROUND_ID': '17','Marker':'Background','ms':150})
metadata_cy3.append({'ROUND_ID': '18','Marker':'SMA','ms':12})
metadata_cy5.append({'ROUND_ID': '00','Marker':'WGA','ms':75})
metadata_cy5.append({'ROUND_ID': '01','Marker':'Background','ms':75})
metadata_cy5.append({'ROUND_ID': '02','Marker':'CD45','ms':1000})
metadata_cy5.append({'ROUND_ID': '03','Marker':'Background','ms':1000})
metadata_cy5.append({'ROUND_ID': '04','Marker':'CD20','ms':500})
metadata_cy5.append({'ROUND_ID': '05','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '06','Marker':'pSTAT3','ms':500})
metadata_cy5.append({'ROUND_ID': '07','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '08','Marker':'CD4','ms':500})
metadata_cy5.append({'ROUND_ID': '09','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '10','Marker':'HLAA','ms':200})
metadata_cy5.append({'ROUND_ID': '11','Marker':'Background','ms':200})
metadata_cy5.append({'ROUND_ID': '12','Marker':'CD8','ms':500})
metadata_cy5.append({'ROUND_ID': '13','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '14','Marker':'NaKATPase','ms':100})
metadata_cy5.append({'ROUND_ID': '15','Marker':'Background','ms':100})
metadata_cy5.append({'ROUND_ID': '16','Marker':'FOXP3','ms':500})
metadata_cy5.append({'ROUND_ID': '17','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '18','Marker':'ERBB2','ms':1000})

metadata_cy2.append({'ROUND_ID': '00','Marker':'AF','ms':75})
metadata_cy2.append({'ROUND_ID': '01','Marker':'Background','ms':75})
metadata_cy2.append({'ROUND_ID': '02','Marker':'Muc2','ms':75}) #
metadata_cy2.append({'ROUND_ID': '03','Marker':'Background','ms':75})
metadata_cy2.append({'ROUND_ID': '04','Marker':'CD11B','ms':500}) #
metadata_cy2.append({'ROUND_ID': '05','Marker':'Background','ms':500})
metadata_cy2.append({'ROUND_ID': '06','Marker':'PCNA','ms':175}) #
metadata_cy2.append({'ROUND_ID': '07','Marker':'Background','ms':175})
metadata_cy2.append({'ROUND_ID': '08','Marker':'pEGFR','ms':175}) #
metadata_cy2.append({'ROUND_ID': '09','Marker':'Background','ms':175})
metadata_cy2.append({'ROUND_ID': '10','Marker':'Cox2','ms':500}) #
metadata_cy2.append({'ROUND_ID': '11','Marker':'Background','ms':500})
metadata_cy2.append({'ROUND_ID': '12','Marker':'PanCK','ms':100}) #
metadata_cy2.append({'ROUND_ID': '13','Marker':'Background','ms':100})
metadata_cy2.append({'ROUND_ID': '14','Marker':'ACTININ','ms':350}) #
metadata_cy2.append({'ROUND_ID': '15','Marker':'Background','ms':350})
metadata_cy2.append({'ROUND_ID': '16','Marker':'Vimentin','ms':75}) #
metadata_cy2.append({'ROUND_ID': '17','Marker':'Background','ms':75})
metadata_cy2.append({'ROUND_ID': '18','Marker':'Lysozyme','ms':200}) #

#sample_id_list = ['GCA020ACB_TISSUE01','GCA020ACB_TISSUE02','GCA020ACB_TISSUE03','GCA020TIB_TISSUE01','GCA020TIB_TISSUE02','GCA020TIB_TISSUE03','GCA022ACB_TISSUE01','GCA022ACB_TISSUE02','GCA022ACB_TISSUE03','GCA022TIB_TISSUE01','GCA022TIB_TISSUE02','GCA033ACB_TISSUE01','GCA033ACB_TISSUE02','GCA033TIB_TISSUE01','GCA033TIB_TISSUE02','GCA033TIB_TISSUE03','GCA035ACB_TISSUE01','GCA035ACB_TISSUE02','GCA035ACB_TISSUE03','GCA035TIB_TISSUE01','GCA035TIB_TISSUE02','GCA035TIB_TISSUE03','GCA039ACB_TISSUE01','GCA039ACB_TISSUE02','GCA039TIB_TISSUE01','GCA039TIB_TISSUE02','GCA045ACB','GCA045TIB_TISSUE01','GCA045TIB_TISSUE02','GCA059ACB_TISSUE01','GCA059ACB_TISSUE02','GCA059ACB_TISSUE03','GCA059TIB_TISSUE01','GCA059TIB_TISSUE02','GCA059TIB_TISSUE03']
sample_id_list = ['GCA020TIB_TISSUE01']

def histogram_normalization(img_bg, img_marker):
    hist1, _ = np.histogram(img_bg.flatten(), 256, [0, 256])
    hist2, _ = np.histogram(img_marker.flatten(), 256, [0, 256])

    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()

   # Compute the reference histogram (example: average histogram)
    reference_hist = (hist1 + hist2) / 2
    reference_cdf = reference_hist.cumsum()

    mapping_func1 = np.zeros(256, dtype=np.uint8)
    mapping_func2 = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diff1 = np.abs(cdf1[i] - reference_cdf)
        diff2 = np.abs(cdf2[i] - reference_cdf)
        mapping_func1[i] = np.argmin(diff1)
        mapping_func2[i] = np.argmin(diff2)

    result_img_bg = cv2.LUT(img_bg, mapping_func1)
    result_img_marker = cv2.LUT(img_marker, mapping_func2)

    af_removed_image = cv2.subtract(result_img_marker, result_img_bg)

    # Set negative values to 0
    af_removed_image = cv2.max(af_removed_image, 0)

    return af_removed_image

def histogram_normalization_my_version(img_bg, img_marker):
    # Create masks for non-zero pixels
    mask_bg = img_bg > 0
    mask_marker = img_marker > 0

    # Compute histograms only for non-zero pixels
    hist1, _ = np.histogram(img_bg[mask_bg].flatten(), 255, [0, 255])
    hist2, _ = np.histogram(img_marker[mask_marker].flatten(), 255, [0, 255])

    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()

    # Compute the reference histogram (example: average histogram)
    reference_hist = (hist1 + hist2) / 2
    reference_cdf = reference_hist.cumsum()

    mapping_func1 = np.zeros(256, dtype=np.uint8)
    mapping_func2 = np.zeros(256, dtype=np.uint8)

    for i in range(255):
        diff1 = np.abs(cdf1[i] - reference_cdf)
        diff2 = np.abs(cdf2[i] - reference_cdf)
        mapping_func1[i] = np.argmin(diff1)
        mapping_func2[i] = np.argmin(diff2)

    result_img_bg = cv2.LUT(img_bg, mapping_func1)
    result_img_marker = cv2.LUT(img_marker, mapping_func2)

    af_removed_image = cv2.subtract(result_img_marker, result_img_bg)

    # Set negative values to 0
    af_removed_image = cv2.max(af_removed_image, 0)

    return af_removed_image


import os
marker_round = []
for i in range(0,19):
    if i < 10:
        tmp_round_id = '0%s' % i
    else:
        tmp_round_id = i
    marker_round.append(tmp_round_id)

for sample_id in sample_id_list:
    print(sample_id)
    #if 'TISSUE' in sample_id:
    #    sample_dir = sample_id.split('_TISSUE')[0]
    #else:
    sample_dir = sample_id

    marker_path = f'/fs5/p_masi/rudravg/MxIF_Vxm_Registered/{sample_dir}_v1/Unregistered'
    print(marker_path)
    marker_tmp_result_path = f'/fs5/p_masi/rudravg/MxIF_Vxm_Registered/{sample_dir}_v1/Unregistered/AF_Removed'
    os.mkdir(marker_tmp_result_path)

    
    for cur_round_id in range(0,19,2): # jump step wise by 2. in total there are 18 rounds. 0,2,4,6,8,10,12,14,16,18
        cur_round_name = marker_round[cur_round_id]

        cur_cy2_exposure = metadata_cy2[cur_round_id]['ms']
        cur_cy2_marker = metadata_cy2[cur_round_id]['Marker']
        cur_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy2_marker}_CY2_{cur_cy2_exposure}ms_ROUND_{cur_round_name}.tif')
        print(f'{marker_path}/{sample_id}_{cur_cy2_marker}_CY2_{cur_cy2_exposure}ms_ROUND_{cur_round_name}.tif')

        cur_cy3_exposure = metadata_cy3[cur_round_id]['ms']
        cur_cy3_marker = metadata_cy3[cur_round_id]['Marker']
        cur_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')
        print(f'{marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')

        cur_cy5_exposure = metadata_cy5[cur_round_id]['ms']
        cur_cy5_marker = metadata_cy5[cur_round_id]['Marker']
        cur_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy5_marker}_CY5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')
        print(f'{marker_path}/{sample_id}_{cur_cy5_marker}_CY5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')

        if cur_round_id == 0:
            #in set02, we don't need the marker in round 0, just skip. 
            # no background image available, only deal with af removal
            cur_gfp_image_normalized_corrected = cur_cy2_image
            cur_cy3_image_normalized_corrected = cur_cy3_image
            cur_cy5_image_normalized_corrected = cur_cy5_image

            
        else:
            # get the background image in the previous round
            bg_round_id = cur_round_id - 1 

            bg_round_name = marker_round[bg_round_id]

            bg_cy2_exposure = metadata_cy2[bg_round_id]['ms']
            bg_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY2_{bg_cy2_exposure}ms_ROUND_{bg_round_name}.tif')
            #print(f'{marker_path}/{sample_id}_Background_CY2_{bg_cy2_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy3_exposure = metadata_cy3[bg_round_id]['ms']
            bg_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY3_{bg_cy3_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy5_exposure = metadata_cy5[bg_round_id]['ms']
            bg_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY5_{bg_cy5_exposure}ms_ROUND_{bg_round_name}.tif')

            cur_gfp_image_normalized_corrected = histogram_normalization(bg_cy2_image, cur_cy2_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(bg_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(bg_cy5_image, cur_cy5_image)

        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY2_{sample_id}_{cur_cy2_marker}_normalized_corrected.tif', cur_gfp_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY3_{sample_id}_{cur_cy3_marker}_normalized_corrected.tif', cur_cy3_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY5_{sample_id}_{cur_cy5_marker}_normalized_corrected.tif', cur_cy5_image_normalized_corrected)