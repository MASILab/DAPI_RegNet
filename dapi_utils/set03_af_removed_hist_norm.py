import numpy as np
import tifffile as tiff
import cv2

metadata_cy3=[]
metadata_cy5=[]
metadata_cy2=[]
metadata_cy3.append({'ROUND_ID': '00','Marker':'AF','ms':100})
metadata_cy3.append({'ROUND_ID': '01','Marker':'ACTG1','ms':100})
metadata_cy3.append({'ROUND_ID': '02','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '03','Marker':'COLLAGEN','ms':40})
metadata_cy3.append({'ROUND_ID': '04','Marker':'Background','ms':40})
metadata_cy3.append({'ROUND_ID': '05','Marker':'BCATENIN','ms':100})
metadata_cy3.append({'ROUND_ID': '06','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '07','Marker':'CGA','ms':200})
metadata_cy3.append({'ROUND_ID': '08','Marker':'Background','ms':200})
metadata_cy3.append({'ROUND_ID': '09','Marker':'CD3D','ms':150})
metadata_cy3.append({'ROUND_ID': '10','Marker':'Background','ms':150})
metadata_cy3.append({'ROUND_ID': '11','Marker':'OLFM4','ms':100})
metadata_cy3.append({'ROUND_ID': '12','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '13','Marker':'CD68','ms':300})
metadata_cy3.append({'ROUND_ID': '14','Marker':'Background','ms':300})
metadata_cy3.append({'ROUND_ID': '15','Marker':'SOX9','ms':200})
metadata_cy3.append({'ROUND_ID': '16','Marker':'Background','ms':200})
metadata_cy3.append({'ROUND_ID': '17','Marker':'SMA','ms':12})
metadata_cy5.append({'ROUND_ID': '00','Marker':'AF','ms':800})
metadata_cy5.append({'ROUND_ID': '01','Marker':'CD45','ms':800})
metadata_cy5.append({'ROUND_ID': '02','Marker':'Background','ms':800})
metadata_cy5.append({'ROUND_ID': '03','Marker':'CD20','ms':600})
metadata_cy5.append({'ROUND_ID': '04','Marker':'Background','ms':600})
metadata_cy5.append({'ROUND_ID': '05','Marker':'PSTAT3','ms':550})
metadata_cy5.append({'ROUND_ID': '06','Marker':'Background','ms':550})
metadata_cy5.append({'ROUND_ID': '07','Marker':'CD4','ms':500})
metadata_cy5.append({'ROUND_ID': '08','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '09','Marker':'HLAA','ms':100})
metadata_cy5.append({'ROUND_ID': '10','Marker':'Background','ms':100})
metadata_cy5.append({'ROUND_ID': '11','Marker':'CD8','ms':500})
metadata_cy5.append({'ROUND_ID': '12','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '13','Marker':'NAKATPASE','ms':75})
metadata_cy5.append({'ROUND_ID': '14','Marker':'Background','ms':75})
metadata_cy5.append({'ROUND_ID': '15','Marker':'FOXP3','ms':1000})
metadata_cy5.append({'ROUND_ID': '16','Marker':'Background','ms':1000})
metadata_cy5.append({'ROUND_ID': '17','Marker':'ERBB2','ms':1000})
metadata_cy2.append({'ROUND_ID': '00','Marker':'AF','ms':115})
metadata_cy2.append({'ROUND_ID': '01','Marker':'MUC2','ms':115})
metadata_cy2.append({'ROUND_ID': '02','Marker':'Background','ms':115})
metadata_cy2.append({'ROUND_ID': '03','Marker':'CD11B','ms':600})
metadata_cy2.append({'ROUND_ID': '04','Marker':'Background','ms':600})
metadata_cy2.append({'ROUND_ID': '05','Marker':'PCNA','ms':200})
metadata_cy2.append({'ROUND_ID': '06','Marker':'Background','ms':200})
metadata_cy2.append({'ROUND_ID': '07','Marker':'PEGFR','ms':150})
metadata_cy2.append({'ROUND_ID': '08','Marker':'Background','ms':150})
metadata_cy2.append({'ROUND_ID': '09','Marker':'COX2','ms':500})
metadata_cy2.append({'ROUND_ID': '10','Marker':'Background','ms':500})
metadata_cy2.append({'ROUND_ID': '11','Marker':'PANCK','ms':100})
metadata_cy2.append({'ROUND_ID': '12','Marker':'Background','ms':100})
metadata_cy2.append({'ROUND_ID': '13','Marker':'ACTININ','ms':350})
metadata_cy2.append({'ROUND_ID': '14','Marker':'Background','ms':350})
metadata_cy2.append({'ROUND_ID': '15','Marker':'VIMENTIN','ms':75})
metadata_cy2.append({'ROUND_ID': '16','Marker':'Background','ms':75})
metadata_cy2.append({'ROUND_ID': '17','Marker':'LYSOZYME','ms':250})

#sample_id_list = ['GCA019ACA','GCA019TIA_TISSUE01','GCA019TIA_TISSUE02','GCA054ACB_TISSUE01','GCA054ACB_TISSUE02','GCA054ACB_TISSUE03','GCA054ACB_TISSUE04','GCA054TIA_TISSUE01','GCA054TIA_TISSUE02','GCA062ACA_TISSUE01','GCA062ACA_TISSUE02','GCA062TIA_TISSUE01','GCA062TIA_TISSUE02','GCA062TIA_TISSUE03','GCA066ACB_TISSUE01','GCA066ACB_TISSUE02','GCA066TIB_TISSUE01','GCA066TIB_TISSUE02','GCA069ACB_TISSUE01','GCA069ACB_TISSUE02','GCA069TIB_TISSUE01','GCA069TIB_TISSUE02','GCA071ACB_TISSUE01','GCA071ACB_TISSUE02','GCA071ACB_TISSUE03','GCA071TIA','GCA072ACB_TISSUE01','GCA072ACB_TISSUE02','GCA072TIB_TISSUE01','GCA072TIB_TISSUE02','GCA075ACB_TISSUE01','GCA075ACB_TISSUE02','GCA075TIB_TISSUE01','GCA075TIB_TISSUE02','GCA077ACB_TISSUE01','GCA077ACB_TISSUE02','GCA077TIA','GCA081ACB_TISSUE01','GCA081ACB_TISSUE02','GCA081TIB_TISSUE01','GCA081TIB_TISSUE02']
sample_id_list = ['GCA054ACB_TISSUE02']

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


marker_round = []
for i in range(0,19):
    if i < 10:
        tmp_round_id = '0%s' % i
    else:
        tmp_round_id = i
    marker_round.append(tmp_round_id)

for sample_id in sample_id_list:
    print(sample_id)

    if 'TISSUE' in sample_id:
        sample_dir = sample_id.split('_TISSUE')[0]
    else:
        sample_dir = sample_id

    marker_path = f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set03/{sample_dir}/Registered'
    marker_tmp_result_path = f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set03/{sample_dir}/tmp'
    # Let's set AF image as reference
    AF_ROUND_SET03 = '00' # hardcoded, can search metadata to get the round id.
    af_cy2_exposure = metadata_cy2[0]['ms']
    af_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY2_{af_cy2_exposure}ms_ROUND_{AF_ROUND_SET03}.tif')
    print(f'{marker_path}/{sample_id}_AF_CY2_{af_cy2_exposure}ms_ROUND_{AF_ROUND_SET03}.tif')

    af_cy3_exposure = metadata_cy3[0]['ms']
    af_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET03}.tif')
    print(f'{marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET03}.tif')

    af_cy5_exposure = metadata_cy5[0]['ms']
    af_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY5_{af_cy5_exposure}ms_ROUND_{AF_ROUND_SET03}.tif')
    print(f'{marker_path}/{sample_id}_AF_CY5_{af_cy5_exposure}ms_ROUND_{AF_ROUND_SET03}.tif')

    # af_cy2_image_normalized = af_cy2_image #/ np.max(af_cy2_image) #/ 255 #/ af_cy2_exposure
    # af_cy3_image_normalized = af_cy3_image #/ np.max(af_cy3_image)#/ 255 #/ af_cy3_exposure
    # af_cy5_image_normalized = af_cy5_image #/ np.max(af_cy5_image)#/ 255 #/ af_cy5_exposure
    
    for cur_round_id in range(1,18,2): # jump step wise by 2. in total there are 17 rounds. 1,3,5,7,9,11,13,15,17
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

        if cur_round_id == 1:
            #AF removed for round 1 ONLY. No background image is available

            # no background image available, only deal with af removal
            cur_cy2_image_normalized_corrected = histogram_normalization(af_cy2_image, cur_cy2_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(af_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(af_cy5_image, cur_cy5_image)

            
        else:
            # get the background image in the previous round
            bg_round_id = cur_round_id - 1 

            bg_round_name = marker_round[bg_round_id]

            bg_cy2_exposure = metadata_cy2[bg_round_id]['ms']
            bg_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY2_{bg_cy2_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy3_exposure = metadata_cy3[bg_round_id]['ms']
            bg_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY3_{bg_cy3_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy5_exposure = metadata_cy5[bg_round_id]['ms']
            bg_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY5_{bg_cy5_exposure}ms_ROUND_{bg_round_name}.tif')

            cur_cy2_image_normalized_corrected = histogram_normalization(bg_cy2_image, cur_cy2_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(bg_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(bg_cy5_image, cur_cy5_image)
               
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY2_{sample_id}_{cur_cy2_marker}_normalized_corrected.tif', cur_cy2_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY3_{sample_id}_{cur_cy3_marker}_normalized_corrected.tif', cur_cy3_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY5_{sample_id}_{cur_cy5_marker}_normalized_corrected.tif', cur_cy5_image_normalized_corrected)