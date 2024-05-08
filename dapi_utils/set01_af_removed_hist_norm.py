import numpy as np
import cv2
import tifffile as tiff

#Define whole sets metadata
metadata_cy5=[]
metadata_cy5.append({'ROUND_ID': '00','Marker':'SNA','ms':200})
metadata_cy5.append({'ROUND_ID': '01','Marker':'AF','ms':200})
metadata_cy5.append({'ROUND_ID': '02','Marker':'CD20','ms':100})
metadata_cy5.append({'ROUND_ID': '03','Marker':'Background','ms':100})
metadata_cy5.append({'ROUND_ID': '04','Marker':'PSTAT3','ms':910})
metadata_cy5.append({'ROUND_ID': '05','Marker':'Background','ms':910})
metadata_cy5.append({'ROUND_ID': '06','Marker':'CD4','ms':400})
metadata_cy5.append({'ROUND_ID': '07','Marker':'Background','ms':400})
metadata_cy5.append({'ROUND_ID': '08','Marker':'HLAA','ms':50})
metadata_cy5.append({'ROUND_ID': '09','Marker':'Background','ms':50})
metadata_cy5.append({'ROUND_ID': '10','Marker':'CD8','ms':150})
metadata_cy5.append({'ROUND_ID': '11','Marker':'Background','ms':150})
metadata_cy5.append({'ROUND_ID': '12','Marker':'NAKATPASE','ms':150})
metadata_cy5.append({'ROUND_ID': '13','Marker':'Background','ms':150})
metadata_cy5.append({'ROUND_ID': '14','Marker':'FOXP3','ms':1000})
metadata_cy5.append({'ROUND_ID': '15','Marker':'Background','ms':1000})
metadata_cy5.append({'ROUND_ID': '16','Marker':'ERBB2','ms':500})
metadata_cy5.append({'ROUND_ID': '17','Marker':'Background','ms':500})
metadata_cy5.append({'ROUND_ID': '18','Marker':'CD45','ms':1500})

metadata_cy3 = []
metadata_cy3.append({'ROUND_ID': '00','Marker':'COLLAGEN','ms':15})
metadata_cy3.append({'ROUND_ID': '01','Marker':'AF','ms':15})
metadata_cy3.append({'ROUND_ID': '02','Marker':'CD45 bad','ms':3000})
metadata_cy3.append({'ROUND_ID': '03','Marker':'Background','ms':3000})
metadata_cy3.append({'ROUND_ID': '04','Marker':'BCATENIN','ms':150})
metadata_cy3.append({'ROUND_ID': '05','Marker':'Background','ms':150})
metadata_cy3.append({'ROUND_ID': '06','Marker':'CGA','ms':200})
metadata_cy3.append({'ROUND_ID': '07','Marker':'Background','ms':200})
metadata_cy3.append({'ROUND_ID': '08','Marker':'CD3D','ms':50})
metadata_cy3.append({'ROUND_ID': '09','Marker':'Background','ms':50})
metadata_cy3.append({'ROUND_ID': '10','Marker':'OLFM4','ms':100})
metadata_cy3.append({'ROUND_ID': '11','Marker':'Background','ms':100})
metadata_cy3.append({'ROUND_ID': '12','Marker':'CD68','ms':75})
metadata_cy3.append({'ROUND_ID': '13','Marker':'Background','ms':75})
metadata_cy3.append({'ROUND_ID': '14','Marker':'SOX9','ms':150})
metadata_cy3.append({'ROUND_ID': '15','Marker':'Background','ms':150})
metadata_cy3.append({'ROUND_ID': '16','Marker':'SMA','ms':12})
metadata_cy3.append({'ROUND_ID': '17','Marker':'Background','ms':12})
metadata_cy3.append({'ROUND_ID': '18','Marker':'CD45 bad','ms':250})

metadata_gfp=[]
metadata_gfp.append({'ROUND_ID': '00','Marker':'MUC2','ms':100})
metadata_gfp.append({'ROUND_ID': '01','Marker':'AF','ms':100})
metadata_gfp.append({'ROUND_ID': '02','Marker':'CD11B','ms':250})
metadata_gfp.append({'ROUND_ID': '03','Marker':'Background','ms':250})
metadata_gfp.append({'ROUND_ID': '04','Marker':'PCNA','ms':200})
metadata_gfp.append({'ROUND_ID': '05','Marker':'Background','ms':200})
metadata_gfp.append({'ROUND_ID': '06','Marker':'PEGFR','ms':150})
metadata_gfp.append({'ROUND_ID': '07','Marker':'Background','ms':150})
metadata_gfp.append({'ROUND_ID': '08','Marker':'COX2','ms':250})
metadata_gfp.append({'ROUND_ID': '09','Marker':'Background','ms':250})
metadata_gfp.append({'ROUND_ID': '10','Marker':'PANCK','ms':150})
metadata_gfp.append({'ROUND_ID': '11','Marker':'Background','ms':150})
metadata_gfp.append({'ROUND_ID': '12','Marker':'ACTININ','ms':175})
metadata_gfp.append({'ROUND_ID': '13','Marker':'Background','ms':175})
metadata_gfp.append({'ROUND_ID': '14','Marker':'VIMENTIN','ms':100})
metadata_gfp.append({'ROUND_ID': '15','Marker':'Background','ms':100})
metadata_gfp.append({'ROUND_ID': '16','Marker':'LYSOZYME','ms':400})
metadata_gfp.append({'ROUND_ID': '17','Marker':'Background','ms':400})
metadata_gfp.append({'ROUND_ID': '18','Marker':'ACTG1','ms':75})

#sample_id_list = ['GCA002ACB','GCA002TIB','GCA003ACA','GCA003TIB','GCA004TIB','GCA011ACB','GCA011TIB','GCA012ACB','GCA012TIB']
sample_id_list = ['GCA003ACA']

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


def find_min_exposure_time(af_exp,bg_exp,marker_exp):
    if af_exp > bg_exp:
        if bg_exp > marker_exp:
            return 'marker'
        else:
            return 'background'
    else:
        if af_exp > marker_exp:
            return 'marker'
        else:
            return 'AF'

marker_round = []
for i in range(0,19):
    if i < 10:
        tmp_round_id = '0%s' % i
    else:
        tmp_round_id = i
    marker_round.append(tmp_round_id)

for sample_id in sample_id_list:
    print(sample_id)
    marker_path = f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set01/{sample_id}/Registered'
    marker_tmp_result_path = f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set01/{sample_id}/tmp'
    # Let's set AF image as reference
    AF_ROUND_SET01 = '01' # hardcoded, can search metadata to get the round id.
    af_gfp_exposure = metadata_gfp[1]['ms']
    af_gfp_image = tiff.imread(f'{marker_path}/{sample_id}_AF_GFP_{af_gfp_exposure}ms_ROUND_{AF_ROUND_SET01}.tif')
    print(f'{marker_path}/{sample_id}_AF_GFP_{af_gfp_exposure}ms_ROUND_{AF_ROUND_SET01}.tif')

    af_cy3_exposure = metadata_cy3[1]['ms']
    af_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_AF_Cy3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET01}.tif')
    print(f'{marker_path}/{sample_id}_AF_Cy3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET01}.tif')

    af_cy5_exposure = metadata_cy5[1]['ms']
    af_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_AF_Cy5_{af_cy5_exposure}ms_ROUND_{AF_ROUND_SET01}.tif')
    print(f'{marker_path}/{sample_id}_AF_Cy5_{af_cy5_exposure}ms_ROUND_{AF_ROUND_SET01}.tif')

    # af_gfp_image_normalized = af_gfp_image #/ np.max(af_gfp_image) #/ 255 #/ af_gfp_exposure
    # af_cy3_image_normalized = af_cy3_image #/ np.max(af_cy3_image)#/ 255 #/ af_cy3_exposure
    # af_cy5_image_normalized = af_cy5_image #/ np.max(af_cy5_image)#/ 255 #/ af_cy5_exposure
    
    for cur_round_id in range(0,19,2): # jump step wise by 2. in total there are 18 rounds. 0,2,4,6,8,10,12,14,16,18
        cur_round_name = marker_round[cur_round_id]

        if int(cur_round_name) == 18:# hardcode for Joe's naming
            cur_gfp_exposure = metadata_gfp[cur_round_id]['ms']
            cur_gfp_marker = metadata_gfp[cur_round_id]['Marker']
            # GCA003ACA_GACTIN_GFP_75ms_ROUND_ACTG1_GFP_75ms_ROUND_18.tif
            cur_gfp_image = tiff.imread(f'{marker_path}/{sample_id}_GACTIN_GFP_{cur_gfp_exposure}ms_ROUND_{cur_gfp_marker}_GFP_{cur_gfp_exposure}ms_ROUND_{cur_round_name}.tif')
            print(f'{marker_path}/{sample_id}_{cur_gfp_marker}_GFP_{cur_gfp_exposure}ms_ROUND_{cur_round_name}.tif')
        else:
            cur_gfp_exposure = metadata_gfp[cur_round_id]['ms']
            cur_gfp_marker = metadata_gfp[cur_round_id]['Marker']
            cur_gfp_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_gfp_marker}_GFP_{cur_gfp_exposure}ms_ROUND_{cur_round_name}.tif')
            print(f'{marker_path}/{sample_id}_{cur_gfp_marker}_GFP_{cur_gfp_exposure}ms_ROUND_{cur_round_name}.tif')

        cur_cy3_exposure = metadata_cy3[cur_round_id]['ms']
        cur_cy3_marker = metadata_cy3[cur_round_id]['Marker']
        cur_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy3_marker}_Cy3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')
        print(f'{marker_path}/{sample_id}_{cur_cy3_marker}_Cy3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')

        cur_cy5_exposure = metadata_cy5[cur_round_id]['ms']
        cur_cy5_marker = metadata_cy5[cur_round_id]['Marker']
        cur_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy5_marker}_Cy5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')
        print(f'{marker_path}/{sample_id}_{cur_cy5_marker}_Cy5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')

        if cur_round_id == 0 or cur_round_id == 2:
            #AF removed for round 0 and 2. Use round_01' AF image.

             # normalize image via exposure
            if cur_gfp_exposure > af_gfp_exposure :
                # cur_gfp_image is reference
                cur_gfp_image_normalized = cur_gfp_image
                af_gfp_image_normalize = af_gfp_image * (cur_gfp_exposure/af_gfp_exposure)
            else:
                # af_gfp_image is reference
                cur_gfp_image_normalized = cur_gfp_image * (af_gfp_exposure / cur_gfp_exposure)
                af_gfp_image_normalize = af_gfp_image

            if cur_cy3_exposure > af_cy3_exposure :
                # cur_cy3_image is reference
                cur_cy3_image_normalized = cur_cy3_image
                af_cy3_image_normalize = af_cy3_image * (cur_cy3_exposure/af_cy3_exposure)
            else:
                # af_cy3_image is reference
                cur_cy3_image_normalized = cur_cy3_image * (af_cy3_exposure / cur_cy3_exposure)
                af_cy3_image_normalize = af_cy3_image
            
            if cur_cy5_exposure > af_cy5_exposure :
                # cur_cy5_image is reference
                cur_cy5_image_normalized = cur_cy5_image
                af_cy5_image_normalize = af_cy5_image * (cur_cy5_exposure/af_cy5_exposure)
            else:
                # af_cy5_image is reference
                cur_cy5_image_normalized = cur_cy5_image * (af_cy5_exposure / cur_cy5_exposure)
                af_cy5_image_normalize = af_cy5_image

            # no background image available, only deal with af removal
            cur_gfp_image_normalized_corrected = histogram_normalization(af_gfp_image, cur_gfp_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(af_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(af_cy5_image, cur_cy5_image)
            
        else:
            # get the background image in the previous round
            bg_round_id = cur_round_id - 1 

            bg_round_name = marker_round[bg_round_id]

            bg_gfp_exposure = metadata_gfp[bg_round_id]['ms']
            bg_gfp_image = tiff.imread(f'{marker_path}/{sample_id}_Background_GFP_{bg_gfp_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy3_exposure = metadata_cy3[bg_round_id]['ms']
            bg_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_Background_Cy3_{bg_cy3_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy5_exposure = metadata_cy5[bg_round_id]['ms']
            bg_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_Background_Cy5_{bg_cy5_exposure}ms_ROUND_{bg_round_name}.tif')

            cur_gfp_image_normalized_corrected = histogram_normalization(bg_gfp_image, cur_gfp_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(bg_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(bg_cy5_image, cur_cy5_image)
               
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_GFP_{sample_id}_{cur_gfp_marker}_normalized_corrected.tif', cur_gfp_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_Cy3_{sample_id}_{cur_cy3_marker}_normalized_corrected.tif', cur_cy3_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_Cy5_{sample_id}_{cur_cy5_marker}_normalized_corrected.tif', cur_cy5_image_normalized_corrected)