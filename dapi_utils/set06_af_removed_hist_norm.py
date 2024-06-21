import numpy as np
import tifffile as tiff
import cv2

metadata_cy3=[]
metadata_cy5=[]
metadata_cy2=[]
metadata_cy2.append({'ROUND_ID': '00','Marker':'AF','ms':200})
metadata_cy2.append({'ROUND_ID': '01','Marker':'EMPTY','ms':200})
metadata_cy2.append({'ROUND_ID': '02','Marker':'BACKGROUND','ms':175})
metadata_cy2.append({'ROUND_ID': '03','Marker':'MUC2','ms':200})
metadata_cy2.append({'ROUND_ID': '04','Marker':'BACKGROUND','ms':200})
metadata_cy2.append({'ROUND_ID': '05','Marker':'PCNA','ms':450})
metadata_cy2.append({'ROUND_ID': '06','Marker':'BACKGROUND','ms':450})
metadata_cy2.append({'ROUND_ID': '07','Marker':'EMPTY','ms':450})
metadata_cy2.append({'ROUND_ID': '08','Marker':'EMPTY','ms':450})
metadata_cy2.append({'ROUND_ID': '09','Marker':'CD11B','ms':320})
metadata_cy2.append({'ROUND_ID': '10','Marker':'BACKGROUND','ms':320})
metadata_cy2.append({'ROUND_ID': '11','Marker':'PANCK','ms':200})
metadata_cy2.append({'ROUND_ID': '12','Marker':'BACKGROUND','ms':320})
metadata_cy2.append({'ROUND_ID': '13','Marker':'PANCK-bad','ms':250})
metadata_cy2.append({'ROUND_ID': '14','Marker':'BACKGROUND','ms':250})
metadata_cy2.append({'ROUND_ID': '15','Marker':'VIMENTIN','ms':150})
metadata_cy2.append({'ROUND_ID': '16','Marker':'BACKGROUND','ms':150})
metadata_cy2.append({'ROUND_ID': '17','Marker':'LYSOZYME','ms':300})
metadata_cy2.append({'ROUND_ID': '18','Marker':'BACKGROUND','ms':300})
metadata_cy2.append({'ROUND_ID': '19','Marker':'PEGFR','ms':330})
metadata_cy3.append({'ROUND_ID': '00','Marker':'AF','ms':80})
metadata_cy3.append({'ROUND_ID': '01','Marker':'COLLAGEN','ms':12})
metadata_cy3.append({'ROUND_ID': '02','Marker':'BACKGROUND','ms':12})
metadata_cy3.append({'ROUND_ID': '03','Marker':'BCATENIN','ms':120})
metadata_cy3.append({'ROUND_ID': '04','Marker':'BACKGROUND','ms':120})
metadata_cy3.append({'ROUND_ID': '05','Marker':'CGA','ms':250})
metadata_cy3.append({'ROUND_ID': '06','Marker':'BACKGROUND','ms':250})
metadata_cy3.append({'ROUND_ID': '07','Marker':'CD68','ms':170})
metadata_cy3.append({'ROUND_ID': '08','Marker':'BACKGROUND','ms':170})
metadata_cy3.append({'ROUND_ID': '09','Marker':'CD3D','ms':75})
metadata_cy3.append({'ROUND_ID': '10','Marker':'BACKGROUND','ms':75})
metadata_cy3.append({'ROUND_ID': '11','Marker':'OLFM4-bad','ms':12})
metadata_cy3.append({'ROUND_ID': '12','Marker':'BACKGROUND','ms':75})
metadata_cy3.append({'ROUND_ID': '13','Marker':'OLFM4','ms':30})
metadata_cy3.append({'ROUND_ID': '14','Marker':'BACKGROUND','ms':30})
metadata_cy3.append({'ROUND_ID': '15','Marker':'SMA','ms':15})
metadata_cy3.append({'ROUND_ID': '16','Marker':'BACKGROUND','ms':15})
metadata_cy3.append({'ROUND_ID': '17','Marker':'SOX9','ms':135})
metadata_cy3.append({'ROUND_ID': '18','Marker':'BACKGROUND','ms':135})
metadata_cy3.append({'ROUND_ID': '19','Marker':'EMPTY','ms':135})
metadata_cy5.append({'ROUND_ID': '00','Marker':'AF','ms':1000})
metadata_cy5.append({'ROUND_ID': '01','Marker':'CD45','ms':500})
metadata_cy5.append({'ROUND_ID': '02','Marker':'BACKGROUND','ms':500})
metadata_cy5.append({'ROUND_ID': '03','Marker':'CD20','ms':770})
metadata_cy5.append({'ROUND_ID': '04','Marker':'BACKGROUND','ms':770})
metadata_cy5.append({'ROUND_ID': '05','Marker':'HLAA','ms':100})
metadata_cy5.append({'ROUND_ID': '06','Marker':'BACKGROUND','ms':100})
metadata_cy5.append({'ROUND_ID': '07','Marker':'CD4','ms':500})
metadata_cy5.append({'ROUND_ID': '08','Marker':'BACKGROUND','ms':500})
metadata_cy5.append({'ROUND_ID': '09','Marker':'NAKATPASE','ms':100})
metadata_cy5.append({'ROUND_ID': '10','Marker':'BACKGROUND','ms':100})
metadata_cy5.append({'ROUND_ID': '11','Marker':'CD8','ms':650})
metadata_cy5.append({'ROUND_ID': '12','Marker':'BACKGROUND','ms':100})
metadata_cy5.append({'ROUND_ID': '13','Marker':'CD8-bad','ms':700})
metadata_cy5.append({'ROUND_ID': '14','Marker':'BACKGROUND','ms':700})
metadata_cy5.append({'ROUND_ID': '15','Marker':'FOXP3','ms':1500})
metadata_cy5.append({'ROUND_ID': '16','Marker':'BACKGROUND','ms':1500})
metadata_cy5.append({'ROUND_ID': '17','Marker':'ERBB2','ms':2000})
metadata_cy5.append({'ROUND_ID': '18','Marker':'BACKGROUND','ms':2000})
metadata_cy5.append({'ROUND_ID': '19','Marker':'ACTG1','ms':1000})

sample_id_list = ['GCA007ACB','GCA007TIB_TISSUE01','GCA007TIB_TISSUE02','GCA053ACB_TISSUE01','GCA053ACB_TISSUE02','GCA053TIA_TISSUE01','GCA053TIA_TISSUE02','GCA093ACB_TISSUE01','GCA093ACB_TISSUE02','GCA093TIA','GCA094ACA_TISSUE01','GCA094ACA_TISSUE02','GCA094TIB_TISSUE01','GCA094TIB_TISSUE02','GCA096ACB','GCA096TIB','GCA099TIA','GCA112ACB','GCA112TIA','GCA113ACA','GCA113TIA','GCA118ACB_TISSUE01','GCA118ACB_TISSUE02','GCA118TIA_TISSUE01','GCA118TIA_TISSUE02','GCA132ACB_TISSUE01','GCA132ACB_TISSUE02','GCA132ACB_TISSUE03','GCA132TIA_TISSUE01','GCA132TIA_TISSUE02','GCA132TIA_TISSUE03']
sample_id_list = ['GCA112TIA']

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
for i in range(0,20):
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

    marker_path = f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set06/GCA112TIA/Registered'

    marker_tmp_result_path = f'/home-local/rudravg/trial'
    # Let's set AF image as reference
    AF_ROUND_SET06 = '00' # hardcoded, can search metadata to get the round id.
    af_cy2_exposure = metadata_cy2[0]['ms']
    af_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY2_{af_cy2_exposure}ms_ROUND_{AF_ROUND_SET06}.tif')
    print(f'{marker_path}/{sample_id}_AF_cy2_{af_cy2_exposure}ms_ROUND_{AF_ROUND_SET06}.tif')

    af_cy3_exposure = metadata_cy3[0]['ms']
    af_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET06}.tif')
    print(f'{marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET06}.tif')

    af_cy5_exposure = metadata_cy5[0]['ms']
    af_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY5_{af_cy5_exposure}ms_ROUND_{AF_ROUND_SET06}.tif')
    print(f'{marker_path}/{sample_id}_AF_CY5_{af_cy5_exposure}ms_ROUND_{AF_ROUND_SET06}.tif')
    
    for cur_round_id in range(1,20,2): # jump step wise by 2. in total there are 17 rounds. 1,3,5,7,9,11,13,15,17,19
        cur_round_name = marker_round[cur_round_id]

        cur_cy2_exposure = metadata_cy2[cur_round_id]['ms']
        cur_cy2_marker = metadata_cy2[cur_round_id]['Marker']
        if int(cur_round_id) not in [1,7,13] : # empty image for round 1 and round 7, round 13 is panck-bad
            cur_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy2_marker}_CY2_{cur_cy2_exposure}ms_ROUND_{cur_round_name}.tif')
            print(f'{marker_path}/{sample_id}_{cur_cy2_marker}_cy2_{cur_cy2_exposure}ms_ROUND_{cur_round_name}.tif')
        else: # hack for empty files only cy2 and cy3 contains empty markers
            cur_cy2_image = af_cy2_image

        if int(cur_round_id) not in [11,19]: # cy3 round 13 is OLFM4-bad which we don't have, 19 is empty
            cur_cy3_exposure = metadata_cy3[cur_round_id]['ms']
            cur_cy3_marker = metadata_cy3[cur_round_id]['Marker']
            cur_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')
            print(f'{marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')
        else:
            cur_cy3_image = af_cy3_image

        cur_cy5_exposure = metadata_cy5[cur_round_id]['ms']
        cur_cy5_marker = metadata_cy5[cur_round_id]['Marker']

        if int(cur_round_id) not in [13]: # cd8-bad for round 13.
            cur_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy5_marker}_CY5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')
            print(f'{marker_path}/{sample_id}_{cur_cy5_marker}_CY5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')
        else:
            cur_cy5_image = af_cy5_image

        if cur_round_id == 1:
            #AF removed for round 1 ONLY. No background image is available
            print(f"AF Image Path: {marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET06}.tif")
            print(f"Current Marker Image Path: {marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif")

            # no background image available, only deal with af removal
            cur_cy2_image_normalized_corrected = histogram_normalization(af_cy2_image, cur_cy2_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(af_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(af_cy5_image, cur_cy5_image)

            
        else:
            # get the background image in the previous round
            bg_round_id = cur_round_id - 1 

            bg_round_name = marker_round[bg_round_id]

            if int(bg_round_id) != 8: # hack for missing background on cy2 round 08. the marker name is EMPTY...
            
                bg_cy2_exposure = metadata_cy2[bg_round_id]['ms']
                bg_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_BACKGROUND_CY2_{bg_cy2_exposure}ms_ROUND_{bg_round_name}.tif')
            else:
                bg_cy2_exposure = metadata_cy2[bg_round_id]['ms']
                bg_cy2_image = af_cy2_image

            bg_cy3_exposure = metadata_cy3[bg_round_id]['ms']
            bg_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_BACKGROUND_CY3_{bg_cy3_exposure}ms_ROUND_{bg_round_name}.tif')

            bg_cy5_exposure = metadata_cy5[bg_round_id]['ms']
            bg_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_BACKGROUND_CY5_{bg_cy5_exposure}ms_ROUND_{bg_round_name}.tif')


            print(f"AF Image Path: {marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_{AF_ROUND_SET06}.tif")
            print(f"Current Marker Image Path: {marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif")

            cur_cy2_image_normalized_corrected = histogram_normalization(bg_cy2_image, cur_cy2_image)
            cur_cy3_image_normalized_corrected = histogram_normalization(bg_cy3_image, cur_cy3_image)
            cur_cy5_image_normalized_corrected = histogram_normalization(bg_cy5_image, cur_cy5_image)

        
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY2_{sample_id}_{cur_cy2_marker}_normalized_corrected.tif', cur_cy2_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY3_{sample_id}_{cur_cy3_marker}_normalized_corrected.tif', cur_cy3_image_normalized_corrected)
        tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY5_{sample_id}_{cur_cy5_marker}_normalized_corrected.tif', cur_cy5_image_normalized_corrected)