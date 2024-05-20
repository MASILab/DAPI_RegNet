import numpy as np
import tifffile as tiff
import cv2
import os

class AF_Removal_Set_02:
    def __init__(self):
        self.metadata_cy2 = self.load_metadata_cy2()
        self.metadata_cy3 = self.load_metadata_cy3()
        self.metadata_cy5 = self.load_metadata_cy5()
        self.marker_round = self.generate_marker_rounds()
        
    def load_metadata_cy2(self):
        return [
            {'ROUND_ID': '00','Marker':'AF','ms':75},
            {'ROUND_ID': '01','Marker':'Background','ms':75},
            {'ROUND_ID': '02','Marker':'Muc2','ms':75},
            {'ROUND_ID': '03','Marker':'Background','ms':75},
            {'ROUND_ID': '04','Marker':'CD11B','ms':500},
            {'ROUND_ID': '05','Marker':'Background','ms':500},
            {'ROUND_ID': '06','Marker':'PCNA','ms':175},
            {'ROUND_ID': '07','Marker':'Background','ms':175},
            {'ROUND_ID': '08','Marker':'pEGFR','ms':175},
            {'ROUND_ID': '09','Marker':'Background','ms':175},
            {'ROUND_ID': '10','Marker':'Cox2','ms':500},
            {'ROUND_ID': '11','Marker':'Background','ms':500},
            {'ROUND_ID': '12','Marker':'PanCK','ms':100},
            {'ROUND_ID': '13','Marker':'Background','ms':100},
            {'ROUND_ID': '14','Marker':'ACTININ','ms':350},
            {'ROUND_ID': '15','Marker':'Background','ms':350},
            {'ROUND_ID': '16','Marker':'Vimentin','ms':75},
            {'ROUND_ID': '17','Marker':'Background','ms':75},
            {'ROUND_ID': '18','Marker':'Lysozyme','ms':200}
        ]
    
    def load_metadata_cy3(self):
        return [
            {'ROUND_ID': '00','Marker':'SNA','ms':50},
            {'ROUND_ID': '01','Marker':'Background','ms':50},
            {'ROUND_ID': '02','Marker':'ACTG1','ms':100},
            {'ROUND_ID': '03','Marker':'Background','ms':100},
            {'ROUND_ID': '04','Marker':'Collagen','ms':100},
            {'ROUND_ID': '05','Marker':'Background','ms':100},
            {'ROUND_ID': '06','Marker':'BCATENIN','ms':50},
            {'ROUND_ID': '07','Marker':'Background','ms':50},
            {'ROUND_ID': '08','Marker':'CgA','ms':150},
            {'ROUND_ID': '09','Marker':'Background','ms':150},
            {'ROUND_ID': '10','Marker':'CD3d','ms':150},
            {'ROUND_ID': '11','Marker':'Background','ms':150},
            {'ROUND_ID': '12','Marker':'OLFM4','ms':50},
            {'ROUND_ID': '13','Marker':'Background','ms':50},
            {'ROUND_ID': '14','Marker':'CD68','ms':100},
            {'ROUND_ID': '15','Marker':'Background','ms':100},
            {'ROUND_ID': '16','Marker':'Sox9','ms':150},
            {'ROUND_ID': '17','Marker':'Background','ms':150},
            {'ROUND_ID': '18','Marker':'SMA','ms':12}
        ]
    
    def load_metadata_cy5(self):
        return [
            {'ROUND_ID': '00','Marker':'WGA','ms':75},
            {'ROUND_ID': '01','Marker':'Background','ms':75},
            {'ROUND_ID': '02','Marker':'CD45','ms':1000},
            {'ROUND_ID': '03','Marker':'Background','ms':1000},
            {'ROUND_ID': '04','Marker':'CD20','ms':500},
            {'ROUND_ID': '05','Marker':'Background','ms':500},
            {'ROUND_ID': '06','Marker':'pSTAT3','ms':500},
            {'ROUND_ID': '07','Marker':'Background','ms':500},
            {'ROUND_ID': '08','Marker':'CD4','ms':500},
            {'ROUND_ID': '09','Marker':'Background','ms':500},
            {'ROUND_ID': '10','Marker':'HLAA','ms':200},
            {'ROUND_ID': '11','Marker':'Background','ms':200},
            {'ROUND_ID': '12','Marker':'CD8','ms':500},
            {'ROUND_ID': '13','Marker':'Background','ms':500},
            {'ROUND_ID': '14','Marker':'NaKATPase','ms':100},
            {'ROUND_ID': '15','Marker':'Background','ms':100},
            {'ROUND_ID': '16','Marker':'FOXP3','ms':500},
            {'ROUND_ID': '17','Marker':'Background','ms':500},
            {'ROUND_ID': '18','Marker':'ERBB2','ms':1000}
        ]
    
    def generate_marker_rounds(self):
        return [f"{i:02d}" for i in range(19)]

    def histogram_normalization(self, img_bg, img_marker):
        hist1, _ = np.histogram(img_bg.flatten(), 256, [0, 256])
        hist2, _ = np.histogram(img_marker.flatten(), 256, [0, 256])
        
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        
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
        af_removed_image = cv2.max(af_removed_image, 0)
        
        return af_removed_image

    def process_sample(self, sample_id, marker_path, marker_tmp_result_path):
                
        if not os.path.exists(marker_tmp_result_path):
            os.mkdir(marker_tmp_result_path)
        
        for cur_round_id in range(0, 19, 2):
            cur_round_name = self.marker_round[cur_round_id]
            
            cur_cy2_exposure = self.metadata_cy2[cur_round_id]['ms']
            cur_cy2_marker = self.metadata_cy2[cur_round_id]['Marker']
            cur_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy2_marker}_CY2_{cur_cy2_exposure}ms_ROUND_{cur_round_name}.tif')
            
            cur_cy3_exposure = self.metadata_cy3[cur_round_id]['ms']
            cur_cy3_marker = self.metadata_cy3[cur_round_id]['Marker']
            cur_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')
            
            cur_cy5_exposure = self.metadata_cy5[cur_round_id]['ms']
            cur_cy5_marker = self.metadata_cy5[cur_round_id]['Marker']
            cur_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy5_marker}_CY5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')
            
            if cur_round_id == 0:
                cur_gfp_image_normalized_corrected = cur_cy2_image
                cur_cy3_image_normalized_corrected = cur_cy3_image
                cur_cy5_image_normalized_corrected = cur_cy5_image
            else:
                bg_round_id = cur_round_id - 1
                bg_round_name = self.marker_round[bg_round_id]
                
                bg_cy2_exposure = self.metadata_cy2[bg_round_id]['ms']
                bg_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY2_{bg_cy2_exposure}ms_ROUND_{bg_round_name}.tif')
                
                bg_cy3_exposure = self.metadata_cy3[bg_round_id]['ms']
                bg_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY3_{bg_cy3_exposure}ms_ROUND_{bg_round_name}.tif')
                
                bg_cy5_exposure = self.metadata_cy5[bg_round_id]['ms']
                bg_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_Background_CY5_{bg_cy5_exposure}ms_ROUND_{bg_round_name}.tif')

                cur_gfp_image_normalized_corrected = self.histogram_normalization(bg_cy2_image, cur_cy2_image)
                cur_cy3_image_normalized_corrected = self.histogram_normalization(bg_cy3_image, cur_cy3_image)
                cur_cy5_image_normalized_corrected = self.histogram_normalization(bg_cy5_image, cur_cy5_image)

            tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY2_{sample_id}_{cur_cy2_marker}_normalized_corrected.tif', cur_gfp_image_normalized_corrected)
            tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY3_{sample_id}_{cur_cy3_marker}_normalized_corrected.tif', cur_cy3_image_normalized_corrected)
            tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY5_{sample_id}_{cur_cy5_marker}_normalized_corrected.tif', cur_cy5_image_normalized_corrected)
        