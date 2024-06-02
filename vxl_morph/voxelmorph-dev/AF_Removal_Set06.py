import numpy as np
import tifffile as tiff
import cv2

class AF_Removal_Set_06:
    def __init__(self):
        self.metadata_cy2 = [
            {'ROUND_ID': '00','Marker':'AF','ms':200},
            {'ROUND_ID': '01','Marker':'EMPTY','ms':200},
            {'ROUND_ID': '02','Marker':'BACKGROUND','ms':175},
            {'ROUND_ID': '03','Marker':'MUC2','ms':200},
            {'ROUND_ID': '04','Marker':'BACKGROUND','ms':200},
            {'ROUND_ID': '05','Marker':'PCNA','ms':450},
            {'ROUND_ID': '06','Marker':'BACKGROUND','ms':450},
            {'ROUND_ID': '07','Marker':'EMPTY','ms':450},
            {'ROUND_ID': '08','Marker':'EMPTY','ms':450},
            {'ROUND_ID': '09','Marker':'CD11B','ms':320},
            {'ROUND_ID': '10','Marker':'BACKGROUND','ms':320},
            {'ROUND_ID': '11','Marker':'PANCK','ms':200},
            {'ROUND_ID': '12','Marker':'BACKGROUND','ms':320},
            {'ROUND_ID': '13','Marker':'PANCK-bad','ms':250},
            {'ROUND_ID': '14','Marker':'BACKGROUND','ms':250},
            {'ROUND_ID': '15','Marker':'VIMENTIN','ms':150},
            {'ROUND_ID': '16','Marker':'BACKGROUND','ms':150},
            {'ROUND_ID': '17','Marker':'LYSOZYME','ms':300},
            {'ROUND_ID': '18','Marker':'BACKGROUND','ms':300},
            {'ROUND_ID': '19','Marker':'PEGFR','ms':330}
        ]

        self.metadata_cy3 = [
            {'ROUND_ID': '00','Marker':'AF','ms':80},
            {'ROUND_ID': '01','Marker':'COLLAGEN','ms':12},
            {'ROUND_ID': '02','Marker':'BACKGROUND','ms':12},
            {'ROUND_ID': '03','Marker':'BCATENIN','ms':120},
            {'ROUND_ID': '04','Marker':'BACKGROUND','ms':120},
            {'ROUND_ID': '05','Marker':'CGA','ms':250},
            {'ROUND_ID': '06','Marker':'BACKGROUND','ms':250},
            {'ROUND_ID': '07','Marker':'CD68','ms':170},
            {'ROUND_ID': '08','Marker':'BACKGROUND','ms':170},
            {'ROUND_ID': '09','Marker':'CD3D','ms':75},
            {'ROUND_ID': '10','Marker':'BACKGROUND','ms':75},
            {'ROUND_ID': '11','Marker':'OLFM4-bad','ms':12},
            {'ROUND_ID': '12','Marker':'BACKGROUND','ms':75},
            {'ROUND_ID': '13','Marker':'OLFM4','ms':30},
            {'ROUND_ID': '14','Marker':'BACKGROUND','ms':30},
            {'ROUND_ID': '15','Marker':'SMA','ms':15},
            {'ROUND_ID': '16','Marker':'BACKGROUND','ms':15},
            {'ROUND_ID': '17','Marker':'SOX9','ms':135},
            {'ROUND_ID': '18','Marker':'BACKGROUND','ms':135},
            {'ROUND_ID': '19','Marker':'EMPTY','ms':135}
        ]

        self.metadata_cy5 = [
            {'ROUND_ID': '00','Marker':'AF','ms':1000},
            {'ROUND_ID': '01','Marker':'CD45','ms':500},
            {'ROUND_ID': '02','Marker':'BACKGROUND','ms':500},
            {'ROUND_ID': '03','Marker':'CD20','ms':770},
            {'ROUND_ID': '04','Marker':'BACKGROUND','ms':770},
            {'ROUND_ID': '05','Marker':'HLAA','ms':100},
            {'ROUND_ID': '06','Marker':'BACKGROUND','ms':100},
            {'ROUND_ID': '07','Marker':'CD4','ms':500},
            {'ROUND_ID': '08','Marker':'BACKGROUND','ms':500},
            {'ROUND_ID': '09','Marker':'NAKATPASE','ms':100},
            {'ROUND_ID': '10','Marker':'BACKGROUND','ms':100},
            {'ROUND_ID': '11','Marker':'CD8','ms':650},
            {'ROUND_ID': '12','Marker':'BACKGROUND','ms':100},
            {'ROUND_ID': '13','Marker':'CD8-bad','ms':700},
            {'ROUND_ID': '14','Marker':'BACKGROUND','ms':700},
            {'ROUND_ID': '15','Marker':'FOXP3','ms':1500},
            {'ROUND_ID': '16','Marker':'BACKGROUND','ms':1500},
            {'ROUND_ID': '17','Marker':'ERBB2','ms':2000},
            {'ROUND_ID': '18','Marker':'BACKGROUND','ms':2000},
            {'ROUND_ID': '19','Marker':'ACTG1','ms':1000}
        ]
        
        self.marker_round = ['%02d' % i for i in range(20)]

    @staticmethod
    def histogram_normalization(img_bg, img_marker):
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
        af_cy2_exposure = self.metadata_cy2[0]['ms']
        af_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY2_{af_cy2_exposure}ms_ROUND_00.tif')

        af_cy3_exposure = self.metadata_cy3[0]['ms']
        af_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY3_{af_cy3_exposure}ms_ROUND_00.tif')

        af_cy5_exposure = self.metadata_cy5[0]['ms']
        af_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_AF_CY5_{af_cy5_exposure}ms_ROUND_00.tif')

        for cur_round_id in range(1, 20, 2):
            cur_round_name = self.marker_round[cur_round_id]

            cur_cy2_exposure = self.metadata_cy2[cur_round_id]['ms']
            cur_cy2_marker = self.metadata_cy2[cur_round_id]['Marker']
            if cur_round_id not in [1, 7, 13]:
                cur_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy2_marker}_CY2_{cur_cy2_exposure}ms_ROUND_{cur_round_name}.tif')
            else:
                cur_cy2_image = af_cy2_image

            if cur_round_id not in [11, 19]:
                cur_cy3_exposure = self.metadata_cy3[cur_round_id]['ms']
                cur_cy3_marker = self.metadata_cy3[cur_round_id]['Marker']
                cur_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy3_marker}_CY3_{cur_cy3_exposure}ms_ROUND_{cur_round_name}.tif')
            else:
                cur_cy3_image = af_cy3_image

            cur_cy5_exposure = self.metadata_cy5[cur_round_id]['ms']
            cur_cy5_marker = self.metadata_cy5[cur_round_id]['Marker']
            if cur_round_id not in [13]:
                cur_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_{cur_cy5_marker}_CY5_{cur_cy5_exposure}ms_ROUND_{cur_round_name}.tif')
            else:
                cur_cy5_image = af_cy5_image

            if cur_round_id == 1:
                cur_cy2_image_normalized_corrected = self.histogram_normalization(af_cy2_image, cur_cy2_image)
                cur_cy3_image_normalized_corrected = self.histogram_normalization(af_cy3_image, cur_cy3_image)
                cur_cy5_image_normalized_corrected = self.histogram_normalization(af_cy5_image, cur_cy5_image)
            else:
                bg_round_id = cur_round_id - 1
                bg_round_name = self.marker_round[bg_round_id]

                if bg_round_id != 8:
                    bg_cy2_exposure = self.metadata_cy2[bg_round_id]['ms']
                    bg_cy2_image = tiff.imread(f'{marker_path}/{sample_id}_BACKGROUND_CY2_{bg_cy2_exposure}ms_ROUND_{bg_round_name}.tif')
                else:
                    bg_cy2_image = af_cy2_image

                bg_cy3_exposure = self.metadata_cy3[bg_round_id]['ms']
                bg_cy3_image = tiff.imread(f'{marker_path}/{sample_id}_BACKGROUND_CY3_{bg_cy3_exposure}ms_ROUND_{bg_round_name}.tif')

                bg_cy5_exposure = self.metadata_cy5[bg_round_id]['ms']
                bg_cy5_image = tiff.imread(f'{marker_path}/{sample_id}_BACKGROUND_CY5_{bg_cy5_exposure}ms_ROUND_{bg_round_name}.tif')

                cur_cy2_image_normalized_corrected = self.histogram_normalization(bg_cy2_image, cur_cy2_image)
                cur_cy3_image_normalized_corrected = self.histogram_normalization(bg_cy3_image, cur_cy3_image)
                cur_cy5_image_normalized_corrected = self.histogram_normalization(bg_cy5_image, cur_cy5_image)

            tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY2_{sample_id}_{cur_cy2_marker}_normalized_corrected.tif', cur_cy2_image_normalized_corrected)
            tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY3_{sample_id}_{cur_cy3_marker}_normalized_corrected.tif', cur_cy3_image_normalized_corrected)
            tiff.imwrite(f'{marker_tmp_result_path}/ROUND_{cur_round_name}_CY5_{sample_id}_{cur_cy5_marker}_normalized_corrected.tif', cur_cy5_image_normalized_corrected)

# Example usage:
# af_removal_object = AF_Removal_Set_06()
# af_removal_object.process_sample(sample_id='GCA112TIA', marker_path='/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/GCA112TIA', marker_tmp_result_path='/home-local/rudravg/trial')
