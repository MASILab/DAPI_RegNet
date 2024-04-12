from ants_reg import ImageRegistration
import ants

# Create an ImageRegistration object with the fixed and moving images   
tiff1='GCA020TIB_TISSUE03_DAPI_DAPI_12ms_ROUND_07_initial_reg.tif'
tiff2='GCA020TIB_TISSUE03_DAPI_DAPI_12ms_ROUND_10_initial_reg.tif'

fixed_image_path = '/nfs2/baos1/rudravg/'+tiff2
moving_image_path = '/nfs2/baos1/rudravg/'+tiff1
registration = ImageRegistration(fixed_image_path, moving_image_path)

# Perform rigid registration
#rigid_result = registration.register_rigid()
# Perform affine registration
#affine_result = registration.register_affine()
# Perform SyN registration
#syn_result = registration.register_syn()
#Save the registered images in reg_check folder
#ants.image_write(rigid_result, 'reg_check/rigid_'+tiff2)
#ants.image_write(affine_result, 'reg_check/affine_'+tiff2)
#ants.image_write(syn_result, 'reg_check/syn_'+tiff2)

#syn_aggro=registration.register_syn_aggro()
#ants.image_write(syn_aggro, 'reg_check/syn_aggro_'+tiff2)

syn_cc=registration.register_syn_cc()
ants.image_write(syn_cc, 'reg_check/syn_cc_'+tiff2)