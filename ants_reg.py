import ants

class ImageRegistration:
    def __init__(self, fixed_image_path, moving_image_path):
        """
        Initialize the ImageRegistration class with the fixed and moving images.
        """
        self.fixed_image = ants.image_read(fixed_image_path)
        self.moving_image = ants.image_read(moving_image_path)

    def register_rigid(self):
        """
        Perform rigid registration.
        """
        result = ants.registration(fixed=self.fixed_image, moving=self.moving_image, type_of_transform='Rigid')
        return result['warpedmovout']

    def register_affine(self):
        """
        Perform affine registration.
        """
        result = ants.registration(fixed=self.fixed_image, moving=self.moving_image, type_of_transform='Affine')
        return result['warpedmovout']

    def register_syn(self):
        """
        Perform SyN registration.
        """
        result = ants.registration(fixed=self.fixed_image, moving=self.moving_image, type_of_transform='SyN')
        return result['warpedmovout']
    
    def register_syn_aggro(self):
        """
        Perform SyNAggro registration.
        """
        result = ants.registration(fixed=self.fixed_image, moving=self.moving_image, type_of_transform='SyNAggro', reg_iterations=(100, 70, 50, 30))
        return result['warpedmovout']
    def register_syn_cc(self):
        """
        Perform SyNAggro registration.
        """
        result = ants.registration(fixed=self.fixed_image, moving=self.moving_image, type_of_transform='SyNCC', reg_iterations=(100, 70, 50, 30))
        return result['warpedmovout']
