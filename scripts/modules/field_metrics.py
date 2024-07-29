import numpy as np
from itertools import combinations

class FieldEvaluator:
    """
    This is a base class which has methods for evaluating fields.
    """
    def __init__(self) -> None:
        pass

    def get_fields_mag(self):
        """
        Returns the magnitude at each voxel
        Keeps the same grid, but collapses axis 3 which contains x,y,z information.
        """        
        return np.sqrt(np.sum(np.square(self.fields), axis=3))


    def get_unit_vectors(self, threshold=0):
        """Gets unit vectors at all points. Threshold is used for fields_diversity"""
        mags = self.get_fields_mag()
        mags[np.abs(mags) <= threshold] = 1
        mags = np.stack((mags, mags, mags), axis=3) #replicates on 4rd axis (3, xyz)
        units = np.divide(self.fields, mags)
        units[np.abs(mags) < threshold] = 1
        return units
        
    def get_fields_diversity(self, threshold=0):
        """
        Gets diversity metric for each point in space. 
        Points below threshold are set to +inf to indicate not high enough magnitude for diversity.
        """
        # First, get unit vectors
        units = self.get_unit_vectors(threshold=threshold)

        # Take inner product of unit vectors:
        inner_product = np.sum(np.prod(units, axis=4),axis=3)
        inner_product = -np.log10(np.abs(inner_product)) # Taking ABS, only for log scale
        
        return inner_product

    def get_fields_diversity2(self):
        """
        John's metric: sum(|Lx|)*sum(|Ly|)*sum(|Lz|)
        """
        return np.prod(np.sum(np.abs(self.fields), axis=4), axis=3)

    def get_fields_diversity3(self):
        """
        Divergence/Gradient method
        """
        # v1
        # self.fields = np.sort(self.fields, axis=4)
        # ret = np.sum(np.gradient(np.gradient(self.fields, axis=(4)), axis=(4)), axis=(4))
        # return np.sum((ret[:,:,:,0], ret[:,:,:,1], -ret[:,:,:,2]))

        # v2
        # self.fields = np.sort(self.fields[:,:,:,(2,1,0),:], axis=4)
        # diff1 = np.gradient(self.fields, axis=(4))
        # diff2 = np.gradient(diff1, axis=(4))
        # ret = np.sum(diff2, axis=(4))
        # # return np.abs(np.sum((-np.abs(ret[:,:,:,0]), np.abs(ret[:,:,:,1]), np.abs(ret[:,:,:,2]))))
        # return np.abs(np.sum((-np.abs(np.sum(diff1[:,:,:,0])), np.abs(np.sum(diff2[:,:,:,1])), np.abs(np.sum(diff2[:,:,:,2])))))

        # v3
        diff_theta = np.sort(self.fields[:,:,:,(1,2,0),:], axis=4)
        diff_theta = np.gradient(diff_theta, axis=(4))
        diff_theta = np.gradient(diff_theta, axis=(4))
        diff_theta = np.sum(diff_theta[:,:,:,0])

        diff_phi = np.sort(self.fields[:,:,:,(2,1,0),:], axis=4)
        diff_phi = np.gradient(diff_phi, axis=(4))
        diff_phi = np.gradient(diff_phi, axis=(4))
        diff_phi = np.sum(diff_phi[:,:,:,0])

        diff_mag = np.sort(self.fields[:,:,:,(0,1,2),:], axis=4)
        diff_mag = np.gradient(diff_mag, axis=(4))  # Only need first derivative here
        diff_mag = np.sum(diff_mag[:,:,:,0])

        return np.abs(np.sum((-0*np.abs(diff_mag), np.abs(diff_theta), np.abs(diff_phi))))

    def get_fields_diversity4(self):
        """
        Ian: mean/variance method
        """
        return

    def get_fields_diversity5(self):
        """
        Ryan idea
        """
        total_tp = np.zeros(np.shape(self.fields)[0:3])
        num_channels = np.shape(self.fields)[4]
        # for i in range(num_channels):
        #     for j in range(num_channels):
        #         for k in range(num_channels):
        #             # total_tp += np.dot(np.cross(self.fields[:,:,:,:,i], self.fields[:,:,:,:,j]), self.fields[:,:,:,:,k])
        #             total_tp = np.add(total_tp, np.abs(np.linalg.det(self.fields[:,:,:,:,(i,j,k)])))

        combos = combinations(range(num_channels), r=3)
        for channels in combos:
            total_tp = np.add(total_tp, np.abs(np.linalg.det(self.fields[:,:,:,:,channels])))

        return total_tp

    def get_fields_diversity9(self):
        """
        Calculate spherical(?) distance to every other point.
        Won't work bc can get high diversity with points along 1 axis.
        """

    def get_fields_diversity10(self):
        """
        Generate n classes for number of vectors.
        Count number that have a point in it.
        Likely inaccurate for small # of vectors.
        """
        bins = np.shape(self.fields)[4]


    def get_fields_diversity11(self):
        """
        Build off of div10, but find distance to class boundary. Add up.
        """

    def get_fields_diversity12(self):
        """
        John's idea: calculate maximum montage (voltage)
        Abs of voltage for each of 3(6?) vectors:
        1,0,0
        0,1,0
        0,0,1
        Only valid for 2 devices
        This ends up being the maximum in each direction.
        """
        shape = np.shape(self.fields)
        if shape[4] != 256:
            raise "Must have 2 devices of 128 channels each."
        # voltage = 0
        # x_dipoles = np.tile(np.array([1,0,0]), (shape[0],shape[1],shape[2],1))
        # y_dipoles = np.tile(np.array([0,1,0]), (shape[0],shape[1],shape[2],1))
        # z_dipoles = np.tile(np.array([0,0,1]), (shape[0],shape[1],shape[2],1))
        #voltage = #multiply montage by x, y, z, sum
        max_x = np.zeros((shape[0], shape[1], shape[2]))
        max_y = np.zeros((shape[0], shape[1], shape[2]))
        max_z = np.zeros((shape[0], shape[1], shape[2]))

        for m1 in range(0,128):
            for m2 in range(128,256):
                montage = np.abs(self.fields[:,:,:,:,m1]-self.fields[:,:,:,:,m2])
                max_x = np.maximum(max_x, montage[:,:,:,0])
                max_y = np.maximum(max_y, montage[:,:,:,1])
                max_z = np.maximum(max_z, montage[:,:,:,2])

                montage = np.abs(self.fields[:,:,:,:,m1]+self.fields[:,:,:,:,m2])
                max_x = np.maximum(max_x, montage[:,:,:,0])
                max_y = np.maximum(max_y, montage[:,:,:,1])
                max_z = np.maximum(max_z, montage[:,:,:,2])
        for m in range(0,256):
            montage = np.abs(self.fields[:,:,:,:,m])
            max_x = np.maximum(max_x, montage[:,:,:,0])
            max_y = np.maximum(max_y, montage[:,:,:,1])
            max_z = np.maximum(max_z, montage[:,:,:,2])

        return np.add(max_x, np.add(max_y, max_z))