import numpy as np
from os import path, listdir
from natsort import natsorted
import cv2
from .field_metrics import FieldEvaluator

class FieldImporter(FieldEvaluator):
    """
    FieldImporter class. Imports G matrix archive files (currently .npz only), and performs operations on them.
    Implements the fieldevaluator class, which 
    """
    def __init__(self, dir_path=None):
        super().__init__()
        self.clear_fields()
        if dir_path:
            self.set_files(dir_path)
        self.device_electrode_count = []

    def set_files(self, dir_path):
        """Sets the lead field top-level directory and the files list"""
        self.leadfield_path = dir_path
        self.leadfield_files = natsorted(listdir(self.leadfield_path))

    def import_header(self, filepath):
        """Imports the header information to save grid details."""
        # use something besides genfromtxt here
        # self.header = np.genfromtxt(filepath, usemask=False, max_rows=2) #gets the first 2 rows
        pass

    # def get_metadata(self):
    #     """Returns metadata generated from headers, devices, and transformations."""
    #     return (self.grid_start, self.grid_stop, self.grid_step)

    # def save(self, archive_file):
    #     """Saves data to numpy archive."""
    #     np.savez_compressed(archive_file, G=self.G, meta=self.get_metadata())

    def load(self, archive_file, clear_fields=True, shape_g_matrix=True):
        """
        Loads data from numpy archive. By default, this overrides fields that were already imported.
        This also shapes the G matrix into voxels by default.
        """
        if clear_fields:
            self.clear_fields()

        # Lead fields will have shape [#Points, (6), #Channels]
        # Axis 1 represents the data of form (x_pos, y_pos, z_pos, x_val, y_val, z_val)
        data = np.load(archive_file, allow_pickle=True)
        self.G = data['G']
        #! This code block has been modified to accomodate a 3x3x3cm cube LF
        # self.grid_start = data['meta'][()]['grid_start']
        # self.grid_stop = data['meta'][()]['grid_stop']
        # self.grid_step = data['meta'][()]['grid_step']

        self.grid_start = [-15,-15,0]
        self.grid_stop = [15,15,30]
        self.grid_step = [0.5,0.5,0.5]

        if shape_g_matrix:
            self.shape_G_matrix()

    def import_fields(self, clear_fields=False, import_first_header=True):
        """
        OBSOLETE

        Imports fields from csv archives into the G matrix.
        """
        if clear_fields:
            self.clear_fields()

        if import_first_header:
            self.import_header(path.join(self.leadfield_path, self.leadfield_files[0]))

        # Lead fields will be of shape [#Points, (6), #Channels]
        # Axis 1 represents the data of form (x_pos, y_pos, z_pos, x_val, y_val, z_val)
        for file in self.leadfield_files:
            data = np.genfromtxt(path.join(self.leadfield_path, file), usemask=False, skip_header=2)
            self.G = np.dstack([self.G,data]) if self.G.size else data

        self.shape_G_matrix()
        
    def shape_G_matrix(self):
        """
        Make G a matrix in coordinate space
        [gridx,gridy,gridz,3,elec,device]
        """
        num_elec = np.shape(self.G)[2]
        self.device_electrode_count.append(num_elec)
        grid_x = round((self.grid_stop[0] - self.grid_start[0]) / self.grid_step[0] + 1)
        grid_y = round((self.grid_stop[1] - self.grid_start[1]) / self.grid_step[1] + 1)
        grid_z = round((self.grid_stop[2] - self.grid_start[2]) / self.grid_step[2] + 1)
        fields_x = np.reshape(self.G[:,3,:], (grid_x, grid_y, grid_z, num_elec))
        fields_y = np.reshape(self.G[:,4,:], (grid_x, grid_y, grid_z, num_elec))
        fields_z = np.reshape(self.G[:,5,:], (grid_x, grid_y, grid_z, num_elec))
        fields = np.stack((fields_x, fields_y, fields_z), axis=3) #x is rows, y is cols

        if self.fields.size:
            self.fields = np.append(self.fields, fields, axis=4) # TODO: way to track which electrodes for which device
        else:
            self.fields = fields

    def duplicate_fields(self, count=2):
        """
        Helper method to duplicate existing lead fields. Useful for duplicating devices of a single type.
        """
        self.fields = np.tile(self.fields, (1,1,1,1,count))

    def translate(self, x=0, y=0, z=0, devices=None, electrodes=None):
        """linear translation of fields"""
        if electrodes and devices:
            raise "Cannot define both electrodes and devices."

        if x > 0:
            if y > 0:
                if z > 0:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((x,0),(y,0),(z,0),(0,0),(0,0)), mode='constant')[:-x,:-y,:-z,:,:]
                else:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((x,0),(y,0),(0,-z),(0,0),(0,0)), mode='constant')[:-x,:-y,-z:,:,:]
            else:
                if z > 0:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((x,0),(0,-y),(z,0),(0,0),(0,0)), mode='constant')[:-x,-y:,:-z,:,:]
                else:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((x,0),(0,-y),(0,-z),(0,0),(0,0)), mode='constant')[:-x,-y:,-z:,:,:]
        else:
            if y > 0:
                if z > 0:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((0,-x),(y,0),(z,0),(0,0),(0,0)), mode='constant')[-x:,:-y,:-z,:,:]
                else:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((0,-x),(y,0),(0,-z),(0,0),(0,0)), mode='constant')[-x:,:-y,-z:,:,:]
            else:
                if z > 0:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((0,-x),(0,-y),(z,0),(0,0),(0,0)), mode='constant')[-x:,-y:,:-z,:,:]
                else:
                    self.fields[:,:,:,:,electrodes] = np.pad(self.fields[:,:,:,:,electrodes], ((0,-x),(0,-y),(0,-z),(0,0),(0,0)), mode='constant')[-x:,-y:,-z:,:,:]

    def rotate(self, theta=None, psi=None, electrodes=None, use_center=False):
        """
        rotates lead fields.

        Note: this does not work well due to interpolation errors.
        TODO: make this work with rotating the dipoles and extracted lead fields instead of the field vectors.
        """
        # First, rotate in XZ plane (psi)
        if psi:
            if use_center:
                image_center = tuple(np.array((self.fields.shape[2]/2, self.fields.shape[0]/2)))
            else:
                image_center = tuple(np.array((0, self.fields.shape[0]/2)))
            rot_mat = cv2.getRotationMatrix2D(image_center, psi, 1.0)
            grid_y = round((self.grid_stop[1] - self.grid_start[1]) / self.grid_step[1] + 1)
            shape = self.fields[:,0,:,0,0].shape[1::-1]

            # rotate field direction
            rot_mat_fld = np.array([[np.cos(psi), 0, np.sin(psi)],
                            [0, 1, 0],
                            [-np.sin(psi), 0, np.cos(psi)]
                        ])
            flds = np.swapaxes(self.fields, 3, 4)
            flds = np.stack([flds], 5)
            flds = np.matmul(rot_mat_fld, flds)
            self.fields = np.swapaxes(flds[:,:,:,:,:,0], 3, 4)

            # rotate positions
            for y in range(grid_y):
                for dir in (0,1,2):
                    for elec in electrodes:
                        self.fields[:,y,:,dir,elec] = cv2.warpAffine(self.fields[:,y,:,dir,elec], rot_mat, shape, flags=cv2.INTER_CUBIC)


        # Now, rotate in XY (theta)
        if theta:
            image_center = tuple(np.array(self.fields.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
            grid_z = round((self.grid_stop[2] - self.grid_start[2]) / self.grid_step[2] + 1)
            shape = self.fields[:,:,0,0,0].shape[1::-1]
            
            for z in range(grid_z):
                for dir in (0,1,2):
                    for elec in electrodes:
                        self.fields[:,:,z,dir,elec] = cv2.warpAffine(self.fields[:,:,z,dir,elec], rot_mat, shape, flags=cv2.INTER_CUBIC)

            # rotate field direction
            rot_mat_fld = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]
                        ])
            flds = np.swapaxes(self.fields, 3, 4)
            flds = np.stack([flds], 5)
            flds = np.matmul(rot_mat_fld, flds)
            self.fields = np.swapaxes(flds[:,:,:,:,:,0], 3, 4)

    def clear_fields(self):
        """
        Clears lead fields in memory by replacing with an empty numpy array.
        """
        self.G = np.array([])
        self.fields = np.array([])