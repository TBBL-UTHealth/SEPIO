# Libraries
import numpy as np
from os import path, listdir
from natsort import natsorted
from scipy.io import savemat

# ANSYS imports
from pyaedt import Maxwell3d, Desktop

class LeadFieldExporter:
    """
    Lead field exporter class.
    On creation, if an ansys project file is specified, then ANSYS will initialize to that project.
    Otherwise, ANSYS will not run. The use case for this is in converting archive types or to view/edit metadata.
    """
    def __init__(self, ansys_project_path=None):
        self.metadata = {}
        self.clear_fields()
        if ansys_project_path:
            self.initialize_ansys(ansys_project_path)

    def initialize_ansys(self, ansys_project_path):
        """
        Helper method to open ANSYS an initialize the Maxwell3D object.
        Usually not called explicitly, but can be if multiple operations are to be done
        in a loop, for example. Note: version must match the installed ANSYS version.
        """
        self.desktop = Desktop(specified_version="2021.1", non_graphical=True, new_desktop_session=True, close_on_exit=True, student_version=False)
        self.m3d = Maxwell3d(projectname=ansys_project_path, close_on_exit=True)
        self.module = self.m3d.get_module("FieldsReporter")

    def export_fields_on_grid(self, channels, grid_start, grid_stop, grid_step, output_dir, 
            export_e=True, export_j=False, export_points=True, analysis="Setup1 : LastAdaptive", 
            parameters=[], coord_system="Cartesian", offset=[0, 0, 0]):
        """
        Export fields on a grid specified by 3 variables: grid start, stop, step.
        
        Channels: ANSYS "channels" parameter to sweep over
        Export E or J fields, only E fields exported by default.
        Parameters: extra params to send to ANSYS to search for solution.
        Coord_system: {cartesian, cylindrical, cylindrical}
        Offset: size-3 vector to offset the grid
        """

        # Add channels to metadata
        self.metadata['channels'] = channels
        self.metadata['grid_start'] = grid_start
        self.metadata['grid_stop'] = grid_stop
        self.metadata['grid_step'] = grid_step
        self.metadata['parameters'] = parameters
        self.metadata['coord_system'] = coord_system
        self.metadata['offset'] = offset

        grid_start = [f'{str(grid_start[0])}mm', f'{str(grid_start[1])}mm', f'{str(grid_start[2])}mm']
        grid_stop = [f'{str(grid_stop[0])}mm', f'{str(grid_stop[1])}mm', f'{str(grid_stop[2])}mm']
        grid_step = [f'{str(grid_step[0])}mm', f'{str(grid_step[1])}mm', f'{str(grid_step[2])}mm']
        offset = [f'{str(offset[0])}mm', f'{str(offset[1])}mm', f'{str(offset[2])}mm']
        
        for channel in channels:
            params = []
            params.extend(parameters)
            params.extend([
                "ch_index:="		, str(channel),
            ])

            if export_e:
                self.module.ClearAllNamedExpr()
                self.module.CopyNamedExprToStack("E_Vector")
                self.module.ExportOnGrid(
                    path.join(output_dir, f"e_field_ch_{str(channel)}.lead"),
                    grid_start,
                    grid_stop,
                    grid_step,
                    analysis, 
                    params,
                    export_points,
                    coord_system,
                    offset,
                    False
                )
            if export_j:
                self.module.ClearAllNamedExpr()
                self.module.CopyNamedExprToStack("J_Vector")
                self.module.ExportOnGrid(
                    path.join(output_dir, f"j_field_ch_{str(channel)}.lead"), 
                    grid_start,
                    grid_stop,
                    grid_step,
                    analysis,
                    params,
                    export_points,
                    coord_system,
                    offset,
                    False
                )

    def export_fields_on_points(self, pts_path, output_dir, channels=None, name=None,
            export_e=True, export_j=False, analysis="Setup1 : LastAdaptive", parameters=[]):
        if channels:
            for channel in channels:
                params = parameters.extend([
                    "ch_index:="		, str(channel),
                ])
                if export_e:
                    self.module.ClearAllNamedExpr()
                    self.module.CopyNamedExprToStack("E_Vector")
                    self.module.ExportToFile(
                        path.join(output_dir, f"e_field_ch_{str(channel)}.lead"), 
                        pts_path, 
                        analysis, 
                        params, 
                        True)
                if export_j:
                    self.module.ClearAllNamedExpr()
                    self.module.CopyNamedExprToStack("J_Vector")
                    self.module.ExportToFile(
                        path.join(output_dir, f"j_field_ch_{str(channel)}.lead"), 
                        pts_path, 
                        analysis, 
                        params, 
                        True)
        else:
            """
            Specifiy a path of points (Nx3) to export fields on.
            """
            for channel in channels:
                params = parameters.extend([
                    "ch_index:="		, str(channel),
                ])
                if export_e:
                        self.module.ClearAllNamedExpr()
                        self.module.CopyNamedExprToStack("E_Vector")
                        self.module.ExportToFile(
                            path.join(output_dir, f"e_field_ch_{name}.lead"), 
                            pts_path, 
                            analysis, 
                            parameters, 
                            True)
                if export_j:
                    self.module.ClearAllNamedExpr()
                    self.module.CopyNamedExprToStack("J_Vector")
                    self.module.ExportToFile(
                        path.join(output_dir, f"j_field_ch_{name}.lead"), 
                        pts_path, 
                        analysis, 
                        parameters, 
                        True)

    def set_files(self, leadfield_dir):
        """Sets the lead field top-level directory and the files list"""
        # TODO: could add multiple directories
        self.leadfield_files = natsorted(listdir(leadfield_dir))

    def import_fields(self, leadfield_dir, skip_header=2):
        """
        Imports fields fields from file directory
        Used to import the fields generated from ANSYS into memory for storage to binary archive file.

        By default, skip_header is 2 for lead fields exported on grid. For lead fields exported at points, use skip_header=1
        """
        # Lead fields will be of shape [#Points, (6), #Channels]
        # Axis 1 represents the data of form (x_pos, y_pos, z_pos, x_val, y_val, z_val)
        self.set_files(leadfield_dir)

        for file in self.leadfield_files:
            data = np.genfromtxt(path.join(leadfield_dir, file), usemask=False, skip_header=skip_header)
            self.G = np.dstack([self.G, data]) if self.G.size else np.reshape(data, (np.shape(data)[0], np.shape(data)[1], 1))

    def get_metadata(self):
        """
        Returns metadata used for export.
        TODO: might need to have some dynamic processing here in the future.
        """
        return self.metadata

    def clear_fields(self):
        """
        Sets the G vector to an empty numpy array.
        Useful if we want to keep a project open and re-export.
        """
        self.G = np.array([])

    def save(self, archive_file, type='numpy'):
        """
        Saves G matrix to an archive file.

        Type: {'numpy', 'matlab'}
        """
        
        # DEBUG STATEMENT -> tentatively staying here
        s= f"Saving metadata to file...\n channels={self.metadata['channels']}\ngrid_start={self.metadata['grid_start']}\ngrid_stop={self.metadata['grid_stop']}\ngrid_step={self.metadata['grid_step']})"
        print(s)

        if type == 'numpy':
            np.savez_compressed(archive_file, G=self.G, meta=self.metadata)
        elif type == 'matlab':
            mdict = {
                'G': self.G,
                'meta': self.metadata
            }
            savemat(archive_file, mdict=mdict)
        else:
            raise 'Error: Incorrect archive type specified.'
    
    def close(self):
        """
        Closes the ANSYS project.
        Note: this is VERY important, and all code paths should result in closing the project at the end.
        Otherwise, ANSYS can reach a state where it is unusable until a PC restart.
        In addition, ANSYS uses a lockfile mechanism so that only one program can access at a time.
        Typical usage is to surround pyaedt calls in a try/catch/finally, where finally always closes.
        """
        self.path = None
        self.leadfield_dir = None
        self.module = None
        self.G = np.array([])
        self.metadata = {}
        self.m3d.close_project()
        self.desktop.close_desktop()