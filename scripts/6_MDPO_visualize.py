### MDPO: Multi-Device Placement Optimization ###
"""MDPO Visualization and Assessment Interface

Purpose:
    Interactive 3D visualization of brain / ROI cortical surface data overlaid with
    simulated sensing metrics (Voltage, SNR, Information Capacity) for user-defined
    invasive device trajectories. Also supports an anatomical angle map (surface
    normal relative to scalp center) independent of device placement.

High-Level Workflow:
    1. Load MRI-derived surface meshes (whole brain and one or more ROIs).
    2. Optionally merge ROI geometry into the brain mesh for centering / visualization.
    3. Recenter geometry (default: brain center of mass) and generate cortical layers:
       - Layer 4 approximation (dipole positions) via partial inflation.
       - Full cortical inflation (outer surface) for optional viewing.
    4. Accept one or more device pose specifications (x, y, z, alpha, beta, gamma).
    5. Transform vertices into each device's local vector space, sample corresponding
       lead field vectors, and compute per-vertex voltage, SNR, and information capacity.
    6. Color-map the chosen attribute using predefined bins and display via Open3D.
    7. Provide optional export of point cloud and reconstructed surface (.obj files).

Primary Script Inputs (file-based):
    - Lead field .npz file (fields_file) containing 5-D array: [X, Y, Z, 3(field vec), E(electrode)].
    - Brain surface .mat file(s) with faces, vertices, normals.
    - ROI surface .mat file(s) with faces, vertices, normals (and optional weighting).

User Interface Inputs:
    - Region selection: ROI-only vs full brain.
    - Inflation toggle: inner cortical surface vs inflated surface for display.
    - Attribute selection: 'Voltage', 'SNR', 'Info.Cap.', or 'Angles'.
    - Color mode: Smooth (muted gradient) vs Contrast (higher differentiation).
    - Device trajectories: Enter manually or load from a pickle of prior optimization.
    - Montage option (difference-based combination across electrodes).
    - OBJ saving toggle for mesh and point cloud export.

Key Computations:
    - Voltage: max absolute electrode response per vertex (or montage spread).
    - SNR: (Voltage / noise)^2 per vertex.
    - Information Capacity: bandwidth * log2(1 + SNR).
    - Angles: angle (degrees) between surface normal and radial position vector.

Performance Notes:
    - Poisson surface reconstruction depth kept at 8 for balance of detail vs speed.
    - Outlier removal (statistical) applied before reconstruction to reduce artifacts.
    - Density filtering removes the lowest ~1% Poisson vertices for cleaner surfaces.

Usage Demo Suggestion:
    Select 'Angles', clear all devices, and toggle inflation / color schemes to explore
    anatomical orientation trends across the cortical surface.

Important Constraints:
    - No functional code is modified by documentation improvements.
    - Coordinate units are millimeters (mm) for spatial data and microvolts (μV) for voltage.
    - Dipole magnitudes assumed constant (magnitude variable) for simulated sensing.

"""
### Imports and Files ###
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import open3d as o3d
from scipy.io import loadmat
from os import path
import sys, os

# Set up paths for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.abspath(current_dir))
grandparent_dir = os.path.dirname(os.path.abspath(parent_dir))
sys.path.insert(0, grandparent_dir)
from modules.leadfield_importer import FieldImporter

### SELECT lead field files
# Currently only supports one device type at a time
device = 'SEEG'  # Device type identifier. Supported examples: {'DISC', 'IMEC', 'SEEG'}
folder = r"...\SEPIO_dataset"  # Root dataset directory (user should update path).
fields_file = path.join(folder,'leadfields', 'SEEG_8e_500um_1500pitch.npz')  # Lead field file.
# Alternative lead field examples:
# fields_file = path.join(folder,'leadfields', 'DISC_30mm_p2-5Sm_MacChr.npz')
# fields_file = path.join(folder,'leadfields', 'ECoG_1mm_500umgrid.npz')

# Save folder; Screenshots not implemented
output = path.join(folder,"outputs")  # Output directory for exported OBJ assets.

# Source space derived from MRI; any number of ROI files in list
# Auditory
roi_data = [loadmat(path.join(folder,'MDPO_data','r_broca.mat'))]  # List of ROI .mat datasets.
brain_data = [loadmat(path.join(folder,'MDPO_data','brain.mat'))]  # List of whole-brain .mat datasets.

brain_name = 'brain'  # Expected struct key inside brain .mat file (e.g., 'ans', 'brain').
roi_name = ['ans']     # List of struct keys for each ROI .mat file.
roi_weights = [1.]     # Per-ROI weighting factors; length must match roi_data.
roi_include = [True]   # Whether to merge each corresponding ROI into brain surface for global view.
do_weights = False     # Toggle: apply roi_weights scaling to computed metrics.
recenter_on_brain = True  # If True, recenter all geometry on brain CoM; else recenter on ROI selection.

# Get fields
field_importer = FieldImporter()
field = field_importer.load(fields_file)  # Loaded lead field data container.
num_electrodes = np.shape(field_importer.fields)[4]  # Electrode dimension length.
fields = field_importer.fields  # 5-D array: [x, y, z, vector(3), electrode].
scale = 0.5  # Spatial resolution (mm per voxel). Lead fields typically at 0.5 mm.
cl_wd = 1.0  # (Currently unused) Local clearance diameter (mm) for device volume masking.

# Define tissues dipoles
# Magnitude is 0.5e-9 (nAm) for real signals or 20e-9 (nAm) for phantom simulations
magnitude = 0.5e-9  # Dipole magnitude (nAm) for physiologic simulation (phantom = 20e-9 nAm).
dipole_offset = 1.25  # Relative depth factor along cortical normal from WM-GM (0) to GM-pial (1).
cortical_thickness = 2.5  # Approximate cortical thickness (mm) used for inflation visualization.
noise = 2.7  # RMS noise (μV). Adjust per device type (e.g., SEEG ~2.7 μV, DiSc ~4.1 μV).

bandwidth = 100  # Sampling bandwidth (samples/s) used for information capacity calculation.

### Place devices and transform vector space for each
# Produces dippos and dipvec for calculating voltages

# Modify for device positions; [x,y,z,alpha,beta,gamma] for the start and end of each device
## uses euler angles, in radians


# Default input position; Right auditory cortex ROI preset
devpos = np.array(
    [[65, -6, -22, 0, 0, 0],]
)

# Clear fields inside device; potentially unnecessary for visualizing
# vr = int(cl_wd//scale) # voxel radius to clear
# mp = fields.shape[0]//2  # field midpoint index
# fields[mp-vr:mp+vr,mp-vr:mp+vr,:,:,:] = np.nan

# define the new origin as the center of the lead field
midpoint = fields.shape[0]//2  # field midpoint index


def obtain_data(data, name):
    """Extract mesh components from a MATLAB-structured surface file.

    Parameters
    ----------
    data : dict
        Dictionary returned by `scipy.io.loadmat` for a brain/ROI surface.
    name : str
        Key inside `data` pointing to region struct: expected shape [1][0] with (faces, vertices, normals).

    Returns
    -------
    header : bytes
        Original MAT-file header metadata.
    faces : np.ndarray, shape (F, 3)
        Triangle face indices.
    vertices : np.ndarray, shape (V, 3)
        XYZ vertex coordinates.
    normals : np.ndarray, shape (V, 3)
        Per-vertex normal vectors.
    """

    # creating corresponding lists in python in place of matlab variables
    header = data['__header__']
    faces = data[name][0][0][0]
    vertices = data[name][0][0][1]
    normals = data[name][0][0][2]

    return header, faces, vertices, normals


### Generate arrays from Brainsuite data; MUST be done before later function definitions
# The following block loads and merges all brain and ROI mesh data into single arrays for later use.
brain_vals = []
brain_vallen = [[],[],[]] # summative index for each start point
brain_count = [0,0,0] # running index sum
roi_vals = []
roi_vallen = [[],[],[]]
roi_count = [0,0,0]

# Load each file and associate with correct weights
for i,fb in enumerate(brain_data):
    _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_data[i], brain_name)
    brain_vals.append([brain_faces,brain_vertices,brain_normals])
    for k in range(3):
        brain_vallen[k].append(brain_vals[i][k].shape[0] + brain_count[k])
        brain_count[k] += brain_vals[i][k].shape[0]
for i,fb in enumerate(roi_data):
    _, roi_faces, roi_vertices, roi_normals = obtain_data(roi_data[i], roi_name[i])
    roi_vals.append([roi_faces,roi_vertices,roi_normals])
    for k in range(3):
        roi_vallen[k].append(roi_vals[i][k].shape[0] + roi_count[k])
        roi_count[k] += roi_vals[i][k].shape[0]

# Initialize final variables
brain_faces = np.empty((brain_vallen[0][-1],3))
brain_vertices = np.empty((brain_vallen[1][-1],3))
brain_normals = np.empty((brain_vallen[2][-1],3))
roi_faces = np.empty((roi_vallen[0][-1],3))
roi_vertices = np.empty((roi_vallen[1][-1],3))
roi_normals = np.empty((roi_vallen[2][-1],3))
roi_vertices_weights = np.zeros((roi_vallen[1][-1],)) # matching index weights

# Merge file variables whole brain
for i in range(len(brain_data)): # i for each file index
    if i == 0: # first file starts from zero in lists
        brain_faces[0:brain_vallen[0][i]] = brain_vals[i][0]
        brain_vertices[0:brain_vallen[1][i]] = brain_vals[i][1]
        brain_normals[0:brain_vallen[2][i]] = brain_vals[i][2]
    else:
        brain_faces[brain_vallen[0][i-1]:brain_vallen[0][i]] = brain_vals[i][0]
        brain_vertices[brain_vallen[1][i-1]:brain_vallen[1][i]] = brain_vals[i][1]
        brain_normals[brain_vallen[2][i-1]:brain_vallen[2][i]] = brain_vals[i][2]

# Merge file variables ROIs
for i in range(len(roi_data)): # i for each roi file
    if i == 0: # first file starts from zero in lists
        roi_faces[0:roi_vallen[0][i]] = roi_vals[i][0]
        roi_vertices[0:roi_vallen[1][i]] = roi_vals[i][1]
        roi_normals[0:roi_vallen[2][i]] = roi_vals[i][2]

        roi_vertices_weights[0:roi_vallen[1][i]] = np.repeat(roi_weights[i],roi_vallen[1][i])
    else:
        roi_faces[roi_vallen[0][i-1]:roi_vallen[0][i]] = roi_vals[i][0]
        roi_vertices[roi_vallen[1][i-1]:roi_vallen[1][i]] = roi_vals[i][1]
        roi_normals[roi_vallen[2][i-1]:roi_vallen[2][i]] = roi_vals[i][2]
        roi_vertices_weights[roi_vallen[1][i-1]:roi_vallen[1][i]] += np.repeat(roi_weights[i],(roi_vallen[1][i] - roi_vallen[1][i-1]))


def recenter(vertices, reference):
    """Translate vertices to a coordinate system centered on a reference set.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Original vertex coordinates to be recentered.
    reference : np.ndarray, shape (M, 3)
        Reference vertex set whose mean defines the new origin.

    Returns
    -------
    vertices_modified : np.ndarray, shape (N, 3)
        Recentered vertex coordinates (reference CoM at (0,0,0)).
    """

    # find midpoint
    center = np.mean(reference, axis = 0)
    avg1 = center[0]
    avg2 = center[1]
    avg3 = center[2]
    print(f"Brain center of mass: ({avg1}, {avg2}, {avg3})") # Prints the ROI centerpoint in MRI xyz
    
    # create a copy of the vertices
    vertices_modified = vertices.copy()

    # modify the data according to the midpoint
    for idx in range(vertices.shape[0]):
        vertices_modified[idx][0] -= avg1
        vertices_modified[idx][1] -= avg2
        vertices_modified[idx][2] -= avg3

    return vertices_modified

# Recenter brain and ROI vertices for visualization and calculation
if recenter_on_brain:
    recentered_brain = recenter(brain_vertices, brain_vertices)
    recentered_roi = recenter(roi_vertices, brain_vertices)
else:
    recentered_brain = recenter(brain_vertices, roi_vertices)
    recentered_roi = recenter(roi_vertices, roi_vertices)

# Optionally add ROIs into whole brain for centering and visualization
if any(roi_include):
    for i in range(len(roi_data)):
        if roi_include[i]:
            if i == 0:
                vec_add = recentered_roi[:roi_vallen[1][i]]
                norm_add = roi_normals[:roi_vallen[2][i]]
            else:
                vec_add = recentered_roi[roi_vallen[1][i-1]:roi_vallen[1][i]]
                norm_add = roi_normals[roi_vallen[2][i-1]:roi_vallen[2][i]]
            tempvec = np.empty((recentered_brain.shape[0]+vec_add.shape[0], 3))
            tempvec[:recentered_brain.shape[0]] = recentered_brain
            tempvec[recentered_brain.shape[0]:] = vec_add
            recentered_brain = tempvec
            tempnorm = np.empty((brain_normals.shape[0]+norm_add.shape[0], 3))
            tempnorm[:brain_normals.shape[0]] = brain_normals
            tempnorm[brain_normals.shape[0]:] = norm_add
            brain_normals = tempnorm
    del tempvec
    del tempnorm

def inflate_cortex(vertices,normals,dist):
    """Generate an inflated cortical surface by displacing vertices along normals.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Base cortical vertex coordinates.
    normals : np.ndarray, shape (N, 3)
        Per-vertex normal vectors.
    dist : float
        Distance (mm) to displace each vertex outward along its normal.

    Returns
    -------
    inflated : np.ndarray, shape (N, 3)
        Inflated vertex coordinates. Vertices with invalid normals become NaN rows.
    Notes
    -----
    - Handles potential zero/invalid normals with defensive try/except.
    - Requires `vertices.shape == normals.shape` for standard operation.
    """
    # Copy and move vertices by cortical thickness along normal vectors
    inflated = np.copy(vertices)
    if vertices.shape == normals.shape:
        for i in range(inflated.shape[0]):
            if np.nan in normals[i]: # If any nans in verts
                inflated[i] = np.array([np.nan,np.nan,np.nan])
            else:
                try: # Normal case
                    inflated[i] += normals[i]*dist/np.linalg.norm(normals[i])
                except: # In case of issue (zeros??)
                    inflated[i] = np.array([np.nan,np.nan,np.nan])
    else:
        print("ERROR: Shape mismatch,  no processes completed.")

    return inflated

# Generate points at layer 4 for calculation and inflated cortex for visualization
layer4_brain = inflate_cortex(recentered_brain,brain_normals,dipole_offset)
layer4_roi = inflate_cortex(recentered_roi,roi_normals,dipole_offset)
inflated_brain = inflate_cortex(recentered_brain,brain_normals,cortical_thickness)
inflated_roi = inflate_cortex(recentered_roi,roi_normals,cortical_thickness)


def transform_vectorspace(vertices, normals, devpos):
    """Transform cortical vertices and normals into a device-centered coordinate frame.

    Applies translation (device origin) and Euler rotations (alpha=yaw, beta=pitch,
    gamma=roll) to facilitate indexing into the lead field volume.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Recentered cortical layer vertex positions (mm).
    normals : np.ndarray, shape (N, 3)
        Corresponding per-vertex normal vectors.
    devpos : array-like, shape (6,)
        Device pose [x, y, z, alpha(rad), beta(rad), gamma(rad)]. Position in mm, angles in radians.

    Returns
    -------
    dippos : np.ndarray, shape (N, 3)
        Integer voxel indices (float cast) mapped into lead field space.
    dipvec : np.ndarray, shape (N, 3)
        Rotated and magnitude-scaled dipole vectors (nAm).
    rot_mat : np.ndarray, shape (3, 3)
        Combined rotation matrix (yaw * pitch * roll).
    Notes
    -----
    - Lead field grid assumed centered; X/Y shifted by half-size, Z positive downward.
    - Spatial scaling by `1/scale` converts mm coordinates to voxel indices.
    """

    # initialize variables (theta and phi)
    alpha = devpos[3]
    beta = devpos[4]
    gamma = devpos[5]
    
    # Rotation matrices for yaw, pitch, roll
    yaw = np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
    pitch = np.array([[np.cos(beta), 0, np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
    roll = np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
    rot_mat = np.matmul(yaw, pitch)
    rot_mat = np.matmul(rot_mat, roll)

    dippos = np.copy(vertices)
    dipvec = np.copy(normals)

    # translation
    dippos-=devpos[:3]
    dippos[0] += fields.shape[0]//2
    dippos[1] += fields.shape[1]//2

    # rotate vertices and normals based on the placement of the device
    for idx in range(vertices.shape[0]):
        dippos[idx] = np.matmul(dippos[idx], rot_mat)
        dipvec[idx] = np.matmul(dipvec[idx], rot_mat)

    # Transfer position into leadfield space
    dippos *= 1/scale

    # to make it compatible with the lead field data
    dippos = (dippos.astype('int')).astype('float')
    
    # Scale to dipole nAm magnitude, see value above
    dipvec *= magnitude 

    return dippos, dipvec, rot_mat

def trim_data(leadfield, vertices, normals):
    """Mask out-of-bounds dipole positions relative to lead field extents.

    Parameters
    ----------
    leadfield : np.ndarray
        Lead field volume with dimensions [X, Y, Z, 3, E].
    vertices : np.ndarray, shape (N, 3)
        Candidate dipole voxel indices (float, already transformed).
    normals : np.ndarray, shape (N, 3)
        Dipole moment vectors (scaled normals).

    Returns
    -------
    dippos : np.ndarray, shape (N, 3)
        Dipole positions with invalid entries set to NaN.
    dipvec : np.ndarray, shape (N, 3)
        Unchanged dipole vectors (normal components) for valid positions.
    Notes
    -----
    - Bounds check assumes cubic X/Y and double-depth Z dimension conventions.
    """
    dippos = np.copy(vertices)
    dipvec = np.copy(normals)
    len_half = leadfield.shape[0]//2

    for idx in range(vertices.shape[0]):
        if((np.abs(vertices[idx][0])>len_half) or (np.abs(vertices[idx][1])>len_half) or (vertices[idx][2])>len_half*2) or (vertices[idx][2]<0):
            dippos[idx] = np.nan

    return dippos, dipvec

def calculate_voltage(fields, vertices, normals, vscale=10**6, montage = False, inter = False):
    """Compute per-vertex sensing metrics (Voltage, SNR, Information Capacity).

    Parameters
    ----------
    fields : np.ndarray
        Lead field array [X, Y, Z, 3, E] containing electric field vectors per electrode.
    vertices : np.ndarray, shape (N, 3)
        Dipole voxel indices (NaN where invalid/out of bounds).
    normals : np.ndarray, shape (N, 3)
        Dipole vectors (nAm) corresponding to each vertex.
    vscale : float, default 1e6
        Scaling factor from Volts to microvolts (μV).
    montage : bool, default False
        If True, compute bipolar montage spread (max - min) instead of max |voltage|.
    inter : bool, default False
        Placeholder for future inter-electrode montage logic (currently unused).

    Returns
    -------
    opt_volt : np.ndarray, shape (N,)
        Optimal voltage per vertex (μV) based on selection rule.
    snr_list : np.ndarray, shape (N,)
        Signal-to-noise ratio per vertex (dimensionless, power ratio).
    info_cap : np.ndarray, shape (N,)
        Information capacity estimate (bits per second).
    Notes
    -----
    - NaN propagation used to exclude out-of-bounds or invalid positions.
    - SNR uses (Voltage / noise)^2 formulation.
    - Information capacity uses Shannon-like approximation bandwidth*log2(1+SNR).
    """
    global do_weights

    # make copies of data
    dippos = np.copy(vertices)
    dipvec = np.copy(normals)

    # create a list of field vectors applicable to the surface, [vertex index, electrode #]
    voltage = np.empty((dippos.shape[0], fields.shape[-1]))
    voltage[:]=np.nan
    opt_volt = np.empty(dippos.shape[0])
    opt_volt[:]=np.nan
    
    # calculate voltage for each vertex
    for idx in range(dippos.shape[0]):
        for e in range(fields.shape[-1]): # for every electrode
            if not np.any(np.isnan(dippos[idx])): # if not nan, calculate and modify entry
                try:
                    lead_field = fields[int(dippos[idx][0]+fields.shape[0]//2), int(dippos[idx][1]+fields.shape[0]//2), int(dippos[idx][2]), :, e]
                    voltage[idx, e] = np.dot(lead_field, dipvec[idx])
                except:
                    voltage[idx, e] = np.nan # outside the LF, fill NaN
        if (not np.all(np.isnan(voltage[idx,:]))) and (not montage): 
            opt_volt[idx] = np.nanmax(np.abs(np.copy(voltage[idx]))) # get the optimal voltage across all electrodes
        elif (not np.all(np.isnan(voltage[idx,:]))) and montage and (not inter):
            opt_volt[idx] = np.nanmax(voltage[idx]) - np.nanmin(voltage[idx])
        # elif (not np.all(np.isnan(voltage[idx,:]))) and montage and inter:

    opt_volt *= vscale #scale it to get the list of voltages
    snr_list = np.square(np.copy(opt_volt)/noise) # get the list of snr values
    info_cap = bandwidth*np.log2(1+snr_list) # get the list of information capacity

    # Scale with given weights for ROI subregions
    if do_weights:
        opt_volt = np.multiply(opt_volt,roi_vertices_weights)
        snr_list = np.multiply(snr_list,roi_vertices_weights)
        info_cap = np.multiply(info_cap,roi_vertices_weights)

    return (opt_volt, snr_list, info_cap)

def calculate_angle(brainvert,brainvec):
    """Calculate angle (degrees) between each surface normal and radial position vector.

    Parameters
    ----------
    brainvert : np.ndarray, shape (N, 3)
        Recentered brain or ROI vertex coordinates.
    brainvec : np.ndarray, shape (N, 3)
        Corresponding per-vertex normal vectors.

    Returns
    -------
    angles : np.ndarray, shape (N,)
        Absolute angle in degrees for each vertex; NaN handling deferred upstream.
    Notes
    -----
    - Uses arccos of normalized dot product; values mapped to [0, 180].
    - Radial reference is the vertex position vector from origin (center of mass).
    """
    angles = np.zeros((brainvert.shape[0]))

    for i in range(angles.shape[0]): # i for each vertex in order
        a1 = np.copy(brainvert[i])
        a2 = np.copy(brainvec[i])
        dotprod = np.dot(a1,a2)
        cos_angle = dotprod/(np.linalg.norm(a1)*np.linalg.norm(a2))
        rad_angle = np.arccos(cos_angle)
        angles[i] = rad_angle*180/np.pi

    angles = np.abs(angles)

    return angles


# Color definitions for visualization
color1, color2, color3, color4, color5, color6, color7 = '#6e6e6e', '#A394CC', '#9779EB', '#BB70FF', '#FF70EC', '#FE3023', '#AE0000' # Reds?
capcol1, capcol2, capcol3, capcol4, capcol5, capcol6, capcol7 = '#6e6e6e', '#CFEFD8', '#9DE4B1', '#73D58F', '#12DFC2', '#00A8E2', '#0839ED' # Blues?
angcol1, angcol2, angcol3, angcol4, angcol5, angcol6, angcol7 = '#8C8C8C','#C23939', '#B6C238', '#C26138', '#38C256', '#38C0C2', '#3858C2' # Contrast rainbow
mangcol1, mangcol2, mangcol3, mangcol4, mangcol5, mangcol6, mangcol7 = '#8C8C8C','#C14343', '#C27A44', '#C2AD44', '#87B640', '#44C29C', '#446AC2' # Muted rainbow
colors = [[color1, color2, color3, color4, color5, color6, color7],
          [capcol1, capcol2, capcol3, capcol4, capcol5, capcol6, capcol7],
          [angcol1, angcol2, angcol3, angcol4, angcol5, angcol6, angcol7],
          [mangcol1, mangcol2, mangcol3, mangcol4, mangcol5, mangcol6, mangcol7]]


# Predefined voltage and attribute bins for color mapping
val1, val2, val3, val4, val5, val6 = 0.77, 1.1, 1.66, 2.62, 4.76, 11.9 # IC 15, 30, 60, 120, 240, (480 unused) for 2.3 uV RMS noise

def infocap(val):
    """Helper to compute simplified information capacity bin value.

    Parameters
    ----------
    val : float
        Voltage-like scalar (μV) or proxy quantity.

    Returns
    -------
    int
        Rounded information capacity approximation (bps) for visualization binning.
    Notes
    -----
    - Uses global `bandwidth` and `noise` definitions.
    - Intended for bin labeling, not analytical precision.
    """
    return round(bandwidth*np.log2(1+val/noise))

n_bin = [ # Voltage bins
        [np.nan, val1, val2, val3, val4, val5, val6],
         [np.nan,15,30,60,120,240,480], # Info cap bins
         [np.nan,np.square(val1/noise),np.square(val2/noise),np.square(val3/noise),
          np.square(val4/noise),np.square(val5/noise),np.square(val6/noise)], # SNR bins
         [np.nan,30,60,90,120,150,180] # Angle bins
         ] 

def hextorgb(hex):
    """Convert a hex color string to an (R, G, B) tuple normalized to [0, 1].

    Parameters
    ----------
    hex : str
        Hex color code, e.g. '#AABBCC'. Leading '#' optional.

    Returns
    -------
    tuple(float, float, float)
        Normalized RGB components.
    """
    hex = str(hex).lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16)/255 for i in (0, 2, 4))
    return rgb

# Convert hex to RGB for Open3D
rgb = []
for idx1 in range(len(colors)):
    rgb.append([])
    for idx2 in range(len(colors[idx1])):
        rgb[idx1].append(hextorgb(colors[idx1][idx2]))

# Initial values; start with smoothed color
set_color = colors[3]
set_rgb = rgb[3]

def get_color_list(data, attri = "Voltage", color = set_color, rgb0 = set_rgb):
    """Map numeric data values to categorical color bins for visualization.

    Parameters
    ----------
    data : np.ndarray, shape (N,)
        Scalar metric values (Voltage, SNR, Info.Cap., or Angles).
    attri : str, default 'Voltage'
        Attribute key selecting bin thresholds: {'Voltage','SNR','Info.Cap.','Angles'}.
    color : list[str]
        List of hex color codes for the chosen palette.
    rgb0 : list[tuple]
        Corresponding normalized RGB tuples.

    Returns
    -------
    color_list : list[str]
        Hex color per data point (NaN mapped to first entry).
    rgblist : list[tuple(float,float,float)]
        RGB color per data point suitable for Open3D.
    Notes
    -----
    - Bin thresholds sourced from global `n_bin` definition.
    - Angle mapping reverses order to emphasize anatomical orientation gradient.
    """
    global n_bin
    data = np.abs(np.copy(data))

    flip = 1
    if attri == "Info.Cap.":
        bin = n_bin[1]
    elif attri == "Voltage":
        bin = n_bin[0]
    elif attri == "SNR":
        bin = n_bin[2]
    elif attri == "Angles":
        bin = n_bin[3]
        flip = -1
    else:
        print("ERROR: Incorrect attribute.")

    # set the list of colors
    color_list = []
    rgblist = []
    for idx in range(data.shape[0]):
        if np.isnan(data[idx]):
            color_list.append(color[0])
            rgblist.append(rgb0[0])
        elif data[idx]<=bin[1]:
            color_list.append(color[flip*-1])
            rgblist.append(rgb0[flip*-1])
        elif bin[1]<data[idx]<=bin[2]:
            color_list.append(color[flip*-2])
            rgblist.append(rgb0[flip*-2])
        elif bin[2]<data[idx]<=bin[3]:
            color_list.append(color[flip*-3])
            rgblist.append(rgb0[flip*-3])
        elif bin[3]<data[idx]<=bin[4]:
            color_list.append(color[flip*-4])
            rgblist.append(rgb0[flip*-4])
        elif bin[4]<data[idx]<=bin[5]:
            color_list.append(color[flip*-5])
            rgblist.append(rgb0[flip*-5])
        else:
            color_list.append(color[flip*-6])
            rgblist.append(rgb0[flip*-6])

    return color_list, rgblist

def legend(attribute):
    """Render a matplotlib legend window showing color bin ranges for selected attribute.

    Parameters
    ----------
    attribute : str
        One of {'Voltage','SNR','Info.Cap.','Angles'} determining bin labels and units.

    Returns
    -------
    None
        Displays a blocking matplotlib window; user must close to proceed.
    Notes
    -----
    - Open3D does not natively provide legends; this compensates with static patches.
    - Numerical bins rounded for clearer reading (decimals context-dependent).
    """
    global colors, n_bin, set_rgb,set_color

    # Creating a figure and 3D subplot
    fig = plt.figure()

    color = set_color
    rgb0 = set_rgb
    decimals = 2 # Number of default display decimals

    # Grab bins for automation
    if attribute == "Info.Cap.":
        decimals = 1
        bin = n_bin[1]
    elif attribute == "Voltage":
        bin = n_bin[0]
    elif attribute == "SNR":
        bin = n_bin[2]
    elif attribute == "Angles":
        bin = n_bin[3]
    else:
        print("ERROR: Incorrect attribute.")

    for i in range(len(bin)):
        bin[i] = round(bin[i],decimals)

    # The following block creates the legend for each attribute type
    if attribute=="Info.Cap.":
        # Creating legend with color box 
        range1 = mpatches.Patch(color=color[0], label='nan') 
        range2 = mpatches.Patch(color=color[6], label='0-'+str(bin[1])+' bps') 
        range3 = mpatches.Patch(color=color[5], label=str(bin[1])+'-'+str(bin[2])+' bps') 
        range4 = mpatches.Patch(color=color[4], label=str(bin[2])+'-'+str(bin[3])+' bps') 
        range5 = mpatches.Patch(color=color[3], label=str(bin[3])+'-'+str(bin[4])+' bps') 
        range6 = mpatches.Patch(color=color[2], label=str(bin[4])+'-'+str(bin[5])+' bps')
        range7 = mpatches.Patch(color=color[1], label='>'+str(bin[5])+' bps')
        plt.text(0.02,0.37,s="this is the legend, save it and close the window to see map")
        plt.legend(handles=[range1, range2, range3, range4, range5, range6, range7]) 
    elif attribute=="SNR":
        # Creating legend with color box 
        range1 = mpatches.Patch(color=color[0], label='NaN')
        range2 = mpatches.Patch(color=color[6], label='0-'+str(bin[1])) 
        range3 = mpatches.Patch(color=color[5], label=str(bin[1])+'-'+str(bin[2])) 
        range4 = mpatches.Patch(color=color[4], label=str(bin[2])+'-'+str(bin[3])) 
        range5 = mpatches.Patch(color=color[3], label=str(bin[3])+'-'+str(bin[4])) 
        range6 = mpatches.Patch(color=color[2], label=str(bin[4])+'-'+str(bin[5]))
        range7 = mpatches.Patch(color=color[1], label='>'+str(bin[5]))
        plt.text(0.02,0.37,s="this is the legend, save it and close the window to see map")
        plt.legend(handles=[range1, range2, range3, range4, range5, range6, range7]) 
    elif attribute=="Voltage":
        range1 = mpatches.Patch(color=color[0], label='NaN') 
        range2 = mpatches.Patch(color=color[6], label='0-'+str(bin[1])+' μV') 
        range3 = mpatches.Patch(color=color[5], label=str(bin[1])+'-'+str(bin[2])+' μV') 
        range4 = mpatches.Patch(color=color[4], label=str(bin[2])+'-'+str(bin[3])+' μV') 
        range5 = mpatches.Patch(color=color[3], label=str(bin[3])+'-'+str(bin[4])+' μV') 
        range6 = mpatches.Patch(color=color[2], label=str(bin[4])+'-'+str(bin[5])+' μV')
        range7 = mpatches.Patch(color=color[1], label='>'+str(bin[5])+' μV')
        plt.text(0.02,0.37,s="this is the legend, save it and close the window to see map")
        plt.legend(handles=[range1, range2, range3, range4, range5, range6, range7])
    elif attribute=="Angles":
        range1 = mpatches.Patch(color=color[0], label='NaN')
        range2 = mpatches.Patch(color=color[1], label='0°-'+str(bin[1])+'°') 
        range3 = mpatches.Patch(color=color[2], label=str(bin[1])+'°-'+str(bin[2])+'°') 
        range4 = mpatches.Patch(color=color[3], label=str(bin[2])+'°-'+str(bin[3])+'°') 
        range5 = mpatches.Patch(color=color[4], label=str(bin[3])+'°-'+str(bin[4])+'°') 
        range6 = mpatches.Patch(color=color[5], label=str(bin[4])+'°-'+str(bin[5])+'°')
        range7 = mpatches.Patch(color=color[6], label=str(bin[5])+'°-'+str(bin[6])+'°')
        plt.text(0.02,0.37,s="this is the legend, save it and close the window to see map")
        plt.legend(handles=[range1, range2, range3, range4, range5, range6, range7])
    else:
        print("ERROR: Incorrect attribute for legend.")
    

    # Display the plot
    plt.show()

### USER INTERFACE

# Initialize UI window
root = tk.Tk()
root.geometry("600x600")
root.title("Trajectory and Sensor Optimization")

# Preset values for plotting
vertices = layer4_roi
infl_vertices = inflated_roi
inner_vertices = recentered_roi
normals = roi_normals

# define a function to assist assigning values to variables
def change_vertices(button_val):
    """
    Switches between ROI and full brain for visualization.
    """
    global vertices, normals, infl_vertices, inner_vertices
    if button_val == "ROI":
        vertices = layer4_roi
        infl_vertices = inflated_roi
        inner_vertices = recentered_roi
        normals = roi_normals
    elif button_val == "brain":
        vertices = layer4_brain
        infl_vertices = inflated_brain
        inner_vertices = recentered_brain
        normals = brain_normals

# Function to assist color selection
def change_color(color_button):
    """
    Switches between color schemes for visualization.
    """
    global set_color, set_rgb
    if color_button == "Smooth":
        set_color = colors[3]
        set_rgb = rgb[3]
    elif color_button == "Contrast":
        set_color = colors[2]
        set_rgb = rgb[2]

# Function to load pickle data and update variables for SEPIO results and trajectories
loaded_pickle = None
def load_pickle_file():
    """
    Loads a pickle file containing device positions or optimization results.
    """
    global loaded_pickle
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    if file_path:
        try:
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)
                # Process the loaded data
                messagebox.showinfo("Data loaded!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

### ROI/Brain selection
# select the region of interest
label = tk.Label(root, text = "Select region", font = ('Futura', 12))
label.pack(pady=(10,0))
# set up for the two buttons
choice_frame = tk.Frame(root)
choice_frame.columnconfigure(0, weight=1)
# choice 1 - ROI
ROI_choice = tk.Button(choice_frame, text = "ROI", font = ('Futura', 12), command= lambda: change_vertices("ROI"))
ROI_choice.grid(row=0, column=0)
# choice 2 - Full Brain
brain_choice = tk.Button(choice_frame, text = "Full Brain", font = ('Futura', 12), command= lambda: change_vertices("brain"))
brain_choice.grid(row=0, column=1)
choice_frame.pack()

### Cortex inflation selector
if_inflate = tk.IntVar()
inflate = tk.Checkbutton(root, text = "Inflated cortex map",
                        variable = if_inflate,
                        onvalue = True,
                        offvalue = False,
                        height = 2,
                        width = 20,
                        font=('Futura', 12))
inflate.pack()

### Plotting variable selector
choices = ['Voltage', 'SNR', 'Info.Cap.','Angles']
variable = tk.StringVar(root)
variable.set('Voltage')
attri = tk.OptionMenu(root, variable, *choices)
attri.config(font = ("Futura", 12))
attri.pack()

### Color selection
# Title
c_label = tk.Label(root,text="Color mode",font=('Futura',12))
c_label.pack(pady=(20,0))
# Button frames
c_frame = tk.Frame(root)
c_frame.columnconfigure(0,weight=1)
# Button 1 - Smooth
c1_button = tk.Button(c_frame,text="Smooth",font=('Futura',12),command=lambda:change_color("Smooth"))
c1_button.grid(row=0,column=0)
# Button 2 - Contrast
c2_button = tk.Button(c_frame,text="Contrast",font=('Futura',12),command=lambda:change_color("Contrast"))
c2_button.grid(row=0,column=1)
c_frame.pack()

### Selecting/loading devices
load_label = tk.Label(root, text = "Enter or load values from optimization", font=('Futura', 12))
load_label.pack(pady=(20,0))

# Load Pickle Button
load_button = ttk.Button(root, text="Load Pickle File", command=load_pickle_file)
load_button.pack(pady=5)

# specify device position
device_position = tk.Frame(root, pady = 10)
device_position.columnconfigure(2, weight=1)
device_pos = []
# xyz pos
# x
xlabel = tk.Label(device_position, text="x position", font = ('Futura', 12))
xlabel.grid(row=0, column=0)
xpos = tk.Entry(device_position, font=("Futura", 12))
xpos.grid(row=0, column=1)
xpos.insert(0, 0)
# y
ylabel = tk.Label(device_position, text="y position", font = ('Futura', 12))
ylabel.grid(row=1, column=0)
ypos = tk.Entry(device_position, font=("Futura", 12))
ypos.grid(row=1, column=1)
ypos.insert(0, 0)
# z
zlabel = tk.Label(device_position, text="z position", font = ('Futura', 12))
zlabel.grid(row=2, column=0)
zpos = tk.Entry(device_position, font=("Futura", 12))
zpos.grid(row=2, column=1)
zpos.insert(0, 0)
# angles
# alpha
alpha = tk.Label(device_position, text="alpha", font = ('Futura', 12))
alpha.grid(row=0, column=2)
getalpha = tk.Entry(device_position, font=("Futura", 12))
getalpha.grid(row=0, column=3)
getalpha.insert(0, 0)
# beta
beta = tk.Label(device_position, text="beta", font = ('Futura', 12))
beta.grid(row=1, column=2)
getbeta = tk.Entry(device_position, font=("Futura", 12))
getbeta.grid(row=1, column=3)
getbeta.insert(0, 0)
# gamma
gamma = tk.Label(device_position, text="gamma", font = ('Futura', 12))
gamma.grid(row=2, column=2)
getgamma = tk.Entry(device_position, font=("Futura", 12))
getgamma.grid(row=2, column=3)
getgamma.insert(0, 180)
device_position.pack()

### Show loaded devices
current_devices = tk.Label(root, text="Trajectories: "+str(device_pos), font=('Futura', 12))
current_devices.pack()

# set up for the two buttons
adding = tk.Frame(root)
choice_frame.columnconfigure(0, weight=1)

# Function to plot the output after all inputs so that variables appear in order
def plot_device(vertices, normals, devices, leadfield, attribute, mont = False, inte = False, inflate=False):
    """Generate 3D visualization of sensing metrics or angle map with device geometry.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates (layer 4 or ROI/brain selection).
    normals : np.ndarray, shape (N, 3)
        Per-vertex normal vectors used for dipole orientation.
    devices : np.ndarray, shape (D, 6)
        Device pose specifications. If empty and attribute=='Angles', angle map only.
    leadfield : np.ndarray
        Lead field volume [X, Y, Z, 3, E].
    attribute : str
        Metric to visualize: 'Voltage', 'SNR', 'Info.Cap.', or 'Angles'.
    mont : bool, default False
        Apply montage (spread) computation instead of per-electrode max magnitude.
    inte : bool, default False
        Placeholder flag for future interactive montage extensions.
    inflate : bool, default False
        If True, use inflated cortical surface; else inner surface for point cloud.

    Returns
    -------
    None
        Opens an Open3D window with colored surface and device meshes.
    Notes
    -----
    - Performs Poisson reconstruction on filtered point cloud for smoother surface display.
    - Exports OBJ assets when `save_obj_var` is set.
    - Angles attribute bypasses dipole & lead field computation.
    """
    global set_color, set_rgb, infl_vertices, inner_vertices

    all_dev_volt = np.empty([vertices.shape[0], len(devices)])
    geometry = []

    # If no devices, add empty array spot to allow angle display
    is_dev = True
    if len(devices) == 0:
        is_dev = False
        devices = np.empty((1,))
    # Perform the procedure of calculating voltage for each device
    for idx in range(devices.shape[0]):
        if is_dev:
            dippos, dipvec, rot_mat = transform_vectorspace(vertices, normals, devices[idx])
            dippos_adj, dipvec_adj = trim_data(leadfield, dippos, dipvec)
        # Options: ['Voltage', 'SNR', 'Info.Cap.','Angles']
        if attribute == "SNR":
            all_dev_volt[:, idx] = calculate_voltage(leadfield, dippos_adj, dipvec_adj, montage = mont, inter = inte)[1]
        elif attribute == "Info.Cap.":
            all_dev_volt[:, idx] = calculate_voltage(leadfield, dippos_adj, dipvec_adj, montage = mont, inter = inte)[2]
        elif attribute == "Voltage":
            all_dev_volt[:, idx] = calculate_voltage(leadfield, dippos_adj, dipvec_adj, montage = mont, inter = inte)[0]
        elif attribute == "Angles": # Calculate angles relative to the skull
            all_dev_volt = calculate_angle(layer4_brain,brain_normals)
        else:
            print("ERROR: Incorrect attribute for calculation.")

        if is_dev:
        # for visualizing devices
            dev_end_length = 20.0
            dev_end = o3d.geometry.TriangleMesh.create_cylinder(radius=1,height=dev_end_length)
            dev_end = dev_end.translate((0, 0, -dev_end_length/2))
            dev_end = dev_end.rotate(rot_mat, center=(0, 0, 0))
            dev_end = dev_end.translate((devices[idx][0],devices[idx][1],devices[idx][2]))
            dev_end.compute_vertex_normals()
            end_color = (235/255, 103/255, 52/255)
            dev_end.paint_uniform_color(end_color)
            geometry.append(dev_end)

            dev_front_length = 10.0
            dev_front = o3d.geometry.TriangleMesh.create_cylinder(radius=1,height=dev_front_length)
            dev_front = dev_front.translate((0, 0, dev_front_length/2))
            dev_front = dev_front.rotate(rot_mat, center=(0, 0, 0))
            dev_front = dev_front.translate((devices[idx][0],devices[idx][1],devices[idx][2]))
            dev_front.compute_vertex_normals()
            front_color = (235/255, 195/255, 52/255)
            dev_front.paint_uniform_color(front_color)
            geometry.append(dev_front)

    # For each vertex, still take the greatest signal across all sensors
    if (devices.shape[0]==1) or (attribute == "Angles"):
        voltage = all_dev_volt
    else:
        voltage = np.nanmax(all_dev_volt, axis=1)

    roi_color_list, roi_rgb = get_color_list(voltage,attri = attribute, color = set_color, rgb0 = set_rgb)

    # Draw brain or ROI with pointcloud
    roi = o3d.geometry.PointCloud()
    # Use inflate bool to determine plot vertices and radii
    if inflate: # Use inflated points
        roi.points = o3d.utility.Vector3dVector(infl_vertices)
    else: # Use inner cortical map for viewability
        roi.points = o3d.utility.Vector3dVector(inner_vertices)
    roi.colors = o3d.utility.Vector3dVector(roi_rgb)
    roi.normals = o3d.utility.Vector3dVector(normals)

    # Try removing outliers
    cl, ind = roi.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    roi = roi.select_by_index(ind)

    # Prior method for meshing by ball pivot
    #radii = [0.75,0.75,0.75,0.75] # [1,1,1,1] for inner map, [1.25, 1.25, 1.25, 1.25] for outer?
    #rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #    roi, o3d.utility.DoubleVector(radii))
    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
       roi,depth=8,scale=1.1,linear_fit=False,n_threads=-1)

    # Try density filtering
    low_density_vertices = densities < np.quantile(densities,0.01)
    rec_mesh.remove_vertices_by_mask(low_density_vertices)

    reference = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=20)
    
    # --- OBJ EXPORT ---
    if save_obj_var.get():
        import os
        from datetime import datetime
        if not os.path.exists(output):
            os.makedirs(output)
        # Get current date and time for filename suffix
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M")
        # Save reconstructed mesh
        mesh_path = os.path.join(output, f"surface_mesh_{date_str}.obj")
        o3d.io.write_triangle_mesh(mesh_path, rec_mesh)
        # Save point cloud as OBJ
        points_path = os.path.join(output, f"surface_points_{date_str}.obj")
        o3d.io.write_point_cloud(points_path, roi)

        # Save mesh with devices
        device_meshes = [g for g in geometry if isinstance(g, o3d.geometry.TriangleMesh) and g is not rec_mesh]
        if device_meshes:
            # Merge all device meshes with rec_mesh
            merged_mesh = rec_mesh
            for dev_mesh in device_meshes:
                merged_mesh += dev_mesh
            mesh_with_dev_path = os.path.join(output, f"surface_mesh_with_devices_{date_str}.obj")
            o3d.io.write_triangle_mesh(mesh_with_dev_path, merged_mesh)
    # --- END OBJ EXPORT ---

    geometry.append(roi) # Displays original point cloud
    geometry.append(rec_mesh) # Displays reconstructed surface
    # geometry.append(reference)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for obj in geometry:
        viewer.add_geometry(obj)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True # Option to display xyz coordinate frame
    viewer.run()
    viewer.destroy_window()

# clear all device positions
def reset_dev():
    """Clear all accumulated device positions (in-place list mutation)."""
    global device_pos
    device_pos=[]


def combine_funcs(*funcs): 
    """Compose multiple no-argument callbacks into a single function for UI binding.

    Parameters
    ----------
    *funcs : callable
        Arbitrary sequence of functions to execute serially.

    Returns
    -------
    inner_combined_func : callable
        Function executing all provided callables in order.
    """
    def inner_combined_func(*args, **kwargs): 
        for f in funcs: 
  
            # Calling functions with arguments, if any 
            f(*args, **kwargs) 
  
    # returning the reference of inner_combined_func 
    # this reference will have the called result of all 
    # the functions that are passed to the combined_funcs 
    return inner_combined_func 

# change the device location display
def change_dev_location():
    """Refresh UI label to reflect current `device_pos` contents."""
    change = "Current Location: "+str(device_pos)
    current_devices.configure(text=change)

# delete all previously defined device locations
resetting = tk.Button(adding, text = "Reset", font = ('Futura', 12), \
                         command=combine_funcs(reset_dev, change_dev_location))
resetting.grid(row=0,column=0)

# Safe float load from entry fields
def try_float(val):
    """Safely parse a string to float.

    Parameters
    ----------
    val : str
        Input string from UI entry widget.

    Returns
    -------
    float | None
        Parsed float value if successful; otherwise None and prints warning.
    """
    try:
        float_val = float(val)
        return float_val
    except ValueError:
        print("Invalid input. Please enter numbers only.")

# add new device position based on the values entered in the window
def collect_devpos():
    """Append a new device trajectory to `device_pos` from current UI entry field values.

    Notes
    -----
    - Angle fields expected in degrees in the UI; converted to radians internally.
    - Uses `try_float` for defensive parsing; invalid entries produce None values.
    """
    x = try_float(xpos.get())
    y = try_float(ypos.get())
    z = try_float(zpos.get())
    a = np.radians(try_float(getalpha.get()))
    b = np.radians(try_float(getbeta.get()))
    g = np.radians(try_float(getgamma.get()))
    global device_pos
    device_pos.append([x,y,z,a,b,g])
    print(device_pos)

add_device = tk.Button(adding, text = "Add Device", font = ('Futura', 12), \
                         command = combine_funcs(collect_devpos, change_dev_location))
add_device.grid(row=0,column=1)

# delete the device last added to the list
def undo_adding():
    """Remove the most recently added device position (LIFO) if present."""
    global device_pos
    if len(device_pos)<=0:
        pass
    else: 
        device_pos.pop()

undo = tk.Button(adding, text = "Undo", font = ('Futura', 12), \
                 command = combine_funcs(undo_adding, change_dev_location))
undo.grid(row=0, column=2)

adding.pack()

# set up checkboxes for montaging
ifmontage = tk.IntVar() 

montage = tk.Checkbutton(root, text = "Montage devices", 
                    variable = ifmontage, 
                    onvalue = True, 
                    offvalue = False, 
                    height = 2, 
                    width = 15,
                    font=('Futura', 12)) 
montage.pack()


# run the program
def run_program():
    """UI callback to validate device presence (unless Angles) and trigger plot + legend.

    Behavior:
        - If attribute != 'Angles' and no devices defined, displays error dialog.
        - Otherwise renders matplotlib legend then Open3D visualization.
    """
    global vertices, normals, device_pos, fields, variable
    if (len(device_pos)<=0) & (variable.get()!="Angles"):
        messagebox.showerror(title = "Missing Device Position", message = "Please enter device position.")
    else:
        combine_funcs(legend(attribute=variable.get()), plot_device(vertices, normals, np.array(device_pos), fields, attribute = variable.get(), mont = ifmontage.get(), inte = False, inflate=if_inflate.get()))


generate_map = tk.Button(root, text = "Generate Map", font = ('Futura', 12), \
                         command=run_program)
generate_map.pack()

# Add checkbox to toggle OBJ saving
save_obj_var = tk.IntVar(value=1)  # Default: save OBJ files
save_obj_checkbox = tk.Checkbutton(root, text="Save OBJ files", variable=save_obj_var, onvalue=1, offvalue=0, font=('Futura', 12))
save_obj_checkbox.pack(pady=(10,0))

root.mainloop()