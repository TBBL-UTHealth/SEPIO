### MDPO: Multi-Device Placement Optimization ###
"""
This script is used to visualize device trajectories within the tissue of interest.
Plots point-wise angle relative to scalp, voltage, SNR, and information capacity.

Script inputs:
- Lead field file and associated variables
- Brain and ROI data files and recording variables

UI inputs:
- Select ROI or whole brain view
- Inflated or deflated view
- Select Voltage, SNR, Information Capacity, or Angle
- Color scheme selection
- Device position values as provided by MDPO_optimize or read out from MDPO_assess

Demo:
To explore the utility of this UI, set the displayed value to 'Angle', clear any devices entered, 
and visualize with different settings to observe the cortex angle relative to scalp.
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
device = 'SEEG'  # Options: {'DISC', 'IMEC', 'SEEG'}
folder = r"...\SEPIO_dataset"
fields_file = path.join(folder,'leadfields', 'SEEG_8e_500um_1500pitch.npz')
#fields_file = path.join(folder,'leadfields', 'DISC_30mm_p2-5Sm_MacChr.npz')
#fields_file = path.join(folder,'leadfields', 'ECoG_1mm_500umgrid.npz')

# Save folder; Screenshots not implemented
output = path.join(folder,"outputs")

# Source space derived from MRI; any number of ROI files in list
# Auditory
roi_data = [loadmat(path.join(folder,'MDPO_data','r_broca.mat'))]
brain_data = [loadmat(path.join(folder,'MDPO_data','brain.mat'))]

brain_name = 'brain' # references file header; ans, brain
roi_name = ['ans'] # references files header; ans, auditory_roi, left_oper, left_tris
roi_weights = [1.] # Weights for ROI regions; len(roi_data) == len(roi_weights); ACC-dlPFC [1.,.5,.5]
roi_include = [True] # Bool; should ROI be included when viewing whole brain
do_weights = False # Bool
recenter_on_brain = True # Bool; Recenter on brain; MUST be used in all standard cases

# Get fields
field_importer = FieldImporter()
field = field_importer.load(fields_file)
num_electrodes = np.shape(field_importer.fields)[4]
fields = field_importer.fields
scale = 0.5 # mm; voxel size; provided lead fields are 0.5 mm/voxel unless stated in file name
cl_wd = 1.0 # mm diameter to clear; device may be slightly offset

# Define tissues dipoles
# Magnitude is 0.5e-9 (nAm) for real signals or 20e-9 (nAm) for phantom simulations
magnitude = 0.5e-9 # nAm
dipole_offset = 1.25 # Placement of dipole between gray/white interface (0) and gray/pial interface (1); est. from Palomero-Gallagher and Zilles 2019
cortical_thickness = 2.5 # Estimate from Fischl and Dale 2000; used for cortex inflation
noise = 2.7 # uV rms; typicall 2.7 uV for standard SEEG or 4.1 uV for standard DiSc

bandwidth = 100 # S/s; May differ between devices depending on backend recording system

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
    """
    Extract data from a given file
    
    inputs: 
        - data = data obtained by the function "loadmat".
        - name = a string that is the name of that brain region in the filename saved in the user's computer

    output: header, faces, vertices, and normals from the provided file
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

    """
    Recenter the data
    
    inputs: 
        - vertices: vertices extracted from a specific brain region
        - reference for avg1, avg2, avg3 calculated from the reference brian region
        
    outputs: vertices_modified, the data in vertices, recentered based on avg1, avg2, and avg3
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
    """
    Transforms the MRI space depending on the device location in order to calculate voltage
    inputs:
        - vertices, array of vertices wrs to the original MRI coordinates
        - normals, array of normal vectors wrs to the original MRI coordinates
        - devpos, manually defined device position
    output: 
        - dippos, shifted and rotated vertices depending on the device location
        - dipvec, shifted and rotated normal vectors depending on the device location
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
    """
    Set values outside of the leadfield as nan so that it is compatible for further calculation
    inputs:
        - leadfield, leadfield data in the form of [x,y,z,[vx,vy,vz],e]
        - vertices, transformed data of the vertices (dippos)
        - normals, transformed data of the normal vectors (dipvec)
    outputs:
        - trimmed dippos, dipvec
    """
    dippos = np.copy(vertices)
    dipvec = np.copy(normals)
    len_half = leadfield.shape[0]//2

    for idx in range(vertices.shape[0]):
        if((np.abs(vertices[idx][0])>len_half) or (np.abs(vertices[idx][1])>len_half) or (vertices[idx][2])>len_half*2) or (vertices[idx][2]<0):
            dippos[idx] = np.nan

    return dippos, dipvec

def calculate_voltage(fields, vertices, normals, vscale=10**6, montage = False, inter = False):
    """
    Calculates voltage for each vertex
    input: 
        - fields, leadfield data in the form of [x,y,z,[vx,vy,vz],e]
        - vertices, transformed and trimmed data of the vertices (dippos)
        - normals, transformed and trimmed data of the normal vectors (dipvec)
        - vscale, to scale it to uV
    output: 
        - opt_volt, a 1-D array that has the optimal voltage for each vertex
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
    snr_list = np.copy(opt_volt)/noise # get the list of snr values
    info_cap = bandwidth*np.log2(1+snr_list) # get the list of information capacity

    # Scale with given weights for ROI subregions
    if do_weights:
        opt_volt = np.multiply(opt_volt,roi_vertices_weights)
        snr_list = np.multiply(snr_list,roi_vertices_weights)
        info_cap = np.multiply(info_cap,roi_vertices_weights)

    return (opt_volt, snr_list, info_cap)

def calculate_angle(brainvert,brainvec):
    """
    Calculates the relative angle of each normal to the scalp.
    input:
        - recentered brain vertices (centered on average CoM)
        - brain normal vectors
    output:
        - angles relative to scale at each vertex, 1-D array as in opt_volt
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
val1, val2, val3, val4, val5, val6 = 0.252, 0.533, 1.186, 2.984, 9.85, 61.8 # IC 15, 30, 60, 120, 240, (480 unused) for 2.3 uV RMS noise

def infocap(val):
    """
    Modified bins for sake of visualization
    """
    return round(bandwidth*np.log2(1+val/noise))

n_bin = [ # Voltage bins
        [np.nan, val1, val2, val3, val4, val5, val6],
         [np.nan,15,30,60,120,240,480], # Info cap bins
         [np.nan,val1/noise,val2/noise,val3/noise,val4/noise,val5/noise,val6/noise], # SNR bins
         [np.nan,30,60,90,120,150,180] # Angle bins
         ] 

def hextorgb(hex):
    """
    Translate hex colors into rgb colors
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
    """
    Get the list of colors for visualization based on the bins the data lies in, and the inquired attribute
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
    """
    To print the legend (because open3d doesn't)
    Uses matplotlib to display color bins for the selected attribute.
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
    """
    Plot vertices and devices using the open3d library for a better view of the 3d space (better rendering)
    Note special case for plotting normal angles at vertices
    inputs:
        - vertices: 2-d array of vertex points of the brain surface
        - normals: 2-d array of vertex normals
        - devices: an array of arrays indicating a list of devices and their position + orientations
        - leadfield: the leadfield array in form of [x, y, z, [vx, vy, vz], e]
        - attribute: a string indicating the desired attribute to calculate (voltage, snr, or infocap)
    output: a plot indicating the voltage sensed by the devices based on location & orientation.
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
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # Get current date and time for filename suffix
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M")
        # Save reconstructed mesh
        mesh_path = os.path.join(savedir, f"surface_mesh_{date_str}.obj")
        o3d.io.write_triangle_mesh(mesh_path, rec_mesh)
        # Save point cloud as OBJ
        points_path = os.path.join(savedir, f"surface_points_{date_str}.obj")
        o3d.io.write_point_cloud(points_path, roi)

        # Save mesh with devices
        device_meshes = [g for g in geometry if isinstance(g, o3d.geometry.TriangleMesh) and g is not rec_mesh]
        if device_meshes:
            # Merge all device meshes with rec_mesh
            merged_mesh = rec_mesh
            for dev_mesh in device_meshes:
                merged_mesh += dev_mesh
            mesh_with_dev_path = os.path.join(savedir, f"surface_mesh_with_devices_{date_str}.obj")
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
    """
    Clears all device positions from the list.
    """
    global device_pos
    device_pos=[]


def combine_funcs(*funcs): 
    """
    Utility to combine multiple functions into a single callback.
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
    """
    Updates the label showing current device positions.
    """
    change = "Current Location: "+str(device_pos)
    current_devices.configure(text=change)

# delete all previously defined device locations
resetting = tk.Button(adding, text = "Reset", font = ('Futura', 12), \
                         command=combine_funcs(reset_dev, change_dev_location))
resetting.grid(row=0,column=0)

# Safe float load from entry fields
def try_float(val):
    """
    Attempts to convert a string to float, returns None if invalid.
    """
    try:
        float_val = float(val)
        return float_val
    except ValueError:
        print("Invalid input. Please enter numbers only.")

# add new device position based on the values entered in the window
def collect_devpos():
    """
    Collects device position and orientation from entry fields and appends to device_pos.
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
    """
    Removes the last device position from the list.
    """
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
    """
    Main callback to generate the visualization map.
    Checks for device positions and calls legend and plot functions.
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