### MDPO Multi-device optimization ###
"""
This script is designed to optimize the placement of multiple devices in a given brain region.
Accepts N-devices of M-types for broader application and UI integration.
Requires input data from Brainsuite and leadfield files - provided in the data folder.

Adjustable variables end at line 103.
"""

##### Imports and Files #####
import numpy as np
from os import path
import sys, os, datetime, time
import pygad
import pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
from modules.leadfield_importer import FieldImporter
from scipy.io import loadmat


##### Device Settings #####
# Files and details
folder = r"...\SEPIO_dataset" # Point to data folder

# Source space derived from MRI; any number of ROI files in list
roi_data = [loadmat(path.join(folder,'MDPO_data','r_broca.mat'))]
brain_data = [loadmat(path.join(folder,'MDPO_data','brain.mat'))]
roi_weights = [1.] # Weights for ROI regions; len(roi_data) == len(roi_weights)
roi_label = ['ans'] # Dict labels for each ROI; most are 'ans', older may be different

# Provide a different list entry for each different device type used
device_types = ['intracortical']  # Options: ['intracortical', 'surface']; format as Python list
device_names = ['SEEG'] # Device names for specific processes; unique processes must be added for each
fields_files = [#path.join(folder,'leadfields','DISC_30mm_p2-5Sm_MacChr-meta.npz')]#, # Point to leadfield files
                path.join(folder,'leadfields','SEEG_8e_500um_1500pitch.npz')]#,
                #path.join(folder,'leadfields','ECoG_1mm_500umgrid.npz')]

# Settings for each device; corresponding list elements
N_multi = [1] # number of each device; assumes the same settings for multiples unless assigned separately
N = sum(N_multi) # Total device count
scale = [0.5] # mm; voxel size; 0.5 mm/voxel for all current lead fields (1/27/2025)
cl_wd = [1.0] # mm diameter to clear; device may be slightly offset; also used for avoiding device contact
                        # Typically 1.0 for intracortical, larger for a grid of surface electrodes (~2-5+)
cl_d = [13] # mm depth to clear; cl variables also used to check device overlap; DiSc 15, SEEG 10-15, ECoG 1
cl_back = [20] # mm distance to check behind devices for overlap of backend recording components; intracortical 20, surface 5
cl_offset = 3 # minimum distance between cl zones when considering device separation; MRI coordinate units
v_scale = [10**6] # Factor to scale LF results to uV; 10**6 typically; 10**-6 for 1mm ecog
noise = [2.3] # uV RMS noise; DiSc 4.1, SEEG 2.3, ECoG 1mm 1.4
bandwidth = [100] # Samples/second; Recording system limit or reasonable maximum
depth_limits = [[np.nan,np.nan]] # mm [min,max] depths for each device; 0 is at cortex edge; positive is inward
                        # Typically [0,np.nan] for depth, [-1,0] for surface
angle_limits = np.array([60.]) # degrees; angle limitations for each device type; ~30 for depth; ~5 for surface; np.nan for no limit (global limits)
angle_limits *= np.pi/180 # convert to radian
Montage = False # Montage devices; Not used in paper
do_depth = True # Boolean; Decide if checking depth limitations
do_proximity = True # Boolean; Decide if checking device proximity limitations

# Global angle limitations; will later evolve to regional exclusion
# Angle [alpha,beta,gamma] points directly upward in MRI space; alpha rotates clockwise looking from above
#       beta rotates clockwise looking from posterior (usually zero); gamma rotates clockwise looking from the left
#       it is easiest to hold one of these constant when setting large exclusion regions (hemispheres)
do_global_angle = False # Boolean; Decide if providing global angle restrictions
global_angle = np.array([[0.,360.],[0.,0.],[120.,240.]]) # [[min,max]...] degrees; limits for each of 3 angles
global_angle *= np.pi/180. # convert to radian

# Grid device specific settings; x,y coordinates of electrodes relative to the desired device center point in mm
grid_positions = [ # Boolean for grid presence; One inner list of [x,y] for each grid device; will be rounded to nearest voxel
    False#,
    #False,
    #[[-1,-1],[1,1],[-1,1],[1,-1]]
]

# Device-specific settings
disc_sensors = 128 # 64 or 128 for DiSc; sensors reduced evenly  in grid

##### Tissue Settings #####
# Define dipoles
# Magnitude is 0.5e-9 (nAm) for real signals or 20e-9 (nAm) for phantom simulations
magnitude = 0.5e-9 # nAm; Default power at given cortical column length
dipole_base_length = 2.5 # mm; Default cortical column length
do_offset = True # Boolean; Decide if providing a uniform offset
dipole_offset = 0.5 # Placement of dipole between gray/white interface (0) and gray/pial interface (1); est. from Palomero-Gallagher and Zilles 2019
cortical_thickness = 2.5 # Estimate from Fischl and Dale 2000; used for cortex inflation
limit_to_roi = False # Force all placement to be within the ROI; only accounts for device origin
verbose = False # Printout control for testing


##### Optimization Settings #####
method = 'genetic' # Available: genetic, anneal, gradient
measure = 'ic' # factor to output and train on; 'voltage', 'snr', or 'ic' ONLY
# Genetic general
num_generations = 30 # Total generation cycles; 50 is often sufficient up to ~5 devices
num_parents_mating = 15 # Top number kept from population between generations; 20
sol_per_pop = 44 # Population per generation; 50
process_count = 30 # Parallel processing count; !!!!! System dependent !!!!! (30)
ss_gens = 3 # number of steady state generations; Disturbs state after this; 5 or 1/10 of generations
# Anneal general
anneal_iterations = 35 # Number of stochastic test steps per temp
anneal_itemp = 100 # Initial temp
anneal_ftemp = 1e-3 # Stopping temp
anneal_cooling_rate = 0.6 # Exponentiated to reduce itemp; 0.5-0.99
anneal_cart_step = 15 # Initial/max cartesian step size
anneal_rot_step = np.pi/3 # Initial/max rotational step size



### Variable processes
# Reassign measure to select the correct value output from 'calculate_voltage' function; acts as index later
if measure == 'voltage':
    measure = 0
elif measure == 'snr':
    measure = 1
elif measure == 'ic':
    measure = 2
else:
    print("ERROR: Incorrect output 'measure' assignment.")

# Load lead fields for each device type
field_importer = FieldImporter()
fields = []
num_electrodes = []
midpoint = []
for f in fields_files:
    field = field_importer.load(f)
    num_electrodes.append(np.shape(field_importer.fields)[4])
    fields.append(field_importer.fields)
    midpoint.append([fields[-1].shape[0]//2,fields[-1].shape[1]//2,fields[-1].shape[2]//2])  # field midpoint index
    if method != 'genetic': # TO DO: Keep genetic from reloading LFs constantly
        print(f"Loaded lead field from {f}")

# Device index list for sorting efforts
N_index = []
for i,n in enumerate(N_multi):
    k = 0
    while k < n:
        N_index.append(i) # add an index for each duplicate device
        k +=1

### Device-specific processing

def field_reduce(field,count):
    # Reduce number of sensors in leadfield if needed (e.g., DiSc 128->64)
    if field.shape[-1] == 128 and count != 128:
        if count == 64:
            keep = np.arange(0,128,2)
            field_out = field[:,:,:,:,keep]
        else:
            print("ERROR: No valid sensor count provided!")
            field_out = None # break process
    else:
        field_out = field # Change nothing
    return field_out

def build_grid(dev_id):
    # Function to construct more complex ECoG grids with custom instructions
    '''
    Inputs:
        - points: [[x_i,y_i],...] for each electrode relative to perceived origin
        Origin must be user defined as their point of reference measurement
    Outputs:
        - updated fields variable
        Iterates the same field at different positions, expanding the total volume covered
    Issues:
        - Assumes a completely flat array
        - User can potentially assign unrealistic or overlapping electrodes
    '''
    # Split x and y values
    x = []
    y = []
    for p in grid_positions[dev_id]:
        x.append(p[0])
        y.append(p[1])
    x = np.array(x)
    y = np.array(y)

    # Create new empty LF of correct size & shape; z unaffected until curvature is added
    field_x = fields[dev_id].shape[0]
    field_y = fields[dev_id].shape[1]
    field_z = fields[dev_id].shape[2]
    dev_scale = scale[dev_id]
    x_range = (np.max(x)-np.min(x))/dev_scale
    y_range = (np.max(y)-np.min(y))/dev_scale
    grid_field = np.zeros((round(field_x+x_range),round(field_y+y_range),field_z,3,x.shape[0]))

    # Fill new field
    for i in range(x.shape[0]):
        xo = (x[i]-np.min(x))/dev_scale
        yo = (y[i]-np.min(y))/dev_scale
        grid_field[round(xo):round(xo+field_x),round(yo):round(yo+field_y),:,:,i,np.newaxis] += fields[dev_id]
    
    return grid_field


### Device-specific processing
for i in range(len(fields)):
    if device_names[i] == 'DiSc':
        fields[i] = field_reduce(fields[i],disc_sensors)
    if device_names[i] == 'ECoG':
        pass
    if device_names[i] == 'SEEG':
        pass
    if device_names[i] == 'other':
        pass
# Device grid construction
for i,g in enumerate(grid_positions): # Check each defined device
    if g: # False for singles, list for grids (True)
        fields[i] = build_grid(i)

### Load extracted MRI data
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

def uniform_offset(vertices,normals,offset):
    # Offset vertices along their normals by a given amount (e.g., to simulate dipole depth)
    temp = vertices.copy()
    #normals = np.nan_to_num(normals)
    for i in range(normals.shape[0]):
        if not np.isnan(normals[i]).any() and np.sum(normals[i]) != 0.:
            k = i # keep last changed value
            vector = normals[i] / np.linalg.norm(normals[i])
            # New normals are shifted by offset along each vector
            vertices[i] = vertices[i] + vector*offset
    # Save last initial value for comparison
    temp = temp[k]
    return vertices,k,temp

### Generate arrays from Brainsuite data; MUST be done before later function definitions
# Values to collect
brain_vals = []
brain_vallen = [[],[],[]] # summative index for each start point
brain_count = [0,0,0] # running index sum
roi_vals = []
roi_vallen = [[],[],[]]
roi_count = [0,0,0]

# Load each file and merge all faces/vertices/normals into single arrays
for i,fb in enumerate(brain_data):
    _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_data[i], 'brain') # 'ans' or 'brain'
    brain_vals.append([brain_faces,brain_vertices,brain_normals])
    for k in range(3):
        brain_vallen[k].append(brain_vals[i][k].shape[0] + brain_count[k])
        brain_count[k] += brain_vals[i][k].shape[0]
for i,fb in enumerate(roi_data):
    _, roi_faces, roi_vertices, roi_normals = obtain_data(roi_data[i], roi_label[i]) # 'ans' or 'roi'
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

# Merge file variables into single arrays for brain and ROI
for i in range(len(brain_data)): # i for each file index
    if i == 0: # first file starts from zero in lists
        brain_faces[0:brain_vallen[0][i]] = brain_vals[i][0]
        brain_vertices[0:brain_vallen[1][i]] = brain_vals[i][1]
        brain_normals[0:brain_vallen[2][i]] = brain_vals[i][2]
    else:
        brain_faces[brain_vallen[0][i-1]:brain_vallen[0][i]] = brain_vals[i][0]
        brain_vertices[brain_vallen[1][i-1]:brain_vallen[1][i]] = brain_vals[i][1]
        brain_normals[brain_vallen[2][i-1]:brain_vallen[2][i]] = brain_vals[i][2]
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

### Provide offset for normal locations to realistic dipole depth in cortex
# Note: Consider using outer cortical mesh or inner mesh curvature to improve accuracy
if do_offset:
    roi_vertices,k,temp = uniform_offset(roi_vertices,roi_normals,dipole_offset*cortical_thickness)
    if verbose: # Print to compare shift of offset
        print("Vertices shape:",roi_vertices.shape,"\nNormals shape:",roi_normals.shape)
        print(f"Before movement ex.: {temp}")
        print(f"After movement ex.: {roi_vertices[k]} at index {k}")
        print(f"Change of {temp-roi_vertices[k]}, totalling {np.sqrt(np.sum((temp-roi_vertices[k])**2))} along vector {roi_normals[k]}")


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
    # print(avg1, avg2, avg3)
    
    # create a copy of the vertices
    vertices_modified = vertices.copy()

    # modify the data according to the midpoint
    for idx in range(vertices.shape[0]):
        vertices_modified[idx][0] -= avg1
        vertices_modified[idx][1] -= avg2
        vertices_modified[idx][2] -= avg3

    return vertices_modified,center

# Centering to average position for all brain coordinates; vital to device angle and depth correction
recentered_roi, center = recenter(roi_vertices, brain_vertices) # Use center to reverse the shift for MRI coordinates on final export

def get_rotmat(alpha, beta, gamma):
    """
    Given device position in standard form, return rotation matrix
    """
    # define rotation matrix 
    # (see wikipedia page what these matrices represent: 
    # https://en.wikipedia.org/wiki/Rotation_matrix#Basic_3D_rotations)

    yaw = np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
    pitch = np.array([[np.cos(beta), 0, np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
    roll = np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
    rot_mat = np.matmul(yaw, pitch)
    rot_mat = np.matmul(rot_mat, roll)

    return rot_mat

def transform_vectorspace(dev_id, vertices, normals, devpos):
    """
    Transforms the MRI space depending on the device location in order to calculate voltage
    inputs:
        - dev_id, device index for lead field selection
        - vertices, array of vertices wrs to the original MRI coordinates
        - normals, array of normal vectors wrs to the original MRI coordinates
        - devpos, device position
    output: 
        - dippos, shifted and rotated vertices depending on the device location
        - dipvec, shifted and rotated normal vectors depending on the device location
    """

    rot_mat = get_rotmat(devpos[3],devpos[4],devpos[5])
    inv_rot_mat = np.linalg.inv(rot_mat)

    field = fields[dev_id]

    ###
    ##### No transformation done above this line #####
    ###

    dippos = np.copy(vertices)
    dipvec = np.copy(normals)

    # translation
    dippos-=devpos[:3]
    dippos[0] += field.shape[0]//2
    dippos[1] += field.shape[1]//2

    # rotate vertices and normals based on the placement of the device
    for idx in range(vertices.shape[0]):   # rotate for each vertex point and corresponding vectors
        dippos[idx] = np.matmul(inv_rot_mat, dippos[idx])
        dipvec[idx] = np.matmul(inv_rot_mat, dipvec[idx])

    # Transfer position into leadfield space
    dippos *= 1/scale[dev_id]

    # to make it compatible with the lead field data
    dippos = (dippos.astype('int')).astype('float')
    
    # Scale to dipole nAm magnitude, see value above
    dipvec *= magnitude 

    return dippos, dipvec, rot_mat

def adjust_depth(depthrange, position, vertices):
    """
    Based on the restriction set on device depth, adjust device depth when needed
    inputs:
        - depthrange, min/max depth for a given device
        - position, device position in the form [x, y, z, alpha, beta, gamma] in MRI coordinates
        - vertices, set of vertices in MRI coordinates
    outputs:
        - position, adjusted (if necessary) device position
    """

    mindepth = depthrange[0]
    maxdepth = depthrange[1]

    # Device position unit vector
    dev_vec = position[:3].copy()/np.linalg.norm(position[:3])
    angle_x = np.arctan2(dev_vec[1],dev_vec[0])
    Rz = np.array([
        [np.cos(angle_x),-np.sin(angle_x),0],
        [np.sin(angle_x),np.cos(angle_x),0],
        [0,0,1]
    ])
    rot_pos = np.dot(dev_vec,Rz)
    if verbose:
        print("TEST rotation 1/2",dev_vec,rot_pos)
    angle_z = np.arctan2(rot_pos[2],rot_pos[0])
    Ry = np.array([
        [np.cos(angle_z),0,np.sin(angle_z)],
        [0,1,0],
        [-np.sin(angle_z),0,np.cos(angle_z)]
    ])
    rot_pos = np.dot(rot_pos,Ry.T)
        
    # Change origin to the device then,
    # rotate all vertices such that the x axis points toward the device position
    rot_vert = vertices.copy() - position[:3].copy()
    rot_vert = np.dot(np.dot(rot_vert,Rz),Ry.T)

    # Get value of depth from farthest source; +x is outward from brain, -x is inward
    depth = np.nanmax(rot_vert[:,0]) # distance from device to surface-most source along the device-center axis

    if verbose:
        print("TEST rotation 2/2",rot_pos)
        print(f"TEST vertices: {vertices.shape} with min {np.nanmin(vertices,axis=0)} and max {np.nanmax(vertices,axis=0)}")
        print(f"TEST position: {vertices.shape} with {position}")
        print(f"TEST rot_vert: {rot_vert.shape} with min {np.nanmin(rot_vert,axis=0)}  and max {np.nanmax(rot_vert,axis=0)}")
        print(f"TEST depth min {np.nanmin(rot_vert[0])} max {np.nanmax(rot_vert[0])}")

    # Check and move devices to minimum
    if not np.isnan(mindepth):
        # Check minimum depth is present
        if depth < mindepth:
            # move inward if too far from brain
            dif = abs(depth-mindepth)
            position[:3] -= dif*dev_vec
            #print(f"TEST Moving: {dif} to {mindepth}")

    # Check and move devices to maximum
    if not np.isnan(maxdepth):
        if depth > maxdepth:
            # move outward if too deep in brain
            dif = abs(depth-maxdepth)
            position[:3] += dif*dev_vec
            #print(f"TEST Moving: {dif} to {maxdepth}")

    return position

def check_proximity(population,dev1_id,dev2_id):
    """
    Search algorithm for determining the closest points between two devices
    based on their defined LF clearing areas into and out of the brain.
    This method is a rough distance approximation. Is there an analytic solution?
    inputs:
        - positions, (N,6) vector describing current solution
        - dev1_id and dev2_id, device IDs to compare; references global variables
    outputs:
        - proximity, value of distance between cylinders; can be negative for intersection!
        - axis, unit vector pointing between nearest points from dev1 to dev2
    """
    # Find device id for global variable lists ONLY!
    i1 = N_index[dev1_id]
    i2 = N_index[dev2_id]

    # Define device vectors and normalize
    dev1_vec = np.matmul(get_rotmat(population[dev1_id,3], population[dev1_id,4], population[dev1_id,5]), np.array([0,0,-1]))
    dev2_vec = np.matmul(get_rotmat(population[dev2_id,3], population[dev2_id,4], population[dev2_id,5]), np.array([0,0,-1]))
    dev1_vec *= 1/np.linalg.norm(dev1_vec)
    dev2_vec *= 1/np.linalg.norm(dev2_vec)

    # Calculate radial vectors for each pointing toward the other
    rvec1 = -np.cross(np.cross(dev1_vec,dev2_vec),dev1_vec)
    rvec2 = -np.cross(np.cross(dev2_vec,dev1_vec),dev2_vec)

    # Define end points ; [deep,backend]
    p1 = [population[dev1_id,:3] + cl_d[i1]*dev1_vec, population[dev1_id,:3] - cl_back[i1]*dev1_vec]
    p2 = [population[dev2_id,:3] + cl_d[i2]*dev2_vec, population[dev2_id,:3] - cl_back[i2]*dev2_vec]

    # Search variables
    closest = np.zeros((2,3)) # closest points so far
    divisions = 100 # number of divisions for each search

    # Start search using device end points
    mix = np.linspace(0.,1.,divisions) # array of divisions for point mixing
    mix = np.tile(mix[:,np.newaxis],(1,3))
    # set up points along the given divisions
    d1 = (1-mix)*p1[0] + mix*p1[1]
    d2 = (1-mix)*p2[0] + mix*p2[1]
    best = 10**3 # initialize best proximity
    for j in d1:
        for k in d2:
            # points j and k to find which ones are nearest
            prox = np.linalg.norm(j-k) # distance from j to k
            if prox < best:
                best = prox
                closest[0] = j
                closest[1] = k
    
    # Calculate distance between cylinders assuming line passes through both
    axis = closest[1] - closest[0] # Between closest points
    if np.all(closest[1] != closest[0]):
        proximity = best
    else: # If nearest points overlap, base it on device dirction
        proximity = 0.01
    if proximity < 0.1: # If proximity is close, move apart by device origins
        if np.all(dev1_vec != dev2_vec): # If devices don't overlap
            axis = dev2_vec - dev1_vec
        else: # If devices overlap, give it a defined direction
            axis = np.array([.01,0,0])
    else: # If not too close, move by nearest point
        axis *= 1/best # Normalize the axis vector
    
    # Adjust proximity by device radii
    proximity -= (cl_wd[i1]+cl_wd[i2])/2

    if verbose:
        print(f"TEST prox - axis: {proximity} - {axis}")

    return proximity, axis

def trim_data(dev_id, leadfield, vertices, normals):
    """
    Set values outside of the leadfield as nan so that it is compatible for further calculation
    inputs:
        - dev_id, index for device type variables
        - leadfield, leadfield data in the form of [x,y,z,[vx,vy,vz],e]
        - vertices, transformed data of the vertices (dippos)
        - normals, transformed data of the normal vectors (dipvec)
    outputs:
        - trimmed dippos, dipvec
    """

    dippos = np.copy(vertices)
    dipvec = np.copy(normals)
    # half of side length
    len_half = midpoint[dev_id]

    # NaN dipoles outside of the lead field so they aren't used in later calculation
    for idx in range(vertices.shape[0]):
        if((np.abs(vertices[idx][0])>len_half[0]) or (np.abs(vertices[idx][1])>len_half[1]) or (vertices[idx][2])>len_half[2]*2) or (vertices[idx][2]<0):
            dippos[idx] = np.nan

    return dippos, dipvec

def calculate_voltage(dev_id, vertices, normals, montage = Montage, inter = False):
    """
    Calculates voltage for each vertex
    input: 
        - dev_id, device type index that corresponds to user settings
        - vertices, transformed and trimmed data of the vertices (dippos)
        - normals, transformed and trimmed data of the normal vectors (dipvec)
        - vscale, to scale it to uV
        - inter, an inter-device montage; requires adaptation to work with different device types
    
    output: 
        - opt_volt, a 1-D array that has the optimal voltage for each vertex
    """

    # make copies of data
    dippos = np.copy(vertices)
    dipvec = np.copy(normals)
    field = np.copy(fields[dev_id])

    # create a list of field vectors applicable to the surface, [vertex index, electrode #]
    voltage = np.empty((dippos.shape[0], field.shape[-1]))
    voltage[:]=np.nan
    opt_volt = np.empty(dippos.shape[0])
    opt_volt[:]=np.nan
    
    # calculate voltage for each vertex
    for idx in range(dippos.shape[0]):
        for e in range(field.shape[-1]): # for every electrode
            if not np.any(np.isnan(dippos[idx])): # if not nan, calculate and modify entry
                lead_field = field[int(dippos[idx][0]+field.shape[0]//2), int(dippos[idx][1]+field.shape[0]//2), int(dippos[idx][2]), :, e]
                voltage[idx, e] = np.dot(lead_field, dipvec[idx])
        if (not np.all(np.isnan(voltage[idx,:]))) and (not montage): 
            opt_volt[idx] = np.nanmax(np.abs(np.copy(voltage[idx]))) # get the optimal voltage across all electrodes
        elif (not np.all(np.isnan(voltage[idx,:]))) and montage and (not inter):
            opt_volt[idx] = np.nanmax(voltage[idx]) - np.nanmin(voltage[idx])
        # elif (not np.all(np.isnan(voltage[idx,:]))) and montage and inter: # Universal montage option?

    # Scale and calculate output values
    opt_volt *= v_scale[dev_id] #scale it to get the list of voltages
    snr_list = np.copy(opt_volt)/noise[dev_id] # get the list of snr values
    info_cap = bandwidth[dev_id]*np.log2(1+snr_list) # get the list of information capacity

    # Zero NaN values
    opt_volt = np.nan_to_num(opt_volt)
    snr_list = np.square(np.nan_to_num(snr_list))
    info_cap = np.nan_to_num(info_cap)

    # Scale with given weights for ROI subregions
    opt_volt = np.multiply(opt_volt,roi_vertices_weights)
    snr_list = np.multiply(snr_list,roi_vertices_weights)
    info_cap = np.multiply(info_cap,roi_vertices_weights)

    return (opt_volt, snr_list, info_cap)

def limit_correction(population, coefficient):
    """
    Given a population and the maximum angle, adjust the device positions in the population so that the angles are within the limit.
    Also assesses inter-device distance to avoid collision.
    inputs:
        - population, the initial population of device positions during the optimization
        - coefficient, corresponds to the step size. only the ratio between the two numbers matter, not their magnitude
    outputs:
        - population, the population with corrected positions, angles, and relative distances
    """
    # Setting for cycle correction limit
    max_cycles = 3 # Can only loop angle, depth, proximity this many times

    # Edge case for single solutions
    single_sol = False
    if population.ndim == 1: # One solution; expand
        single_sol = True # Mark to return population shape
        population = population[np.newaxis,:] # add 0 axis
    device_num = population.shape[1] // 6
    sol_shape = population[0].shape

    for idx in range(population.shape[0]): # idx for each solution, one full set of positions
        # Track changes and restart until no changes are made; This accounts for interaction
        #   of angle, depth, and proximity corrections.
        start_pop = np.copy(population[idx]) # grab the initial population for comparison
        cycle = 0
        while (np.all(start_pop != population[idx]) or cycle == 0) and (cycle < max_cycles):
            # Loop maintenance
            cycle += 1 # count cycles

            for dev in range(device_num): # dev for each device
                # Load values for each device
                depth_lim = depth_limits[N_index[dev]] # each set of [lower,upper] limits for each device type
                angle_lim = angle_limits[N_index[dev]] # single angle limit for each device
                counter = dev * 6 # cycles through each device

                # Vectors to compare
                dev_vec = np.matmul(get_rotmat(population[idx][counter+3], population[idx][counter+4], population[idx][counter+5]), np.array([0,0,-1])) # orientation of device
                dev_vec /= np.linalg.norm(dev_vec) # normalize vector
                vertex_vec = np.array([population[idx][counter], population[idx][counter+1], population[idx][counter+2]]) # vector from origin to device position
                vertex_vec /= np.linalg.norm(vertex_vec) # normalize vector
                rotation_changed = False

                # if the angle is out of bound, adjust until it is within the limit
                while (np.dot(dev_vec, vertex_vec) < np.cos(angle_lim)) and angle_lim != np.nan:
                    # Use coefficient to split vector toward allowable angles
                    dev_vec = (coefficient[0] * vertex_vec + coefficient[1] * dev_vec)/np.sum(coefficient)
                    dev_vec /= np.linalg.norm(dev_vec)
                    rotation_changed = True
                
                # Values for global changes and corrections
                xaxis = np.cross(dev_vec, np.array([0,0,10]))
                xaxis /= np.linalg.norm(xaxis)
                yaxis = np.cross(dev_vec, xaxis) # define arbitrary new x and y axis orientations to find the euler angles
                yaxis /= np.linalg.norm(yaxis) 
                alpha = np.arctan(xaxis[1]/xaxis[0])
                beta = np.arcsin(-xaxis[2])
                gamma = np.arctan(yaxis[2]/dev_vec[2])

                # Correct angles to positives and between -2pi to +2pi
                comp = [alpha,beta,gamma]
                for a in range(2):
                    while comp[a] <=-2*np.pi:
                        comp[a] += 2*np.pi
                    while comp[a] >= 2*np.pi:
                        comp[a] -= 2*np.pi

                dif = [0.,0.,0.] # changes for alpha,beta,gamma
                # Check global rotation bounds for 'global_angle'
                if do_global_angle:
                    # Check and correct each rotational axis
                    for a in range(3): # a for alpha, beta, gamme indices
                        if (global_angle[a][0] > abs(comp[a])): # minimum bound
                            if comp[a] < 0: # negative case
                                dif[a] = - global_angle[a][0] - comp[a] # negative to reach -minimum
                            else:
                                dif[a] = global_angle[a][0] - comp[a] # positive to reach +minimum
                            rotation_changed = True
                            if verbose:
                                print("Global angle adjusting from below minimum. Angle index:",a,"- Change:",dif[a])
                        elif (abs(comp[a]) > global_angle[a][1]): # maximum bound
                            if comp[a] < 0: #negative case
                                dif[a] = - global_angle[a][1] - comp[a] # positive to reduce to -maximum
                            else:
                                dif[a] = global_angle[a][1] - comp[a] # negative to reduce to +maximum
                            if verbose:
                                print("Global angle adjusting from above maximum. Angle index:",a,"- Change:",dif[a])
                            rotation_changed = True
                        else:
                            dif[a] = 0.
                
                alpha = comp[0] + dif[0]
                beta = comp[1] + dif[1]
                gamma = comp[2] + dif[2]
                
                # convert it back to the standard device format
                if rotation_changed:
                    result = np.matmul(get_rotmat(alpha, beta, gamma), np.array([0,0,-1]))
                    if dev_vec[1] + result[1] < 0.0001:
                        gamma = -gamma

                population[idx][counter:counter+6] = np.array([population[idx][counter], population[idx][counter+1], population[idx][counter+2], alpha, beta, gamma])

                # If depth is out of bound, adjust until it is within the limit
                if do_depth:
                    population[idx][counter:counter+6] = adjust_depth(depth_lim,population[idx][counter:counter+6],recentered_roi)

            # Check and correct device proximity
            if do_proximity:
                pop = np.reshape(population[idx],shape=(device_num,6))
                for d1 in np.arange(device_num-1):
                    for d2 in np.arange(d1+1,device_num):
                        # d1 and d2 are device ID combinations for every possible pair; axis points to d2
                        proximity, axis = check_proximity(pop,dev1_id=d1,dev2_id=d2)
                        if proximity < cl_offset: # Devices are too close
                            dif = cl_offset - proximity*1.1 # move devices as little as possible; *1.1 as room for error with other methods
                            pop[d1,:3] -= (dif/2)*axis # Move along the nearest proximity axis
                            pop[d2,:3] += (dif/2)*axis
                            population[idx] = np.reshape(pop,shape=(device_num*6))


    # Adjust returned value for edge case of one solution
    if single_sol: # Remove zero dimension for one solution
        population = population[0] 

    return population

def find_snr(devpos):
    """
    Find the total SNR generated by N devices.
    devpos - population variables
    """
    devpos = devpos.reshape((N, 6)) # (device_id,position values)
    all_dev_volt = np.empty([recentered_roi.shape[0], N]) # (vertices,device count)
    for idx in range(N): # idx for device in population, N_index[idx] for device type
        dev_id = N_index[idx] # device type index
        dippos, dipvec, rotmat = transform_vectorspace(dev_id,recentered_roi, roi_normals, devpos[idx])
        dippos_adj, dipvec_adj = trim_data(dev_id, fields, dippos, dipvec)
        all_dev_volt[:, idx] = calculate_voltage(dev_id, dippos_adj, dipvec_adj, montage = Montage, inter = False)[measure]
    
    voltage = np.nanmax(np.nan_to_num(all_dev_volt, nan=0), axis=1)
    total = np.nansum(voltage)
    # print("total IC:", total, "Bits/s")
    return total

def fitness_function(ga_instance, solution, solution_idx):
    # Fitness function for genetic algorithm; uses SNR as objective
    return find_snr(solution)

def on_gen(ga_instance):
    """
    Perform limit correction after each population of solutions is generated.
    """
    last_gen_fitnesses = ga_instance.last_generation_fitness
    ga_instance.population = limit_correction(ga_instance.population, coefficient=[3, 2])
    print("Generation:", ga_instance.generations_completed)
    # print("Fitness values:", ga_instance.last_generation_fitness)
    # print("Best solution:", ga_instance.best_solution()[0])
    print("Fitness of the best solution:", ga_instance.best_solution()[1])
    print("Position:",ga_instance.best_solution()[0])

    fitnesses.append(ga_instance.best_solution()[1])
    best_sols.append(ga_instance.best_solution()[0])
    mean_fitnesses.append(np.mean(ga_instance.last_generation_fitness))
    std_fitnesses.append(np.std(ga_instance.last_generation_fitness))

def generate_initial(vertices, N, min_distance, max_iterations, learning_rate):
    """
    Used to generate sparsely spaced initial positions of devices.
    Use gravitational repel if two devices are placed too close to each other. 
    Inputs:
    - min_distance: minimum distance between two devices 
    - max_iterations: number of maximum iterations 
    - learning_rate: step size of each iteration
    """
    # randomly generate initial positions of N devices 
    indices = np.random.choice(len(vertices), N, replace=False)
    initial_vertices = vertices[indices] 

    # gravitational repel
    forces = np.zeros_like(initial_vertices)
    for _ in range(max_iterations): 
        for i in range(N): 
            for j in range(N): # for each pair of devices i-j
                if i != j: # every combination, saving values for i
                    distance = np.linalg.norm(initial_vertices[i] - initial_vertices[j])
                    if distance < min_distance:
                        force_mag = 1 / np.square(distance) # f = 1/(distance)^2 - inverse square law
                        # normalize force directions 
                        force = force_mag * (initial_vertices[i] - initial_vertices[j])
                        forces[i] += force
                    
        initial_vertices += forces * learning_rate
    
    initial_pop = []

    # rotational coordinates
    for pos in initial_vertices:
        dev_vec = pos / np.linalg.norm(pos) # device position unit vector
        xaxis = np.cross(dev_vec, np.array([0, 0, 10]))
        xaxis /= np.linalg.norm(xaxis)

        yaxis = np.cross(dev_vec, xaxis)
        yaxis /= np.linalg.norm(yaxis)

        alpha = np.arctan2(xaxis[1], xaxis[0])
        beta = np.arcsin(-xaxis[2])
        gamma = np.arctan2(yaxis[2], dev_vec[2])

        if gamma < 0:
            gamma = np.pi + gamma

        result = np.matmul(get_rotmat(alpha, beta, gamma), np.array([0, 0, -1]))
        if dev_vec[1] + result[1] < 0.0001:
            gamma = -gamma

        initial_pop.append([pos[0], pos[1], pos[2], alpha, beta, gamma])
    
    return np.array(initial_pop)

def generate_initial_population(recentered_roi, N, k):
    """
    Generate initial population containing k solutions with N devices. 
    """
    result = []
    for _ in range(k):
        sol = generate_initial(recentered_roi, N, 5, 20, 0.5) 
        result.append(sol)
    return np.array(result)

def mutation_func(offspring, ga_instance):
    """
    Adaptive mutation. 
    """
    # increase mutation rate and decrease mutation step when reaching steady state
    if len(fitnesses) > ss_gens and len(set(fitnesses[-ss_gens:])) == 1:
        mutation_rate = 0.9
        mutation_step = 0.4
    # decrease mutation rate and increase mutation step
    else: 
        mutation_rate = 0.6
        mutation_step = 1 
    for chromosome_idx in range(offspring.shape[0]):
         for gene_idx in range(offspring.shape[1]):
             if np.random.random() < mutation_rate:
                 random_val = np.round(np.random.uniform(offspring[chromosome_idx, gene_idx] - mutation_step,
                                                         offspring[chromosome_idx, gene_idx] + mutation_step), 2)
                 if limit_to_roi:
                    random_val = np.clip(random_val, lower_bounds[gene_idx % 6], upper_bounds[gene_idx % 6])
                 offspring[chromosome_idx, gene_idx] = random_val 
    return offspring 

def crossover_func(parents, offspring_size, ga_instance):
    """
    Two points crossover at first and single point crossover after reaching steady state.
    """
    offspring = []
    # single point (from PyGad)
    if len(fitnesses) > ss_gens and len(set(fitnesses[-ss_gens:])) == 1:
        idx = 0
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            random_split_point = np.random.choice(range(offspring_size[1]))
            parent1[random_split_point:] = parent2[random_split_point:]
            offspring.append(parent1)
            idx += 1
    # two points 
    else: 
        idx = 0 
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            point1 = np.random.randint(1, len(parent1) - 1)
            point2 = np.random.randint(point1, len(parent1))
            offspring.append(np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]]))
            idx += 1
    return np.array(offspring)


def simulated_annealing(initial_solution, objective_function, lower_bounds, upper_bounds, 
                        n_iterations, cooling_rate, initial_temp, min_temp, cartesian_step,
                        rotational_step):
    """
    Apply simulated annealing to find optimal positions for N devices
    - initial_solution: an initial solution generated using gravitational repel (could be replaced by a known solution)
    - objective function: find_snr
    - lower_bounds and upper_bounds: spatial limitations
    - n_iterations
    - colling_rate
    - initial_temp
    - cartesian_step and rotational_step
    """
    N = len(initial_solution)
    current_sol = initial_solution.flatten()
    current_snr = objective_function(current_sol)
    temp = initial_temp
    print('N:',N)
    print('Initial fitness:',current_snr)
    
    snr_list = [current_snr] # Recorded changes in current SNR
    temp_list = [temp] # Correponding temp to SNR list
    solutions = [current_sol]

    cycle = 0
    while temp > min_temp: # stopping criteria
        # Cycle counter & printout
        print(f'Cycle {cycle} Temp {temp} ({initial_temp} to {min_temp})')
        cycle += 1
        
        # Reducing factor for steps; 0-1 based on temp; reduced to slow descent
        temp_factor = ((temp-min_temp)/(initial_temp-min_temp))**(cooling_rate/4) # was /2 before 10.28.24

        # Iterate last solution into population for bulk search
        population = np.repeat(current_sol[None,:],n_iterations,axis=0)

        # Perturb solutions
        for i in range(0, N, 6): # i for each device
            cartesian_perturb = np.random.uniform(-1, 1, size=(n_iterations,3)) * cartesian_step * temp_factor
            population[:,i:i+3] += cartesian_perturb
            rotational_perturb = np.random.uniform(-1, 1, size=(n_iterations,3)) * rotational_step * temp_factor
            population[:,i+3:i+6] += rotational_perturb

        # Apply constraints; Requires reshape on input
        if limit_to_roi:
            population = np.clip(population, lower_bounds, upper_bounds)
        population = limit_correction(population, coefficient=[3, 2])

        # Collect new scores
        new_snr = []
        for i in range(n_iterations):
            new_snr.append(objective_function(population[i]))

        # Decie if adopting a new solution
        if np.max(new_snr) > current_snr or np.random.uniform() < np.exp((np.max(new_snr) - current_snr) / temp):
            # Adopt new solution
            current_sol = population[np.argmax(new_snr)]
            current_snr = np.max(new_snr)
            print(f'New fitness {round(current_snr)} at temp {temp}')
        
        # Save point relative to temp regardless
        snr_list.append(current_snr)
        temp_list.append(temp)
        solutions.append(current_sol)

        # Reduce temp
        temp *= cooling_rate
    
    best_snr = np.max(snr_list)
    best_sol = solutions[np.argmax(snr_list)]

    return best_snr, best_sol, snr_list, temp_list, solutions


### Main Loop ###
if __name__ == "__main__":
    ### Time keeping
    start_time = datetime.datetime.now()
    print(f"Method {method} with {N} devices start at: {time.strftime('%Y-%m-%d %H:%M:%S', start_time.timetuple())}")

    ### Initialize first population
    num_genes = N * 6 # Must be 6 genes times number of devices
    # set cartesian bounds (option to +- perturbation)
    min_values = np.min(recentered_roi, axis=0) 
    max_values = np.max(recentered_roi, axis=0)
    lower_bounds = np.tile([min_values[0], min_values[1], min_values[2], -np.pi, -np.pi, -np.pi], N)
    upper_bounds = np.tile([max_values[0], max_values[1], max_values[2], np.pi, np.pi, np.pi], N)

    # generate initial population and adjust to fit in bounds
    if method == 'genetic':
        init_pop = sol_per_pop
    else:
        init_pop = 50 # Starting spot test for other methods
    initial_population = np.round(generate_initial_population(recentered_roi, N, init_pop).reshape(init_pop, num_genes), 2)
    for i,p in enumerate(initial_population):
        initial_population[i] = np.clip(initial_population[i], lower_bounds, upper_bounds) # Clip to bounds
        initial_population[i] = limit_correction(initial_population[i], coefficient=[3, 2])
    
    ### Run desired optimization
    # Genetic algorithm
    if method == 'genetic':
        # initialize storing variables
        fitnesses = []
        mean_fitnesses = []
        std_fitnesses = []
        best_sols = []

        # Set up GA model
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            gene_space=[[min_values[0], max_values[0]],
                        [min_values[1], max_values[1]],
                        [min_values[2], max_values[2]],
                        [-np.pi, np.pi],
                        [-np.pi, np.pi],
                        [-np.pi, np.pi]] * N, # seems like it works just fine if we don't specify gene_space 
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            parent_selection_type="sss", # steady state selection 
            crossover_type=crossover_func,
            mutation_type=mutation_func,
            initial_population=initial_population,
            on_generation=on_gen,
            save_best_solutions=False,
            keep_elitism=1,
            parallel_processing=['process', process_count])
        
        # Run GA model
        ga_instance.run()

        ### Time keeping
        end_time = datetime.datetime.now()
        print(f"Script end at: {time.strftime('%Y-%m-%d %H:%M:%S', end_time.timetuple())}")
        # Total computation time
        total_time = end_time - start_time
        print(f"Total computation time: {total_time}")

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()

        print("Optimized positions for all devices:", best_solution.reshape((N, 6)))
        print("Highest IC achieved:", best_solution_fitness)
        print("Fitnesses over generations:", fitnesses)

        t_save = time.strftime('%Y%m%d_%H%M', start_time.timetuple())
        with open(path.join(folder,'outputs',f'{t_save}-genetic-{N}.pkl'), 'wb') as file:
            pickle.dump({
                'type':[f'{method}'],
                'best solution:': best_solution,
                'best solution fitness': best_solution_fitness,
                'fitnesses': fitnesses, 
                'best solution over generations': best_sols,
                'mean fitnesses': mean_fitnesses,
                'std fitnesses': std_fitnesses,
                'device names': device_names,
                'device types': device_types,
                'field files': fields_files,
                'N per device': N_multi,
                'LF scale': scale,
                'dev diameter': cl_wd,
                'dev depth': cl_d,
                'dev bacend': cl_back,
                'voltage scale': v_scale,
                'dev bandwidth': bandwidth,
                'depth limits': depth_limits,
                'depth bool': do_depth,
                'angle limits': angle_limits,
                'global angle bool': do_global_angle,
                'global angle limits': global_angle,
                'montaged': Montage
            }, file)
    
    # Annealing
    elif method == 'anneal':
        # Storing variables
        anneal_fitness = []
        anneal_temp = []

        # Set initial population
        initial_scores = np.array([find_snr(sol) for sol in initial_population])
        sorted_indices = np.argsort(initial_scores)[::-1]  # Sort in descending order
        current_solutions = initial_population[sorted_indices] # Sort the solutions
        initial_scores = initial_scores[sorted_indices] # Sort their correponding scores
        initial_solution = initial_population[sorted_indices[0]] # Pull only one solution to do anneal
        print('First soln.:',initial_solution,find_snr(initial_solution))

        # Run anneal operation
        best_snr, best_sol, snr_list, temp_list,solutions = simulated_annealing(initial_solution=initial_solution, 
                        lower_bounds=lower_bounds, 
                        upper_bounds=upper_bounds,
                        objective_function=find_snr, 
                        n_iterations=anneal_iterations, 
                        initial_temp=anneal_itemp,
                        min_temp=anneal_ftemp,
                        cooling_rate=anneal_cooling_rate,
                        cartesian_step=anneal_cart_step, 
                        rotational_step=anneal_rot_step)
        
        print("Best result:",best_snr,"\nSolution:",best_sol)
        
        ### Time keeping
        end_time = datetime.datetime.now()
        print(f"Script end at: {time.strftime('%Y-%m-%d %H:%M:%S', end_time.timetuple())}")
        # Total computation time
        total_time = end_time - start_time
        print(f"Total computation time: {total_time}")

        t_save = time.strftime('%Y%m%d_%H%M', start_time.timetuple())
        with open(path.join(folder,'outputs',f'{t_save}-anneal-{N}.pkl'), 'wb') as file:
            pickle.dump({
                'type':[f'{method}'],
                'best solution:': best_sol,
                'best solution fitness': best_snr,
                'fitnesses': snr_list, 
                'solutions': solutions,
                'temperature': temp_list,
                'device names': device_names,
                'device types': device_types,
                'field files': fields_files,
                'N per device': N_multi,
                'LF scale': scale,
                'dev diameter': cl_wd,
                'dev depth': cl_d,
                'dev bacend': cl_back,
                'voltage scale': v_scale,
                'dev bandwidth': bandwidth,
                'depth bool': do_depth,
                'depth limits': depth_limits,
                'angle limits': angle_limits,
                'global angle bool': do_global_angle,
                'global angle limits': global_angle,
                'montaged': Montage
            }, file)
        

    # Method error
    else:
        print('!!! ERROR: `method` provided incorrectly !!!')