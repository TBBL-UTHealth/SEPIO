"""MDPO project model and compute utilities.

Includes:
- A serializable `Project` container for inputs, device settings, and results.
- Geometry helpers for meshes, transforms, and binning.
- Optimization routines (genetic, anneal variants, gradient, branch-bound, brute).
- SEPIO helpers for computing voltage and IC matrices.

Designed to keep the GUI lightweight by isolating computational logic here.
"""
from __future__ import annotations
import os, pickle, time, datetime, sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np
from scipy.io import loadmat

try:
    import pygad  # Optional; GUI will guard calls if unavailable
    PYGAD_AVAILABLE = True
except ImportError:
    pygad = None  # type: ignore
    PYGAD_AVAILABLE = False

# Attempt relative import fallback for repository structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)  # parent of GUI
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
MODULES_DIR = os.path.join(SCRIPTS_DIR, 'modules')
# Ensure scripts/modules is importable so `leadfield_importer` can be found
for p in (REPO_ROOT, SCRIPTS_DIR, MODULES_DIR):
    if p and p not in sys.path:
        sys.path.insert(0, p)
# Resolve FieldImporter with preference for the new GUI location
FieldImporter = None
try:
    # Sibling import (GUI/leadfield_importer.py)
    from leadfield_importer import FieldImporter as _FI  # type: ignore
    FieldImporter = _FI
except Exception:
    try:
        # Try modules package if present
        from modules.leadfield_importer import FieldImporter as _FI  # type: ignore
        FieldImporter = _FI
    except Exception:
        try:
            # Fallback to dynamic import from scripts/modules by absolute path
            import importlib.util
            lf_path = os.path.join(MODULES_DIR, 'leadfield_importer.py')
            if os.path.exists(lf_path):
                spec = importlib.util.spec_from_file_location('leadfield_importer', lf_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    FieldImporter = getattr(mod, 'FieldImporter', None)
        except Exception:
            FieldImporter = None  # GUI will surface friendly error

# ----------------------------------------------------------------------------------
# Utility extraction from original scripts (minimized & parameterized)
# ----------------------------------------------------------------------------------

def obtain_data(data: dict, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    header = data['__header__']
    faces = data[name][0][0][0]
    vertices = data[name][0][0][1]
    normals = data[name][0][0][2]
    return header, faces, vertices, normals

# ----------------------------------------------------------------------------------
# Mesh utilities for GUI and scripts (pure functions)
# ----------------------------------------------------------------------------------

def parse_mesh_from_mat(mat_path: str, key_hint: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load a mesh from a MATLAB .mat file and return dict with 'vertices','faces', optional 'normals'.

    Heuristics:
    - If key_hint provided, try that struct first; else use first non-meta key.
    - Accept 1x1 struct with dtype-named fields or dict-like arrays.
    """
    data = loadmat(mat_path)
    keys = [k for k in data.keys() if not k.startswith('__')]
    if not keys:
        raise ValueError("No data keys found in MAT file")
    root_key = key_hint if key_hint in data else keys[0]
    root = data[root_key]
    try:
        node = root[0][0]
    except Exception:
        node = root
    vertices = faces = normals = None
    if hasattr(node, 'dtype') and getattr(node.dtype, 'names', None):
        names = list(node.dtype.names)
        names_lower = [n.lower() for n in names]
        def _get(cands):
            for nm in cands:
                if nm in names_lower:
                    idx = names_lower.index(nm)
                    val = node[names[idx]]
                    return val if getattr(val, 'ndim', 0) != 0 else val.item()
            return None
        vertices = _get(['vertices','vertex','verts','v'])
        faces = _get(['faces','triangles','f'])
        normals = _get(['normals','normal','n'])
    elif isinstance(root, dict):
        vertices = root.get('vertices', root.get('V'))
        faces = root.get('faces', root.get('F'))
        normals = root.get('normals', root.get('N'))
    if vertices is None or faces is None:
        raise ValueError("MAT file missing 'vertices' or 'faces'")
    out: Dict[str, np.ndarray] = {
        'vertices': np.asarray(vertices),
        'faces': np.asarray(faces)
    }
    if normals is not None:
        out['normals'] = np.asarray(normals)
    return out

def normalize_faces(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Convert faces to zero-based int indices; fix 1-based MATLAB indices and clamp if needed."""
    f = np.asanyarray(faces)[:, :3].astype(np.int64)
    if f.size == 0:
        return f
    fmin = int(np.min(f)); fmax = int(np.max(f))
    if fmin >= 1 and fmax <= n_vertices:
        f = f - 1
    elif fmax >= n_vertices:
        maybe = f - 1
        if int(np.max(maybe)) < n_vertices and int(np.min(maybe)) >= 0:
            f = maybe
        else:
            f = np.clip(f, 0, n_vertices-1)
    return f

def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    v = np.asarray(vertices)
    f = np.asarray(faces)[:, :3].astype(np.int64)
    v0 = v[f[:,0]]; v1 = v[f[:,1]]; v2 = v[f[:,2]]
    nf = np.cross(v1 - v0, v2 - v0)
    nf /= (np.linalg.norm(nf, axis=1, keepdims=True) + 1e-12)
    nv = np.zeros_like(v)
    for i, tri in enumerate(f):
        nv[tri[0]] += nf[i]
        nv[tri[1]] += nf[i]
        nv[tri[2]] += nf[i]
    nv /= (np.linalg.norm(nv, axis=1, keepdims=True) + 1e-12)
    return nv

def inflate_surface(vertices: np.ndarray, normals: np.ndarray, dist: float) -> np.ndarray:
    """Inflate vertices along normals by a constant distance (mm)."""
    v = np.asarray(vertices)
    n = np.asarray(normals)
    n_unit = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return v + n_unit * dist

def outward_angles(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Angle (deg) between per-vertex normal and outward radial direction from centroid."""
    v = np.asarray(vertices)
    n = np.asarray(normals)
    c = v.mean(axis=0)
    dout = v - c
    dout /= (np.linalg.norm(dout, axis=1, keepdims=True) + 1e-12)
    n_unit = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    cosang = np.clip(np.sum(n_unit * dout, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# ----------------------------------------------------------------------------------
# Angle binning and palette helpers
# ----------------------------------------------------------------------------------

def angle_bins() -> List[int]:
    """Standard angle thresholds for legend/binning (degrees)."""
    return [30, 60, 90, 120, 150, 180]

def get_palette(name: str = 'Muted') -> List[str]:
    """Return 7-color palette (index 0 is base/NaN, indexes 1..6 for bins)."""
    name_l = (name or 'Muted').strip().lower()
    palette_muted = ['#8C8C8C','#C14343','#C27A44','#C2AD44','#87B640','#44C29C','#446AC2']
    palette_contrast = ['#8C8C8C','#C23939','#B6C238','#C26138','#38C256','#38C0C2','#3858C2']
    return palette_muted if name_l == 'muted' else palette_contrast

def legend_labels_from_bins(bins: List[int]) -> List[str]:
    """Build human-friendly angle range labels aligned with bins."""
    labels: List[str] = []
    prev = 0
    for b in bins:
        if prev == 0:
            labels.append(f"≤ {int(b)}°")
        else:
            labels.append(f"{int(prev)}–{int(b)}°")
        prev = b
    return labels

def map_angles_to_colors(vals: np.ndarray, bins: List[int], palette: List[str]) -> List[str]:
    """Map each angle value to a hex color using thresholds and palette."""
    colors: List[str] = []
    for a in np.asarray(vals):
        c = palette[0]
        if not np.isnan(a):
            for i, th in enumerate(bins, start=1):
                if a <= th:
                    c = palette[min(i, len(palette)-1)]
                    break
            else:
                c = palette[-1]
        colors.append(c)
    return colors

def uniform_offset(vertices: np.ndarray, normals: np.ndarray, offset: float) -> np.ndarray:
    out = vertices.copy()
    for i in range(normals.shape[0]):
        n = normals[i]
        if not np.isnan(n).any() and np.linalg.norm(n) > 0:
            out[i] = out[i] + (n/np.linalg.norm(n))*offset
    return out

def recenter(vertices: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.mean(reference, axis=0)
    shifted = vertices - center
    return shifted, center

def get_rotmat(alpha: float, beta: float, gamma: float) -> np.ndarray:
    yaw = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha),  np.cos(alpha), 0],
                    [0,              0,             1]])
    pitch = np.array([[ np.cos(beta), 0,  np.sin(beta)],
                      [ 0,            1,  0],
                      [-np.sin(beta), 0,  np.cos(beta)]])
    roll = np.array([[1, 0, 0],
                     [0,  np.cos(gamma), -np.sin(gamma)],
                     [0,  np.sin(gamma),  np.cos(gamma)]])
    return yaw @ pitch @ roll

def transform_vectorspace(field: np.ndarray, scale: float, magnitude: float, vertices: np.ndarray, normals: np.ndarray, devpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform vertices/normals into leadfield index space using forward rotation.

    Mirrors the behavior in MDPO_visualize_NO.py so that optimization and
    visualization agree exactly:
    - Translate vertices by subtracting device origin (brain-centered mm)
    - Apply forward rotation R = yaw@pitch@roll to both positions and normals
    - Convert positions from mm to voxel units by dividing by `scale`
    - Do NOT flip Z and do NOT add x/y midpoints here (done at indexing)
    - Scale dipole vectors by `magnitude`
    """
    R = get_rotmat(float(devpos[3]), float(devpos[4]), float(devpos[5]))
    dippos = np.asarray(vertices, dtype=float).copy()
    dippos -= np.asarray(devpos[:3], dtype=float)
    dipvec = np.asarray(normals, dtype=float).copy()
    for i in range(dippos.shape[0]):
        dippos[i] = dippos[i] @ R
        dipvec[i] = dipvec[i] @ R
    dippos = dippos * (1.0/float(scale))
    dipvec = dipvec * float(magnitude)
    return dippos.astype(float), dipvec.astype(float)

def device_to_brain_space(devpos: np.ndarray, length_mm: float, brain_center: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute device endpoints in brain space using forward rotation.

    - devpos: [x, y, z, alpha, beta, gamma] in brain coords (radians).
    - length_mm: physical length of device.
    - brain_center: optional centroid to translate by (recentered brain).

    Returns (p0, p1) endpoints as 3D points in brain space.
    """
    pos = np.asarray(devpos[:3], dtype=float)
    alpha, beta, gamma = float(devpos[3]), float(devpos[4]), float(devpos[5])
    R = get_rotmat(alpha, beta, gamma)
    half = length_mm / 2.0
    p0_local = np.array([0.0, 0.0, -half])
    p1_local = np.array([0.0, 0.0,  half])
    base = np.zeros(3) if brain_center is None else np.asarray(brain_center, dtype=float)
    p0 = base + pos + R @ p0_local
    p1 = base + pos + R @ p1_local
    return p0, p1

def trim_data(field: np.ndarray, dippos: np.ndarray, dipvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mark out-of-bounds positions as NaN using reference conventions.

    x,y are allowed to be in [-nx/2, nx/2] prior to adding midpoints;
    z must be within [0, nz].
    """
    nx, ny, nz = int(field.shape[0]), int(field.shape[1]), int(field.shape[2])
    out_pos = np.asarray(dippos, dtype=float).copy()
    out_vec = np.asarray(dipvec, dtype=float).copy()
    half_x = nx // 2
    half_y = ny // 2
    for i in range(out_pos.shape[0]):
        p = out_pos[i]
        if (
            np.isnan(p).any() or
            (abs(p[0]) > half_x) or (abs(p[1]) > half_y) or
            (p[2] < 0) or (p[2] > nz)
        ):
            out_pos[i] = np.array([np.nan, np.nan, np.nan], dtype=float)
    return out_pos, out_vec

def calculate_voltage(field: np.ndarray, dippos: np.ndarray, dipvec: np.ndarray, v_scale: float, noise: float, bandwidth: float, weights: Optional[np.ndarray] = None, montage: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute voltage, SNR^2, and information capacity per vertex.

    Indexing adds x/y midpoints at lookup time (consistent with reference).
    """
    nx, ny, nz = int(field.shape[0]), int(field.shape[1]), int(field.shape[2])
    n_vertices = int(dippos.shape[0])
    n_electrodes = int(field.shape[-1])
    V = np.full((n_vertices, n_electrodes), np.nan, dtype=float)
    out = np.full(n_vertices, np.nan, dtype=float)

    for i in range(n_vertices):
        p = dippos[i]
        v = dipvec[i]
        if np.isnan(p).any():
            continue
        x = int(p[0] + nx//2)
        y = int(p[1] + ny//2)
        z = int(p[2])
        if (x < 0) or (x >= nx) or (y < 0) or (y >= ny) or (z < 0) or (z >= nz):
            continue
        for e in range(n_electrodes):
            lf_vec = field[x, y, z, :, e]
            if not np.isnan(lf_vec).any():
                V[i, e] = float(np.dot(lf_vec, v))

        row = V[i]
        if montage:
            out[i] = np.nansum(np.abs(row))
        else:
            out[i] = np.nanmax(np.abs(row))

    out = out * float(v_scale)
    snr = out / float(noise) if float(noise) != 0 else np.zeros_like(out)
    snr = np.nan_to_num(snr)
    snr2 = np.square(snr)
    ic = float(bandwidth) * np.log2(1.0 + snr2)
    ic = np.nan_to_num(ic)

    if weights is not None:
        out = out * weights
        snr2 = snr2 * weights
        ic = ic * weights

    return out, snr2, ic

# ----------------------------------------------------------------------------------
# Data class for project encapsulation
# ----------------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    method: str
    best_solution: np.ndarray
    best_fitness: float
    fitness_history: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SEPIOResult:
    """Container for SEPIO Monte-Carlo evaluation results.

    Stores label, linkage to optimization label, settings used, and averaged arrays.
    """
    label: str
    opt_label: str
    n_devices: int
    settings: Dict[str, Any]
    coefs_avg: np.ndarray
    accs_avg: np.ndarray
    saccs_avg: Optional[np.ndarray]
    sensor_range: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class Project:
    name: str = "Untitled"
    data_folder: str = ""
    brain_file: str = ""  # path to whole brain .mat
    roi_files: List[str] = field(default_factory=list)
    roi_labels: List[str] = field(default_factory=list)
    leadfield_files: List[str] = field(default_factory=list)
    device_names: List[str] = field(default_factory=list)
    device_counts: List[int] = field(default_factory=list)  # per device type
    # Per-device-type physical geometry (used for visualization and constraints)
    device_lengths: List[float] = field(default_factory=list)  # mm
    device_radii: List[float] = field(default_factory=list)   # mm
    measure: str = "ic"  # voltage|snr|ic
    method: str = "genetic"  # genetic|anneal|multianneal|gradient|mSGD|brute
    montage: bool = False
    noise: List[float] = field(default_factory=list)  # per device type
    bandwidth: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)  # voxel scale per device
    dipole_offset: float = 0.5
    cortical_thickness: float = 2.5
    magnitude: float = 0.5e-9
    angle_limit_rad: List[float] = field(default_factory=list)  # per device type
    results: Optional[OptimizationResult] = None
    # Keep a history of all completed optimizations
    results_history: List[OptimizationResult] = field(default_factory=list)
    # Assessment artifacts now stored as list of dicts: {type, timestamp, path}
    assessment_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    # SEPIO: list of saved Monte-Carlo result entries
    sepio_results: List[SEPIOResult] = field(default_factory=list)
    # Plotting: selected SEPIO result labels for cross-dataset plots
    plotting_datasets: List[str] = field(default_factory=list)

    # Annealing settings
    anneal_iterations: int = 35
    anneal_itemp: float = 100.0
    anneal_ftemp: float = 1e-3
    anneal_cooling_rate: float = 0.6
    anneal_cart_step: float = 15.0
    anneal_rot_step: float = np.pi/3
    # Multi-anneal (restarts)
    multi_anneal_restarts: int = 5

    # Gradient descent settings
    gradient_iterations: int = 200
    gradient_cart_step: float = 10.0
    gradient_rot_step: float = np.pi/3
    gradient_decay: float = 0.99
    gradient_simultaneous: int = -1  # -1 for all axes

    # Branch-and-bound (multi-start zoom) settings
    branch_iterations: int = 36
    branch_instances: int = 24
    branch_top: int = 6
    branch_angle_step: float = np.pi/4
    branch_cart_step: float = 8.0
    branch_threshold: float = 0.1
    branch_decay: float = 0.95

    # Brute force settings
    brute_limit: int = 10000
    brute_batch: int = 10

    # Cached geometry for assessment (not always serialized due to size)
    roi_recentered: Optional[np.ndarray] = None
    brain_faces: Optional[np.ndarray] = None
    brain_vertices: Optional[np.ndarray] = None
    roi_faces: Optional[np.ndarray] = None
    roi_vertices_full: Optional[np.ndarray] = None

    # Global constraint parameters (adapted from original multi-device script)
    depth_limits: List[Tuple[float, float]] = field(default_factory=lambda: [(np.nan, np.nan)])  # per device type [min,max]
    cl_wd: List[float] = field(default_factory=list)  # clearance width (diameter) per device type
    cl_d: List[float] = field(default_factory=list)   # forward (deep) clearance along device axis
    cl_back: List[float] = field(default_factory=list)  # backward clearance (backend)
    cl_offset: float = 3.0  # minimum separation between clearance zones
    do_depth: bool = True
    do_proximity: bool = True
    # Per-device proximity limits (minimum separation baseline); if empty falls back to cl_offset
    proximity_limits: List[float] = field(default_factory=list)

    # Structural Avoidance: files and demo settings
    structural_files: List[str] = field(default_factory=list)
    structural_demo_enabled: bool = False
    structural_demo_radius_mm: float = 5.0
    # Optional cached avoidance array [K,4] with rows [x,y,z,r]
    structural_avoidance: Optional[np.ndarray] = None

    # Genetic settings
    num_generations: int = 30
    num_parents_mating: int = 15
    sol_per_pop: int = 44
    process_count: int = 4
    ga_mutation_prob: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.results is not None:
            d['results']['best_solution'] = self.results.best_solution.tolist()
        # Serialize history (convert numpy arrays to lists)
        if self.results_history:
            d['results_history'] = [{
                'method': r.method,
                'best_solution': r.best_solution.tolist(),
                'best_fitness': r.best_fitness,
                'fitness_history': r.fitness_history,
                'meta': r.meta
            } for r in self.results_history]
        # Serialize SEPIO results (arrays to lists)
        if getattr(self, 'sepio_results', None):
            ser_list = []
            for s in self.sepio_results:
                ser_list.append({
                    'label': s.label,
                    'opt_label': s.opt_label,
                    'n_devices': int(s.n_devices),
                    'settings': dict(s.settings or {}),
                    'coefs_avg': None if s.coefs_avg is None else np.asarray(s.coefs_avg).tolist(),
                    'accs_avg': None if s.accs_avg is None else np.asarray(s.accs_avg).tolist(),
                    'saccs_avg': None if s.saccs_avg is None else np.asarray(s.saccs_avg).tolist(),
                    'sensor_range': None if s.sensor_range is None else np.asarray(s.sensor_range).tolist(),
                    'timestamp': s.timestamp,
                })
            d['sepio_results'] = ser_list
        if self.roi_recentered is not None:
            d['roi_recentered'] = self.roi_recentered.tolist()
        # Large arrays (faces/vertices) optionally serialized
        if self.brain_faces is not None:
            d['brain_faces'] = self.brain_faces.tolist()
        if self.brain_vertices is not None:
            d['brain_vertices'] = self.brain_vertices.tolist()
        if self.roi_faces is not None:
            d['roi_faces'] = self.roi_faces.tolist()
        if self.roi_vertices_full is not None:
            d['roi_vertices_full'] = self.roi_vertices_full.tolist()
        return d

    def save(self, path_out: str) -> None:
        with open(path_out, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    @staticmethod
    def load(path_in: str) -> 'Project':
        with open(path_in, 'rb') as f:
            data = pickle.load(f)
        if 'results' in data and data['results'] is not None:
            res = data['results']
            data['results'] = OptimizationResult(
                method=res['method'],
                best_solution=np.array(res['best_solution']),
                best_fitness=res['best_fitness'],
                fitness_history=res.get('fitness_history', []),
                meta=res.get('meta', {})
            )
        if 'roi_recentered' in data and data['roi_recentered'] is not None:
            data['roi_recentered'] = np.array(data['roi_recentered'])
        for k in ['brain_faces','brain_vertices','roi_faces','roi_vertices_full']:
            if k in data and data[k] is not None:
                data[k] = np.array(data[k])
        # Rehydrate structural avoidance array if present
        if 'structural_avoidance' in data and data['structural_avoidance'] is not None:
            data['structural_avoidance'] = np.array(data['structural_avoidance'])
        # Deserialize history
        if 'results_history' in data and data['results_history'] is not None:
            hist_list = []
            for r in data['results_history']:
                hist_list.append(OptimizationResult(
                    method=r['method'],
                    best_solution=np.array(r['best_solution']),
                    best_fitness=r['best_fitness'],
                    fitness_history=r.get('fitness_history', []),
                    meta=r.get('meta', {})
                ))
            data['results_history'] = hist_list
        # Deserialize SEPIO results if present
        if 'sepio_results' in data and data['sepio_results'] is not None:
            sepio_list = []
            for s in data['sepio_results']:
                sepio_list.append(SEPIOResult(
                    label=s.get('label',''),
                    opt_label=s.get('opt_label',''),
                    n_devices=int(s.get('n_devices', 0)),
                    settings=dict(s.get('settings', {})),
                    coefs_avg=np.array(s['coefs_avg']) if s.get('coefs_avg', None) is not None else np.array([]),
                    accs_avg=np.array(s['accs_avg']) if s.get('accs_avg', None) is not None else np.array([]),
                    saccs_avg=None if s.get('saccs_avg', None) is None else np.array(s['saccs_avg']),
                    sensor_range=np.array(s['sensor_range']) if s.get('sensor_range', None) is not None else np.array([], dtype=int),
                    timestamp=s.get('timestamp', datetime.datetime.now().isoformat())
                ))
            data['sepio_results'] = sepio_list
        return Project(**data)

    # ---------------------------------------------------------------
    # Results import/export and assessment artifact management
    # ---------------------------------------------------------------
    @staticmethod
    def load_legacy_result_pickle(path_in: str) -> OptimizationResult:
        """Load legacy optimization result pickle formats and return OptimizationResult.

        Expected keys (legacy):
        - 'best solution:' : array-like (flattened devices)
        - 'best solution fitness' : float
        - optional 'fitnesses' : list[float]
        - optional 'type' : str or [str]
        Fallbacks attempt more modern keys if present.
        """
        with open(path_in, 'rb') as f:
            data = pickle.load(f)
        method = 'unknown'
        best = None
        fitness = None
        history: List[float] = []
        if isinstance(data, dict):
            # Modern-style keys fallback
            if 'best_solution' in data and 'best_fitness' in data:
                best = data['best_solution']
                fitness = data['best_fitness']
                history = data.get('fitness_history', [])
                method = data.get('method', 'unknown')
            # Legacy keys
            if best is None and 'best solution:' in data and 'best solution fitness' in data:
                best = data['best solution:']
                fitness = data['best solution fitness']
                history = data.get('fitnesses', [])
                t = data.get('type', 'unknown')
                if isinstance(t, (list, tuple)) and len(t):
                    method = t[0]
                elif isinstance(t, str):
                    method = t
        if best is None or fitness is None:
            raise ValueError("Unsupported result pickle format; missing best solution or fitness")
        return OptimizationResult(method=str(method), best_solution=np.array(best), best_fitness=float(fitness), fitness_history=list(history), meta={})

    def import_result_pickle(self, path_in: str) -> OptimizationResult:
        """Set self.results from a legacy or modern result pickle and return it."""
        res = Project.load_legacy_result_pickle(path_in)
        self.results = res
        return res

    def add_assessment_artifact(self, artifact_type: str, path_out: str) -> None:
        """Record an assessment artifact path with timestamp and type label."""
        self.assessment_artifacts.append({
            'type': artifact_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'path': path_out
        })

    # ----------------------------------------------------------------------------------
    # Core computational pipeline
    # ----------------------------------------------------------------------------------
    def run_genetic_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if not PYGAD_AVAILABLE:
            raise RuntimeError("pygad not installed; cannot run genetic optimization.")
        if FieldImporter is None:
            raise RuntimeError("Leadfield importer unavailable.")
        # Load leadfields
        importer = FieldImporter()
        field_list = []
        num_electrodes = []
        midpoints = []
        for lf in self.leadfield_files:
            field = importer.load(lf)
            field_arr = importer.fields
            field_list.append(field_arr)
            num_electrodes.append(field_arr.shape[4])
            midpoints.append([field_arr.shape[0]//2, field_arr.shape[1]//2, field_arr.shape[2]//2])
        # Load brain + ROI surface meshes
        brain_mat = loadmat(self.brain_file)
        _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_mat, 'brain')
        self.brain_faces = brain_faces
        self.brain_vertices = brain_vertices
        roi_vertices_all = []
        roi_normals_all = []
        roi_weights = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, _, v, n = obtain_data(mat, label)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
            roi_weights.append(np.ones(v.shape[0]))
        roi_vertices = np.vstack(roi_vertices_all)
        roi_normals = np.vstack(roi_normals_all)
        roi_faces_list = []
        # Faces for each ROI are needed; gather from each dataset
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, faces_tmp, _, _ = obtain_data(mat, label)
            roi_faces_list.append(faces_tmp)
        if len(roi_faces_list):
            self.roi_faces = np.vstack(roi_faces_list)
        self.roi_vertices_full = roi_vertices
        weights = np.concatenate(roi_weights)
        # Dipole offset to approximate cortical depth
        roi_vertices_depth = uniform_offset(roi_vertices, roi_normals, self.dipole_offset * self.cortical_thickness)
        recentered_roi, center = recenter(roi_vertices_depth, brain_vertices)
        self.roi_recentered = recentered_roi  # cache for assessment
        # Device indexing
        N_per_type = self.device_counts
        total_N = int(np.sum(N_per_type))
        N_index = []
        for i, count in enumerate(N_per_type):
            N_index.extend([i]*count)
        # Measure index mapping
        measure_map = {'voltage': 0, 'snr': 1, 'ic': 2}
        measure_idx = measure_map.get(self.measure, 2)
        # Initial population
        num_genes = total_N * 6
        min_values = np.min(recentered_roi, axis=0)
        max_values = np.max(recentered_roi, axis=0)
        gene_space = [[min_values[0], max_values[0]],
                      [min_values[1], max_values[1]],
                      [min_values[2], max_values[2]],
                      [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]] * total_N
        initial_population = self._generate_initial_population(recentered_roi, total_N, self.sol_per_pop)
        initial_population = np.round(initial_population.reshape(self.sol_per_pop, num_genes), 2)
        # Fitness accumulation lists
        fitness_history: List[float] = []
        best_solutions_per_epoch: List[List[float]] = []

        def transform_and_score(solution: np.ndarray) -> float:
            sol = solution.reshape((total_N, 6))
            # Apply constraints (angle/depth/proximity) before scoring
            sol = self.limit_correction(sol)
            all_dev_vals = np.empty((recentered_roi.shape[0], total_N))
            for d in range(total_N):
                t_idx = N_index[d]
                field = field_list[t_idx]
                dippos, dipvec = transform_vectorspace(field, self.scale[t_idx], self.magnitude, recentered_roi, roi_normals, sol[d])
                dippos_t, dipvec_t = trim_data(field, dippos, dipvec)
                vals = calculate_voltage(field, dippos_t, dipvec_t, v_scale=1e6, noise=self.noise[t_idx], bandwidth=self.bandwidth[t_idx], weights=weights, montage=self.montage)
                all_dev_vals[:, d] = vals[measure_idx]
            combined = np.nanmax(np.nan_to_num(all_dev_vals, nan=0.0), axis=1)
            return float(np.nansum(combined))

        def fitness_func(ga_inst, solution, idx):  # pygad interface
            return transform_and_score(solution)

        best_solutions_per_epoch: List[List[float]] = []
        def on_gen_func(ga_inst):
            best_tuple = ga_inst.best_solution()
            best = best_tuple[1]
            fitness_history.append(best)
            # Record corrected trajectory (flattened) for export
            try:
                raw = np.asarray(best_tuple[0]).astype(float)
                corr = self.limit_correction(raw.reshape((total_N, 6))).reshape(-1)
                best_solutions_per_epoch.append(corr.astype(float).tolist())
            except Exception:
                pass
            gen = ga_inst.generations_completed
            if progress_callback is not None:
                try:
                    progress_callback(gen, float(best), self.num_generations)
                except Exception:
                    pass
            #print(f"Generation {gen} Best {best:.2f}")

        # Use threads for parallelism to avoid pickling local fitness_func closures.
        # PyGAD's process-based executor requires the fitness function to be picklable.
        # Thread-based parallelism works here since the heavy work releases the GIL in NumPy.
        ga = pygad.GA(num_generations=self.num_generations,
                      num_parents_mating=self.num_parents_mating,
                      fitness_func=fitness_func,
                      sol_per_pop=self.sol_per_pop,
                      num_genes=num_genes,
                      initial_population=initial_population,
                      gene_space=gene_space,
                      parent_selection_type="sss",
                      keep_elitism=1,
                      on_generation=on_gen_func,
                      parallel_processing=['thread', self.process_count],
                      mutation_probability=self.ga_mutation_prob)
        assert pygad is not None  # runtime guarantee for linter
        start = time.time()
        # Reader note: See line line 2003
        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()
        # Ensure saved best solution is limit-corrected
        try:
            best_solution = self.limit_correction(np.asarray(best_solution).reshape((total_N, 6))).reshape(-1)
        except Exception:
            best_solution = np.asarray(best_solution)
        elapsed = time.time() - start
        meta = {
            'elapsed_sec': elapsed,
            'device_counts': self.device_counts,
            'measure': self.measure,
            'method': 'genetic',
            'timestamp': datetime.datetime.now().isoformat(),
            # Shorthand GA parameters for labeling
            'num_generations': self.num_generations,
            'sol_per_pop': self.sol_per_pop,
            'num_parents_mating': self.num_parents_mating,
            'process_count': self.process_count,
            'mutation_probability': self.ga_mutation_prob,
            # Per-epoch best solution trajectory for export
            'best_solutions_per_epoch': best_solutions_per_epoch
        }
        result = OptimizationResult(method='genetic', best_solution=np.array(best_solution), best_fitness=float(best_fitness), fitness_history=fitness_history, meta=meta)
        self.results = result
        self.results_history.append(result)

    def export_epoch_history_csv(self, path_out: str) -> None:
        """Export per-epoch results to CSV: epoch, fitness, and full trajectory components (N*6).
        For anneal: exports fitness per epoch and fills trajectory columns with NaN.
        For genetic: uses fitness_history and meta['best_solutions_per_epoch'].
        """
        if self.results is None:
            raise RuntimeError("No results to export.")
        fitness = self.results.fitness_history or []
        traj: List[List[float]] = []
        # For both genetic and anneal, use stored per-epoch trajectories if available
        traj = self.results.meta.get('best_solutions_per_epoch', []) or []
        # Determine number of trajectory components
        num_components = 0
        if traj and len(traj) and isinstance(traj[0], (list, tuple)):
            num_components = len(traj[0])
        else:
            # Fallback to device_counts if available
            try:
                num_components = int(np.sum(self.device_counts)) * 6
            except Exception:
                num_components = 0
        import csv
        with open(path_out, 'w', newline='') as f:
            w = csv.writer(f)
            # Header: epoch, fitness, comp1..compM
            header = ['epoch', 'fitness'] + [f'comp{j+1}' for j in range(num_components)]
            w.writerow(header)
            for i, fit in enumerate(fitness):
                row = [i+1, float(fit)]
                if num_components > 0:
                    if traj and i < len(traj):
                        vec = traj[i]
                        # pad or trim to num_components
                        comps = [vec[j] if j < len(vec) else np.nan for j in range(num_components)]
                    else:
                        comps = [np.nan] * num_components
                    row += comps
                w.writerow(row)

    # ------------------------------------------------------------------
    # Simulated Annealing Optimization (simplified port)
    # ------------------------------------------------------------------
    def run_anneal_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if FieldImporter is None:
            raise RuntimeError("Leadfield importer unavailable.")
        importer = FieldImporter()
        field_list = []
        for lf in self.leadfield_files:
            field = importer.load(lf)
            field_list.append(importer.fields)
        brain_mat = loadmat(self.brain_file)
        _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_mat, 'brain')
        self.brain_faces = brain_faces
        self.brain_vertices = brain_vertices
        roi_vertices_all = []
        roi_normals_all = []
        roi_weights = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, _, v, n = obtain_data(mat, label)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
            roi_weights.append(np.ones(v.shape[0]))
        roi_vertices = np.vstack(roi_vertices_all)
        roi_normals = np.vstack(roi_normals_all)
        roi_faces_list = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, faces_tmp, _, _ = obtain_data(mat, label)
            roi_faces_list.append(faces_tmp)
        if len(roi_faces_list):
            self.roi_faces = np.vstack(roi_faces_list)
        self.roi_vertices_full = roi_vertices
        weights = np.concatenate(roi_weights)
        roi_vertices_depth = uniform_offset(roi_vertices, roi_normals, self.dipole_offset * self.cortical_thickness)
        recentered_roi, center = recenter(roi_vertices_depth, brain_vertices)
        self.roi_recentered = recentered_roi
        N_per_type = self.device_counts
        total_N = int(np.sum(N_per_type))
        N_index = []
        for i, count in enumerate(N_per_type):
            N_index.extend([i]*count)
        measure_map = {'voltage': 0, 'snr': 1, 'ic': 2}
        measure_idx = measure_map.get(self.measure, 2)
        # Bounds for spatial coordinates based on ROI envelope
        min_values = np.min(recentered_roi, axis=0)
        max_values = np.max(recentered_roi, axis=0)

        def score(solution_flat: np.ndarray) -> float:
            sol = solution_flat.reshape((total_N,6))
            # Clamp spatial coordinates inside ROI bounds and wrap angles
            for d in range(total_N):
                sol[d, 0] = float(np.clip(sol[d, 0], min_values[0], max_values[0]))
                sol[d, 1] = float(np.clip(sol[d, 1], min_values[1], max_values[1]))
                sol[d, 2] = float(np.clip(sol[d, 2], min_values[2], max_values[2]))
                sol[d, 3:6] = np.clip(sol[d, 3:6], -np.pi, np.pi)
            sol = self.limit_correction(sol)
            all_dev_vals = np.empty((recentered_roi.shape[0], total_N))
            for d in range(total_N):
                t_idx = N_index[d]
                field = field_list[t_idx]
                dippos, dipvec = transform_vectorspace(field, self.scale[t_idx], self.magnitude, recentered_roi, roi_normals, sol[d])
                dippos_t, dipvec_t = trim_data(field, dippos, dipvec)
                vals = calculate_voltage(field, dippos_t, dipvec_t, v_scale=1e6, noise=self.noise[t_idx], bandwidth=self.bandwidth[t_idx], weights=weights, montage=self.montage)
                all_dev_vals[:, d] = vals[measure_idx]
            combined = np.nanmax(np.nan_to_num(all_dev_vals, nan=0.0), axis=1)
            return float(np.nansum(combined))

        # Unified initialization: generate population and choose best as single start
        init_pop = self._generate_initial_population(recentered_roi, total_N, int(self.sol_per_pop))
        init_pop = init_pop.reshape(int(self.sol_per_pop), total_N*6)
        scores = [score(row) for row in init_pop]
        best_idx = int(np.argmax(scores))
        # Initialize with limit-corrected seed
        current = self.limit_correction(init_pop[best_idx].copy()).reshape(-1)
        current_score = float(scores[best_idx])
        temperature = self.anneal_itemp
        fitness_history: List[float] = [current_score]
        best_solution = current.copy()
        best_solutions_per_epoch: List[List[float]] = [best_solution.tolist()]
        best_score = current_score
        min_values = np.min(recentered_roi, axis=0)
        max_values = np.max(recentered_roi, axis=0)

        # Pre-compute total epoch estimate for progress bar (cooling steps * iterations)
        if self.anneal_cooling_rate <= 0 or self.anneal_cooling_rate >= 1:
            # Fallback: treat each iteration as single epoch
            estimated_cooling_steps = 1
        else:
            estimated_cooling_steps = int(np.ceil(np.log(self.anneal_ftemp / self.anneal_itemp) / np.log(self.anneal_cooling_rate)))
            estimated_cooling_steps = max(1, estimated_cooling_steps)
        total_epochs_est = estimated_cooling_steps * self.anneal_iterations
        anneal_epoch = 0
        while temperature > self.anneal_ftemp:
            # Generate trial solutions
            for _ in range(self.anneal_iterations):
                trial = current.copy()
                # Cartesian perturb
                for d in range(total_N):
                    base = d*6
                    trial[base:base+3] += np.random.uniform(-1,1,3) * self.anneal_cart_step * (temperature/self.anneal_itemp)
                    trial[base+3:base+6] += np.random.uniform(-1,1,3) * self.anneal_rot_step * (temperature/self.anneal_itemp)
                    # Clamp spatial coords into ROI bounds
                    trial[base] = np.clip(trial[base], min_values[0], max_values[0])
                    trial[base+1] = np.clip(trial[base+1], min_values[1], max_values[1])
                    trial[base+2] = np.clip(trial[base+2], min_values[2], max_values[2])
                    # Clamp angles to [-pi, pi]
                    trial[base+3:base+6] = np.clip(trial[base+3:base+6], -np.pi, np.pi)
                trial_score = score(trial)
                accept = (trial_score > current_score) or (np.random.rand() < np.exp((trial_score - current_score)/max(temperature,1e-9)))
                if accept:
                    # Persist corrected accepted state
                    current = self.limit_correction(trial).reshape(-1)
                    current_score = trial_score
                    if trial_score > best_score:
                        best_score = trial_score
                        best_solution = current.copy()
                fitness_history.append(best_score)
                best_solutions_per_epoch.append(best_solution.tolist())
                anneal_epoch += 1
                if progress_callback is not None:
                    try:
                        progress_callback(anneal_epoch, float(best_score), total_epochs_est)
                    except Exception:
                        pass
            temperature *= self.anneal_cooling_rate
            #print(f"Anneal temp {temperature:.4f} best {best_score:.2f}")

        meta = {
            'elapsed_sec': None,
            'device_counts': self.device_counts,
            'measure': self.measure,
            'method': 'anneal',
            'timestamp': datetime.datetime.now().isoformat(),
            'initial_temp': self.anneal_itemp,
            'final_temp': self.anneal_ftemp,
            'cooling_rate': self.anneal_cooling_rate,
            'iterations': self.anneal_iterations,
            'best_solutions_per_epoch': best_solutions_per_epoch
        }
        result = OptimizationResult(method='anneal', best_solution=best_solution, best_fitness=best_score, fitness_history=fitness_history, meta=meta)
        self.results = result
        self.results_history.append(result)

    # ------------------------------------------------------------------
    # Multi-Anneal Optimization (multiple independent anneal restarts)
    # ------------------------------------------------------------------
    def run_multianneal_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if FieldImporter is None:
            raise RuntimeError("Leadfield importer unavailable.")
        importer = FieldImporter()
        field_list = []
        for lf in self.leadfield_files:
            importer.load(lf)
            field_list.append(importer.fields)
        brain_mat = loadmat(self.brain_file)
        _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_mat, 'brain')
        self.brain_faces = brain_faces
        self.brain_vertices = brain_vertices
        roi_vertices_all = []
        roi_normals_all = []
        roi_weights = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, _, v, n = obtain_data(mat, label)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
            roi_weights.append(np.ones(v.shape[0]))
        roi_vertices = np.vstack(roi_vertices_all)
        roi_normals = np.vstack(roi_normals_all)
        weights = np.concatenate(roi_weights)
        roi_vertices_depth = uniform_offset(roi_vertices, roi_normals, self.dipole_offset * self.cortical_thickness)
        recentered_roi, center = recenter(roi_vertices_depth, brain_vertices)
        self.roi_recentered = recentered_roi
        N_per_type = self.device_counts
        total_N = int(np.sum(N_per_type))
        N_index = []
        for i, count in enumerate(N_per_type):
            N_index.extend([i]*count)
        measure_map = {'voltage': 0, 'snr': 1, 'ic': 2}
        measure_idx = measure_map.get(self.measure, 2)
        min_values = np.min(recentered_roi, axis=0)
        max_values = np.max(recentered_roi, axis=0)

        def score(solution_flat: np.ndarray) -> float:
            sol = solution_flat.reshape((total_N,6))
            for d in range(total_N):
                sol[d, 0] = float(np.clip(sol[d, 0], min_values[0], max_values[0]))
                sol[d, 1] = float(np.clip(sol[d, 1], min_values[1], max_values[1]))
                sol[d, 2] = float(np.clip(sol[d, 2], min_values[2], max_values[2]))
                sol[d, 3:6] = np.clip(sol[d, 3:6], -np.pi, np.pi)
            sol = self.limit_correction(sol)
            all_dev_vals = np.empty((recentered_roi.shape[0], total_N))
            for d in range(total_N):
                t_idx = N_index[d]
                field = field_list[t_idx]
                dippos, dipvec = transform_vectorspace(field, self.scale[t_idx], self.magnitude, recentered_roi, roi_normals, sol[d])
                dippos_t, dipvec_t = trim_data(field, dippos, dipvec)
                vals = calculate_voltage(field, dippos_t, dipvec_t, v_scale=1e6, noise=self.noise[t_idx], bandwidth=self.bandwidth[t_idx], weights=weights, montage=self.montage)
                all_dev_vals[:, d] = vals[measure_idx]
            combined = np.nanmax(np.nan_to_num(all_dev_vals, nan=0.0), axis=1)
            return float(np.nansum(combined))

        # Estimate cooling steps for progress accounting
        if self.anneal_cooling_rate <= 0 or self.anneal_cooling_rate >= 1:
            cooling_steps = 1
        else:
            cooling_steps = int(np.ceil(np.log(self.anneal_ftemp / self.anneal_itemp) / np.log(self.anneal_cooling_rate)))
            cooling_steps = max(1, cooling_steps)
        epochs_per_restart = cooling_steps * self.anneal_iterations
        total_epochs = epochs_per_restart * max(1, self.multi_anneal_restarts)

        global_best_score = -np.inf
        global_best_solution: Optional[np.ndarray] = None
        combined_history: List[float] = []
        combined_trajectory: List[List[float]] = []
        epoch_counter = 0

        for restart in range(max(1, self.multi_anneal_restarts)):
            # Unified initialization per restart: best-of-population seed
            init_pop = self._generate_initial_population(recentered_roi, total_N, int(self.sol_per_pop))
            init_pop = init_pop.reshape(int(self.sol_per_pop), total_N*6)
            scores = [score(row) for row in init_pop]
            best_idx = int(np.argmax(scores))
            # Initialize restart with limit-corrected seed
            current = self.limit_correction(init_pop[best_idx].copy()).reshape(-1)
            current_score = float(scores[best_idx])
            best_solution = current.copy()
            best_score = current_score
            temperature = self.anneal_itemp
            while temperature > self.anneal_ftemp:
                for _ in range(self.anneal_iterations):
                    trial = current.copy()
                    for d in range(total_N):
                        base_i = d*6
                        trial[base_i:base_i+3] += np.random.uniform(-1,1,3) * self.anneal_cart_step * (temperature/self.anneal_itemp)
                        trial[base_i+3:base_i+6] += np.random.uniform(-1,1,3) * self.anneal_rot_step * (temperature/self.anneal_itemp)
                        trial[base_i]   = np.clip(trial[base_i],   min_values[0], max_values[0])
                        trial[base_i+1] = np.clip(trial[base_i+1], min_values[1], max_values[1])
                        trial[base_i+2] = np.clip(trial[base_i+2], min_values[2], max_values[2])
                        trial[base_i+3:base_i+6] = np.clip(trial[base_i+3:base_i+6], -np.pi, np.pi)
                    trial_score = score(trial)
                    accept = (trial_score > current_score) or (np.random.rand() < np.exp((trial_score - current_score)/max(temperature,1e-9)))
                    if accept:
                        current = self.limit_correction(trial).reshape(-1)
                        current_score = trial_score
                        if trial_score > best_score:
                            best_score = trial_score
                            best_solution = current.copy()
                    if best_score > global_best_score:
                        global_best_score = best_score
                        global_best_solution = best_solution.copy()
                    combined_history.append(global_best_score)
                    combined_trajectory.append(best_solution.tolist())
                    epoch_counter += 1
                    if progress_callback is not None:
                        try:
                            progress_callback(epoch_counter, float(global_best_score), total_epochs)
                        except Exception:
                            pass
                temperature *= self.anneal_cooling_rate
        if global_best_solution is None:
            raise RuntimeError("Multi-anneal produced no solution.")
        meta = {
            'elapsed_sec': None,
            'device_counts': self.device_counts,
            'measure': self.measure,
            'method': 'multianneal',
            'timestamp': datetime.datetime.now().isoformat(),
            'iterations': self.anneal_iterations,
            'restarts': self.multi_anneal_restarts,
            'cooling_rate': self.anneal_cooling_rate,
            'initial_temp': self.anneal_itemp,
            'final_temp': self.anneal_ftemp,
            'best_solutions_per_epoch': combined_trajectory
        }
        result = OptimizationResult(method='multianneal', best_solution=global_best_solution, best_fitness=global_best_score, fitness_history=combined_history, meta=meta)
        self.results = result
        self.results_history.append(result)

    # ------------------------------------------------------------------
    # Gradient Descent Optimization (finite-difference, decaying steps)
    # ------------------------------------------------------------------
    def run_gradient_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if FieldImporter is None:
            raise RuntimeError("Leadfield importer unavailable.")
        importer = FieldImporter()
        field_list = []
        for lf in self.leadfield_files:
            importer.load(lf)
            field_list.append(importer.fields)
        brain_mat = loadmat(self.brain_file)
        _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_mat, 'brain')
        self.brain_faces = brain_faces
        self.brain_vertices = brain_vertices
        roi_vertices_all = []
        roi_normals_all = []
        roi_weights = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, _, v, n = obtain_data(mat, label)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
            roi_weights.append(np.ones(v.shape[0]))
        roi_vertices = np.vstack(roi_vertices_all)
        roi_normals = np.vstack(roi_normals_all)
        roi_faces_list = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, faces_tmp, _, _ = obtain_data(mat, label)
            roi_faces_list.append(faces_tmp)
        if len(roi_faces_list):
            self.roi_faces = np.vstack(roi_faces_list)
        self.roi_vertices_full = roi_vertices
        weights = np.concatenate(roi_weights)
        roi_vertices_depth = uniform_offset(roi_vertices, roi_normals, self.dipole_offset * self.cortical_thickness)
        recentered_roi, center = recenter(roi_vertices_depth, brain_vertices)
        self.roi_recentered = recentered_roi
        N_per_type = self.device_counts
        total_N = int(np.sum(N_per_type))
        N_index = []
        for i, count in enumerate(N_per_type):
            N_index.extend([i]*count)
        measure_map = {'voltage': 0, 'snr': 1, 'ic': 2}
        measure_idx = measure_map.get(self.measure, 2)

        def score(solution_flat: np.ndarray) -> float:
            sol = solution_flat.reshape((total_N,6))
            sol = self.limit_correction(sol)
            all_dev_vals = np.empty((recentered_roi.shape[0], total_N))
            for d in range(total_N):
                t_idx = N_index[d]
                field = field_list[t_idx]
                dippos, dipvec = transform_vectorspace(field, self.scale[t_idx], self.magnitude, recentered_roi, roi_normals, sol[d])
                dippos_t, dipvec_t = trim_data(field, dippos, dipvec)
                vals = calculate_voltage(field, dippos_t, dipvec_t, v_scale=1e6, noise=self.noise[t_idx], bandwidth=self.bandwidth[t_idx], weights=weights, montage=self.montage)
                all_dev_vals[:, d] = vals[measure_idx]
            combined = np.nanmax(np.nan_to_num(all_dev_vals, nan=0.0), axis=1)
            return float(np.nansum(combined))

        # Unified initialization: best-of-population as starting point
        init_pop = self._generate_initial_population(recentered_roi, total_N, int(self.sol_per_pop))
        init_pop = init_pop.reshape(int(self.sol_per_pop), total_N*6)
        scores = [score(row) for row in init_pop]
        best_idx = int(np.argmax(scores))
        current = self.limit_correction(init_pop[best_idx].copy()).reshape(-1)
        current_score = float(scores[best_idx])
        best = current.copy()
        best_score = current_score
        fitness_history: List[float] = [best_score]
        cart_step = float(self.gradient_cart_step)
        rot_step = float(self.gradient_rot_step)
        decay = float(self.gradient_decay)
        simultaneous = int(self.gradient_simultaneous)
        eps = 1e-4
        for it in range(self.gradient_iterations):
            # Finite-difference gradient on selected axes
            step = np.zeros_like(current)
            axes = list(range(current.shape[0]))
            if simultaneous == -1 or simultaneous >= len(axes):
                chosen = axes
            else:
                # Randomly choose a subset of axes each iteration
                chosen = list(np.random.choice(axes, size=max(1, simultaneous), replace=False))
            for ax in chosen:
                plus = current.copy(); minus = current.copy()
                base = cart_step if (ax % 6) < 3 else rot_step
                plus[ax] += eps; minus[ax] -= eps
                g = (score(plus) - score(minus)) / (2*eps)
                # Use gradient ascent; scale by base step with sign to stabilize
                step[ax] = base * np.sign(g)
            proposal = current + step
            # Clamp proposal into ROI bounds and angle ranges per-device
            for d in range(total_N):
                base_i = d*6
                proposal[base_i]   = float(np.clip(proposal[base_i],   min_values[0], max_values[0]))
                proposal[base_i+1] = float(np.clip(proposal[base_i+1], min_values[1], max_values[1]))
                proposal[base_i+2] = float(np.clip(proposal[base_i+2], min_values[2], max_values[2]))
                proposal[base_i+3:base_i+6] = np.clip(proposal[base_i+3:base_i+6], -np.pi, np.pi)
            proposal = self.limit_correction(proposal)
            prop_score = score(proposal)
            if prop_score > current_score:
                current = proposal
                current_score = prop_score
                if current_score > best_score:
                    best = current.copy(); best_score = current_score
            fitness_history.append(best_score)
            if progress_callback is not None:
                try:
                    progress_callback(it+1, float(best_score), self.gradient_iterations)
                except Exception:
                    pass
            cart_step *= decay; rot_step *= decay
        meta = {
            'elapsed_sec': None,
            'device_counts': self.device_counts,
            'measure': self.measure,
            'method': 'gradient',
            'timestamp': datetime.datetime.now().isoformat(),
            'iterations': self.gradient_iterations,
        }
        result = OptimizationResult(method='gradient', best_solution=best, best_fitness=best_score, fitness_history=fitness_history, meta=meta)
        self.results = result
        self.results_history.append(result)

    # ------------------------------------------------------------------
    # Branch-and-Bound (multi-start zoom)
    # ------------------------------------------------------------------
    def run_branch_bound_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if FieldImporter is None:
            raise RuntimeError("Leadfield importer unavailable.")
        importer = FieldImporter()
        field_list = []
        for lf in self.leadfield_files:
            importer.load(lf)
            field_list.append(importer.fields)
        brain_mat = loadmat(self.brain_file)
        _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_mat, 'brain')
        self.brain_faces = brain_faces
        self.brain_vertices = brain_vertices
        roi_vertices_all = []
        roi_normals_all = []
        roi_weights = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, _, v, n = obtain_data(mat, label)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
            roi_weights.append(np.ones(v.shape[0]))
        roi_vertices = np.vstack(roi_vertices_all)
        roi_normals = np.vstack(roi_normals_all)
        roi_faces_list = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, faces_tmp, _, _ = obtain_data(mat, label)
            roi_faces_list.append(faces_tmp)
        if len(roi_faces_list):
            self.roi_faces = np.vstack(roi_faces_list)
        self.roi_vertices_full = roi_vertices
        weights = np.concatenate(roi_weights)
        roi_vertices_depth = uniform_offset(roi_vertices, roi_normals, self.dipole_offset * self.cortical_thickness)
        recentered_roi, center = recenter(roi_vertices_depth, brain_vertices)
        self.roi_recentered = recentered_roi
        N_per_type = self.device_counts
        total_N = int(np.sum(N_per_type))
        N_index = []
        for i, count in enumerate(N_per_type):
            N_index.extend([i]*count)
        measure_map = {'voltage': 0, 'snr': 1, 'ic': 2}
        measure_idx = measure_map.get(self.measure, 2)

        def score(solution_flat: np.ndarray) -> float:
            sol = solution_flat.reshape((total_N,6))
            sol = self.limit_correction(sol)
            all_dev_vals = np.empty((recentered_roi.shape[0], total_N))
            for d in range(total_N):
                t_idx = N_index[d]
                field = field_list[t_idx]
                dippos, dipvec = transform_vectorspace(field, self.scale[t_idx], self.magnitude, recentered_roi, roi_normals, sol[d])
                dippos_t, dipvec_t = trim_data(field, dippos, dipvec)
                vals = calculate_voltage(field, dippos_t, dipvec_t, v_scale=1e6, noise=self.noise[t_idx], bandwidth=self.bandwidth[t_idx], weights=weights, montage=self.montage)
                all_dev_vals[:, d] = vals[measure_idx]
            combined = np.nanmax(np.nan_to_num(all_dev_vals, nan=0.0), axis=1)
            return float(np.nansum(combined))

        # Unified initialization: best-of-population seed
        init_pop = self._generate_initial_population(recentered_roi, total_N, int(self.sol_per_pop))
        init_pop = init_pop.reshape(int(self.sol_per_pop), total_N*6)
        scores = [score(row) for row in init_pop]
        best_idx = int(np.argmax(scores))
        # Start from limit-corrected seed
        best = self.limit_correction(init_pop[best_idx].copy()).reshape(-1)
        best_score = float(scores[best_idx])
        fitness_history: List[float] = [best_score]
        angle_step = float(self.branch_angle_step)
        cart_step = float(self.branch_cart_step)
        for it in range(self.branch_iterations):
            instances = []
            for _ in range(self.branch_instances):
                prop = best.copy()
                mask = np.ones_like(prop)
                for d in range(total_N):
                    base_i = d*6
                    mask[base_i:base_i+3] *= cart_step
                    mask[base_i+3:base_i+6] *= angle_step
                prop += np.random.uniform(-1,1, size=prop.shape) * mask
                # Clamp spatial coords into ROI bounds and angles to [-pi,pi]
                for d in range(total_N):
                    base_i = d*6
                    prop[base_i]   = float(np.clip(prop[base_i],   recentered_roi[:,0].min(), recentered_roi[:,0].max()))
                    prop[base_i+1] = float(np.clip(prop[base_i+1], recentered_roi[:,1].min(), recentered_roi[:,1].max()))
                    prop[base_i+2] = float(np.clip(prop[base_i+2], recentered_roi[:,2].min(), recentered_roi[:,2].max()))
                    prop[base_i+3:base_i+6] = np.clip(prop[base_i+3:base_i+6], -np.pi, np.pi)
                prop = self.limit_correction(prop)
                instances.append(prop)
            scores = [score(x) for x in instances]
            order = np.argsort(-np.array(scores))
            top = [instances[i] for i in order[:self.branch_top]]
            top_scores = [scores[i] for i in order[:self.branch_top]]
            if top_scores and top_scores[0] > best_score:
                improve_ratio = (top_scores[0] - best_score)/(abs(best_score)+1e-9)
                best = top[0]
                best_score = top_scores[0]
            fitness_history.append(best_score)
            if progress_callback is not None:
                try:
                    progress_callback(it+1, float(best_score), self.branch_iterations)
                except Exception:
                    pass
            # Decay steps
            cart_step *= float(self.branch_decay)
            angle_step *= float(self.branch_decay)
            # Early stop on small improvement
            if len(fitness_history) > 2:
                delta = fitness_history[-1] - fitness_history[-2]
                if abs(delta)/(abs(fitness_history[-2])+1e-9) < float(self.branch_threshold):
                    # continue few more or break; keep going to complete iterations for consistency
                    pass
        meta = {
            'elapsed_sec': None,
            'device_counts': self.device_counts,
            'measure': self.measure,
            'method': 'branch_bound',
            'timestamp': datetime.datetime.now().isoformat(),
            'iterations': self.branch_iterations,
            'instances': self.branch_instances,
            'top': self.branch_top,
            'angle_step': self.branch_angle_step,
            'cart_step': self.branch_cart_step,
            'threshold': self.branch_threshold,
            'decay': self.branch_decay,
        }
        result = OptimizationResult(method='mSGD', best_solution=best, best_fitness=best_score, fitness_history=fitness_history, meta=meta)
        self.results = result
        self.results_history.append(result)

    # ------------------------------------------------------------------
    # Brute Force (batched evaluation of initial population)
    # ------------------------------------------------------------------
    def run_brute_force_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if FieldImporter is None:
            raise RuntimeError("Leadfield importer unavailable.")
        importer = FieldImporter()
        field_list = []
        for lf in self.leadfield_files:
            importer.load(lf)
            field_list.append(importer.fields)
        brain_mat = loadmat(self.brain_file)
        _, brain_faces, brain_vertices, brain_normals = obtain_data(brain_mat, 'brain')
        self.brain_faces = brain_faces
        self.brain_vertices = brain_vertices
        roi_vertices_all = []
        roi_normals_all = []
        roi_weights = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, _, v, n = obtain_data(mat, label)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
            roi_weights.append(np.ones(v.shape[0]))
        roi_vertices = np.vstack(roi_vertices_all)
        roi_normals = np.vstack(roi_normals_all)
        roi_faces_list = []
        for idx, rf in enumerate(self.roi_files):
            mat = loadmat(rf)
            label = self.roi_labels[idx] if idx < len(self.roi_labels) else 'ans'
            _, faces_tmp, _, _ = obtain_data(mat, label)
            roi_faces_list.append(faces_tmp)
        if len(roi_faces_list):
            self.roi_faces = np.vstack(roi_faces_list)
        self.roi_vertices_full = roi_vertices
        weights = np.concatenate(roi_weights)
        roi_vertices_depth = uniform_offset(roi_vertices, roi_normals, self.dipole_offset * self.cortical_thickness)
        recentered_roi, center = recenter(roi_vertices_depth, brain_vertices)
        self.roi_recentered = recentered_roi
        N_per_type = self.device_counts
        total_N = int(np.sum(N_per_type))
        N_index = []
        for i, count in enumerate(N_per_type):
            N_index.extend([i]*count)
        measure_map = {'voltage': 0, 'snr': 1, 'ic': 2}
        measure_idx = measure_map.get(self.measure, 2)

        def score(solution_flat: np.ndarray) -> float:
            sol = solution_flat.reshape((total_N,6))
            sol = self.limit_correction(sol)
            all_dev_vals = np.empty((recentered_roi.shape[0], total_N))
            for d in range(total_N):
                t_idx = N_index[d]
                field = field_list[t_idx]
                dippos, dipvec = transform_vectorspace(field, self.scale[t_idx], self.magnitude, recentered_roi, roi_normals, sol[d])
                dippos_t, dipvec_t = trim_data(field, dippos, dipvec)
                vals = calculate_voltage(field, dippos_t, dipvec_t, v_scale=1e6, noise=self.noise[t_idx], bandwidth=self.bandwidth[t_idx], weights=weights, montage=self.montage)
                all_dev_vals[:, d] = vals[measure_idx]
            combined = np.nanmax(np.nan_to_num(all_dev_vals, nan=0.0), axis=1)
            return float(np.nansum(combined))

        # Build initial population from ROI points (size = sol_per_pop)
        init_pop = self._generate_initial_population(recentered_roi, total_N, self.sol_per_pop)
        init_pop = init_pop.reshape(self.sol_per_pop, total_N*6)
        limit = int(self.brute_limit)
        batch = max(1, int(self.brute_batch))
        evaluated = 0
        best_score = -np.inf
        best_sol = None
        fitness_history: List[float] = []
        total_to_eval = min(limit, init_pop.shape[0])
        for i in range(0, total_to_eval, batch):
            chunk = init_pop[i:i+batch]
            for row in chunk:
                sc = score(row)
                fitness_history.append(sc)
                if sc > best_score:
                    best_score = sc
                    # Persist corrected solution
                    best_sol = self.limit_correction(row).reshape(-1).copy()
            evaluated += chunk.shape[0]
            if progress_callback is not None:
                try:
                    progress_callback(min(evaluated, total_to_eval), float(best_score), total_to_eval)
                except Exception:
                    pass
        if best_sol is None:
            raise RuntimeError("No solutions evaluated in brute force.")
        meta = {
            'elapsed_sec': None,
            'device_counts': self.device_counts,
            'measure': self.measure,
            'method': 'brute',
            'timestamp': datetime.datetime.now().isoformat(),
            'evaluated': evaluated,
        }
        result = OptimizationResult(method='brute', best_solution=best_sol, best_fitness=best_score, fitness_history=fitness_history, meta=meta)
        self.results = result
        self.results_history.append(result)

    # Dispatcher
    def run_optimization(self, progress_callback: Optional[Callable[[int, float, int], None]] = None) -> None:
        if self.method == 'genetic':
            self.run_genetic_optimization(progress_callback=progress_callback)
        elif self.method == 'anneal':
            self.run_anneal_optimization(progress_callback=progress_callback)
        elif self.method == 'multianneal':
            self.run_multianneal_optimization(progress_callback=progress_callback)
        elif self.method == 'gradient':
            self.run_gradient_optimization(progress_callback=progress_callback)
        elif self.method == 'mSGD':
            self.run_branch_bound_optimization(progress_callback=progress_callback)
        elif self.method == 'brute':
            self.run_brute_force_optimization(progress_callback=progress_callback)
        else:
            raise ValueError(f"Unknown optimization method '{self.method}'")

    # ---------------------------------------------------------------
    # Constraint helper functions (simplified adaptations)
    # ---------------------------------------------------------------
    def adjust_depth(self, depthrange: Tuple[float,float], position: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        mindepth, maxdepth = depthrange
        dev_vec = position[:3].copy()/np.linalg.norm(position[:3]) if np.linalg.norm(position[:3])>0 else position[:3]
        # Get depth along device axis (approximation): max projection of (vertex - device) onto dev_vec
        diffs = vertices - position[:3]
        proj = diffs @ dev_vec
        depth = np.nanmax(proj)
        if not np.isnan(mindepth) and depth < mindepth:
            position[:3] -= (mindepth - depth)*dev_vec
        if not np.isnan(maxdepth) and depth > maxdepth:
            position[:3] += (depth - maxdepth)*dev_vec
        return position

    def check_proximity(self, population: np.ndarray, dev1_id: int, dev2_id: int) -> Tuple[float, np.ndarray]:
        # Minimal cylinder center distance approximation using origins only
        p1 = population[dev1_id,:3]
        p2 = population[dev2_id,:3]
        axis = p2 - p1
        dist = np.linalg.norm(axis)
        if dist == 0:
            axis = np.array([1e-3,0,0])
            dist = 1e-3
        axis = axis/dist
        # Adjust by radii (clearance widths)
        i1 = self._device_type_index(dev1_id)
        i2 = self._device_type_index(dev2_id)
        if self.cl_wd and i1 < len(self.cl_wd) and i2 < len(self.cl_wd):
            dist -= (self.cl_wd[i1] + self.cl_wd[i2])/2
        return float(dist), axis.astype(float)

    def _device_type_index(self, dev_id: int) -> int:
        # Map flattened device index to type index
        N_index = []
        for i,c in enumerate(self.device_counts):
            N_index.extend([i]*c)
        return N_index[dev_id]

    def _get_structural_avoidance_spheres(self) -> np.ndarray:
        """Return structural avoidance spheres as (K,4) [x,y,z,r] in recentered brain space.

        Prefers cached `self.structural_avoidance`. Otherwise, aggregates from
        `self.structural_files` and optional ROI demo centers using
        `self.structural_demo_radius_mm`.
        """
        try:
            if getattr(self, 'structural_avoidance', None) is not None:
                arr = np.asarray(self.structural_avoidance, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 4:
                    return arr
            rows: list = []
            # Load spheres from MAT files (expect 4xN or Nx4 arrays)
            for path in getattr(self, 'structural_files', []) or []:
                try:
                    d = loadmat(path)
                    keys = [k for k in d.keys() if not str(k).startswith('__')]
                    arr = None
                    for k in keys:
                        v = d[k]
                        if hasattr(v, 'shape') and len(v.shape) >= 2:
                            sh = v.shape
                            if 4 in sh:
                                arr = np.asarray(v, dtype=float)
                                break
                    if arr is None:
                        continue
                    if arr.shape[0] == 4:
                        arr = arr.T
                    elif arr.shape[-1] == 4:
                        arr = arr.reshape((-1, 4))
                    else:
                        arr = np.squeeze(arr)
                        if arr.ndim == 2 and (arr.shape[0] == 4 or arr.shape[1] == 4):
                            arr = arr.T if arr.shape[0] == 4 else arr
                        else:
                            continue
                    arr = np.asarray(arr, dtype=float)
                    mask = ~np.isnan(arr).any(axis=1)
                    arr = arr[mask]
                    if arr.size:
                        rows.append(arr)
                except Exception:
                    continue
            # Demo @ ROI: add one sphere per ROI centroid
            try:
                if bool(getattr(self, 'structural_demo_enabled', False)):
                    rad = float(getattr(self, 'structural_demo_radius_mm', 5.0))
                    brain_center = None
                    if getattr(self, 'brain_vertices', None) is not None:
                        brain_center = np.mean(np.asarray(self.brain_vertices), axis=0)
                    for rp in getattr(self, 'roi_files', []) or []:
                        try:
                            mesh = parse_mesh_from_mat(rp)
                            verts = np.asarray(mesh['vertices'], dtype=float)
                            if brain_center is not None:
                                verts = verts - brain_center
                            cent = np.mean(verts, axis=0)
                            rows.append(np.array([[cent[0], cent[1], cent[2], rad]], dtype=float))
                        except Exception:
                            continue
            except Exception:
                pass
            out = np.vstack(rows) if rows else np.zeros((0, 4), dtype=float)
            # Cache for reuse
            self.structural_avoidance = out
            return out
        except Exception:
            return np.zeros((0, 4), dtype=float)

    def _apply_structural_avoidance(self, pop2d: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply structural avoidance as the final correction.

        For each device line (position, orientation), compute shortest vector to each
        sphere center; if within radius sum, push device position away along -v by
        `(radius_sum - distance) + 0.25` mm. Returns (updated_pop2d, triggered).
        """
        spheres = self._get_structural_avoidance_spheres()
        if spheres.size == 0:
            return pop2d, False
        N = (pop2d.shape[1] // 6)
        triggered = False
        out = np.copy(pop2d)
        for row_idx in range(out.shape[0]):
            row = out[row_idx]
            for d in range(N):
                base = d * 6
                x, y, z, a, b, g = float(row[base+0]), float(row[base+1]), float(row[base+2]), float(row[base+3]), float(row[base+4]), float(row[base+5])
                # Device axis from angles (local +Z forward)
                R = get_rotmat(a, b, g)
                axis = (R @ np.array([0.0, 0.0, 1.0]))
                norm = float(np.linalg.norm(axis))
                if norm < 1e-9:
                    continue
                axis = axis / norm
                p = np.array([x, y, z], dtype=float)
                # Device radius by type index
                try:
                    t_idx = self._device_type_index(d)
                    dev_rad = float(self.device_radii[t_idx]) if (self.device_radii and t_idx < len(self.device_radii)) else 0.5
                except Exception:
                    print("Warning: invalid device radius settings; defaulting to 0.5 mm.")
                    dev_rad = 0.5
                for s in spheres:
                    sc = np.array([float(s[0]), float(s[1]), float(s[2])], dtype=float)
                    sr = float(s[3]) if np.isfinite(s[3]) else 0.0
                    # Closest point on infinite line to sphere center
                    t = float(np.dot(sc - p, axis))
                    closest = p + t * axis
                    v = sc - closest  # vector from line to sphere center
                    dmag = float(np.linalg.norm(v))
                    radius_sum = sr + dev_rad
                    if not np.isfinite(dmag):
                        continue
                    if dmag <= radius_sum:
                        # Push device position away along -v
                        delta = (radius_sum - dmag) + 0.25
                        v_dir = v / (dmag + 1e-12)
                        shift = -v_dir * delta
                        p_new = p + shift
                        out[row_idx, base+0] = float(p_new[0])
                        out[row_idx, base+1] = float(p_new[1])
                        out[row_idx, base+2] = float(p_new[2])
                        p = p_new  # update for subsequent spheres
                        triggered = True
        return out, triggered

    def limit_correction(self, population: np.ndarray) -> np.ndarray:
        """Delegate limit correction to the shared adapter for consistency.

        This builds per-device angle and depth limits from per-type settings,
        uses `self.roi_recentered` for depth envelopes (origin-centered), and
        leverages `self.check_proximity` if proximity is enabled. Structural
        avoidance is applied last, and if it triggers any movement, the full
        correction sequence (angle, depth, proximity, avoidance) repeats up to
        5 total passes or until no further avoidance intersections.
        """
        # Ensure 2D shape for adapter; keep track to restore original shape
        single = population.ndim == 1
        pop2d = population.reshape(1, -1) if single else population
        N = (pop2d.shape[1] // 6)
        # Map per-type settings to per-device arrays
        angle_per_dev = np.full((N,), np.nan, dtype=float)
        depth_per_dev: list = [(np.nan, np.nan)] * N
        for d in range(N):
            t_idx = self._device_type_index(d)
            if self.angle_limit_rad and t_idx < len(self.angle_limit_rad):
                angle_per_dev[d] = float(self.angle_limit_rad[t_idx])
            if self.depth_limits and t_idx < len(self.depth_limits):
                depth_per_dev[d] = tuple(self.depth_limits[t_idx])  # type: ignore
        adapter = make_limit_correction_adapter(
            angle_limits_rad=angle_per_dev,
            do_depth=bool(self.do_depth),
            depth_limits_per_dev=depth_per_dev,
            recentered_vertices=self.roi_recentered if self.roi_recentered is not None else np.zeros((0, 3)),
            proximity_enabled=bool(self.do_proximity),
            check_proximity_fn=self.check_proximity,
        )
        passes = 0
        current = np.copy(pop2d)
        while passes < 5:
            corrected = adapter(current, (3, 2))
            corrected2, triggered = self._apply_structural_avoidance(corrected)
            current = corrected2
            passes += 1
            if not triggered:
                break
        return current.reshape(-1) if single else current


    @staticmethod
    def _generate_initial_population(recentered_roi: np.ndarray, N: int, k: int) -> np.ndarray:
        result = []
        for _ in range(k):
            indices = np.random.choice(recentered_roi.shape[0], N, replace=False)
            verts = recentered_roi[indices]
            sol = []
            for v in verts:
                dev_vec = v / np.linalg.norm(v)
                xaxis = np.cross(dev_vec, np.array([0,0,10])); xaxis /= np.linalg.norm(xaxis)
                yaxis = np.cross(dev_vec, xaxis); yaxis /= np.linalg.norm(yaxis)
                alpha = np.arctan2(xaxis[1], xaxis[0])
                beta = np.arcsin(-xaxis[2])
                gamma = np.arctan2(yaxis[2], dev_vec[2])
                sol.append([v[0], v[1], v[2], alpha, beta, gamma])
            result.append(sol)
        return np.array(result)

# ----------------------------------------------------------------------------------
# Convenience helpers for plots
# ----------------------------------------------------------------------------------

def export_best_solution_csv(project: Project, path_out: str) -> None:
    if project.results is None:
        raise ValueError("No optimization results to export.")
    arr = project.results.best_solution.reshape((-1,6))
    header = 'x,y,z,alpha,beta,gamma'
    np.savetxt(path_out, arr, delimiter=',', header=header, comments='')

# ----------------------------------------------------------------------------------
# Shared limit correction adapter for GUI and optimization schemes
# ----------------------------------------------------------------------------------
def make_limit_correction_adapter(angle_limits_rad: Optional[np.ndarray],
                                  do_depth: bool,
                                  depth_limits_per_dev: Optional[list],
                                  recentered_vertices: Optional[np.ndarray],
                                  proximity_enabled: bool,
                                  check_proximity_fn: Optional[Callable[[np.ndarray, int, int], Tuple[float, np.ndarray]]],
                                  coefficient_default: Tuple[int, int] = (3, 2)) -> Callable[[np.ndarray, Tuple[int, int]], np.ndarray]:
    """
    Build a limit_correction function that clamps angles iteratively toward the brain center,
    optionally adjusts depth using local envelopes, and enforces proximity via a callback.

    - population: (k, N*6) or (N*6,) array of device parameters [x,y,z,alpha,beta,gamma].
    - angle_limits_rad: per-device-type angle limits in radians (None or NaN to disable).
    - do_depth/depth_limits_per_dev: enable and provide [min,max] depth limits per device type.
    - recentered_vertices: vertices in brain-centered coordinates (origin is center) for depth envelopes.
    - proximity_enabled/check_proximity_fn: enforce minimum separations between devices.
    - coefficient_default: kept for API parity; not used in this adapter.
    Returns corrected population of same shape.
    """

    center = np.zeros(3)

    def _rot(a: float, b: float, g: float) -> np.ndarray:
        return get_rotmat(a, b, g)

    def _depth_env(pos_xyz: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
        if recentered_vertices is None or recentered_vertices.size == 0:
            return np.nan, np.nan
        v = np.asarray(recentered_vertices, dtype=float) - pos_xyz.reshape(1, 3)
        v_local = (R.T @ v.T).T
        mask = (v_local[:, 1]**2 + v_local[:, 2]**2) <= 10.0
        vx = v_local[mask, 0] if np.any(mask) else v_local[:, 0]
        return float(np.nanmin(vx)), float(np.nanmax(vx))

    def _limit(population: np.ndarray, coefficient: Tuple[int, int] = coefficient_default) -> np.ndarray:
        single = False
        if population.ndim == 1:
            single = True
            population = population[np.newaxis, :]
        k, dims = population.shape
        N = dims // 6
        pop = np.copy(population)

        for row_idx in range(k):
            row = pop[row_idx]
            for i in range(N):
                base = i * 6
                x, y, z, a, b, g = row[base:base+6]
                pos = np.array([x, y, z], dtype=float)
                R = _rot(float(a), float(b), float(g))

                # Angle limit correction: align device axis (+Z) toward inward radial (-pos)
                # I'm very sorry for this nightmare
                if angle_limits_rad is not None and angle_limits_rad.size >= 1:
                    limit_val = angle_limits_rad[i % len(angle_limits_rad)]
                    if not np.isnan(limit_val):
                        limit = float(limit_val)
                        radial = -pos
                        if np.linalg.norm(radial) > 0:
                            radial /= np.linalg.norm(radial)
                        else:
                            radial = np.array([0.0, 0.0, 1.0])
                        max_iters = 10
                        blend = 0.6 # 0.0-1.0 blending factor per iteration
                        for _ in range(max_iters):
                            # Use +Z as the device axis consistently
                            dev_dir = (R @ np.array([0.0, 0.0, 1.0]))
                            dev_dir /= (np.linalg.norm(dev_dir) + 1e-12)
                            dev_dot = float(np.dot(dev_dir, radial))
                            cosang = float(np.clip(dev_dot, -1.0, 1.0))
                            ang = float(np.arccos(cosang))
                            if ang <= limit:
                                break
                            # Blend current direction toward radial to reduce angle monotonically
                            new_dir = (1.0 - blend) * dev_dir + blend * radial
                            new_dir /= (np.linalg.norm(new_dir) + 1e-12)
                            # Early exit if update is too small to avoid oscillation
                            if float(np.arccos(np.clip(np.dot(dev_dir, new_dir), -1.0, 1.0))) < 1e-6:
                                #print("Small direction update; breaking.")
                                break
                            # Align local +Z to new_dir and extract Euler angles (RzRyRx convention)
                            z_axis = new_dir
                            ref = np.array([0.0, 1.0, 0.0]) if abs(np.dot(z_axis, np.array([0.0, 1.0, 0.0]))) < 0.9 else np.array([1.0, 0.0, 0.0])
                            x_axis = np.cross(ref, z_axis); x_axis /= (np.linalg.norm(x_axis) + 1e-12)
                            y_axis = np.cross(z_axis, x_axis)
                            Rnew = np.column_stack([x_axis, y_axis, z_axis])
                            beta_new = float(np.arcsin(-Rnew[2, 0]))
                            alpha_new = float(np.arctan2(Rnew[1, 0], Rnew[0, 0]))
                            gamma_new = float(np.arctan2(Rnew[2, 1], Rnew[2, 2]))
                            row[base+3:base+6] = np.array([alpha_new, beta_new, gamma_new], dtype=float)
                            R = _rot(alpha_new, beta_new, gamma_new)

                # Depth correction using local z-axis envelope within 10 mm radius
                if do_depth and depth_limits_per_dev is not None and i < len(depth_limits_per_dev):
                    mn_lim, mx_lim = depth_limits_per_dev[i]
                    # Only proceed if at least one bound is provided
                    if not (np.isnan(mn_lim) and np.isnan(mx_lim)):
                        # Compute local coordinates of ROI vertices relative to device position
                        if recentered_vertices is not None and recentered_vertices.size > 0:
                            v = np.asarray(recentered_vertices, dtype=float) - pos.reshape(1, 3)
                            v_local = (R.T @ v.T).T
                            # Consider only sources within 10 mm radial distance from z-axis
                            radial2 = v_local[:, 0]**2 + v_local[:, 1]**2
                            mask = radial2 <= (10.0**2)
                            local_z = v_local[mask, 2] if np.any(mask) else v_local[:, 2]
                            # Current depth is the highest source along local z
                            current_depth = float(np.nanmax(local_z)) if local_z.size else np.nan
                        else:
                            current_depth = np.nan
                        # If current depth is valid, adjust along z-axis only when outside bounds
                        if np.isfinite(current_depth):
                            z_axis = (R @ np.array([0.0, 0.0, 1.0])); z_axis /= (np.linalg.norm(z_axis) + 1e-12)
                            shift = 0.0
                            if not np.isnan(mn_lim) and current_depth < float(mn_lim):
                                shift = float(mn_lim) - current_depth
                            elif not np.isnan(mx_lim) and current_depth > float(mx_lim):
                                shift = float(mx_lim) - current_depth
                            if shift != 0.0:
                                pos = pos + shift * z_axis
                                row[base:base+3] = pos

            # Proximity correction via callback
            if proximity_enabled and check_proximity_fn is not None:
                devs = row.reshape((N, 6))
                for i in range(N):
                    for j in range(i+1, N):
                        try:
                            _, corrected = check_proximity_fn(devs, i, j)
                            devs = corrected.reshape((N, 6))
                        except Exception:
                            pass
                pop[row_idx] = devs.flatten()

        return pop[0] if single else pop

    return _limit


# ----------------------------------------------------------------------------------
# SEPIO utilities: full voltage and IC matrices
# ----------------------------------------------------------------------------------
def _device_type_slices(device_counts: List[int]) -> List[Tuple[int, int]]:
    """Return (start,end) index slices for each device type over flattened devices."""
    slices: List[Tuple[int, int]] = []
    s = 0
    for c in device_counts:
        e = s + int(c)
        slices.append((s, e))
        s = e
    return slices

def _ensure_roi_vertices_and_normals(project: "Project") -> Tuple[np.ndarray, np.ndarray]:
    """Return depth-offset, recentered ROI vertices and corresponding normals.

    - Uses cached `project.roi_recentered` when available for vertices.
    - Computes normals from cached faces/vertices if available; otherwise parses ROI files.
    - Applies the same dipole offset used during optimization.
    """
    # Vertices (depth-offset + recentered)
    if project.roi_recentered is not None:
        roi_vertices = np.asarray(project.roi_recentered, dtype=float)
    else:
        # Build from ROI files similarly to genetic path
        roi_vertices_all = []
        roi_normals_all = []
        for rf in project.roi_files:
            mesh = parse_mesh_from_mat(rf)
            v = mesh['vertices']
            f = normalize_faces(mesh['faces'], v.shape[0])
            n = vertex_normals(v, f)
            roi_vertices_all.append(v)
            roi_normals_all.append(n)
        roi_vertices_full = np.vstack(roi_vertices_all)
        roi_normals_full = np.vstack(roi_normals_all)
        roi_vertices = uniform_offset(roi_vertices_full, roi_normals_full, project.dipole_offset * project.cortical_thickness)
        # Recenter to brain
        if project.brain_vertices is None:
            # Load brain mesh minimally
            brain_mat = loadmat(project.brain_file)
            _, _, brain_vertices, _ = obtain_data(brain_mat, 'brain')
        else:
            brain_vertices = np.asarray(project.brain_vertices)
        roi_vertices, _center = recenter(roi_vertices, brain_vertices)
        project.roi_recentered = roi_vertices

    # Normals
    if project.roi_faces is not None and project.roi_vertices_full is not None:
        v_full = np.asarray(project.roi_vertices_full)
        f_full = normalize_faces(np.asarray(project.roi_faces), v_full.shape[0])
        roi_normals = vertex_normals(v_full, f_full)
    else:
        # Parse and stack normals from ROI files
        roi_normals_all = []
        for rf in project.roi_files:
            mesh = parse_mesh_from_mat(rf)
            v = mesh['vertices']
            f = normalize_faces(mesh['faces'], v.shape[0])
            n = vertex_normals(v, f)
            roi_normals_all.append(n)
        roi_normals = np.vstack(roi_normals_all)

    return roi_vertices.astype(float), roi_normals.astype(float)

def _load_leadfields(project: "Project") -> List[np.ndarray]:
    """Load leadfield fields for each device type as arrays shaped [nx,ny,nz,3,elec]."""
    if FieldImporter is None:
        raise RuntimeError("Leadfield importer unavailable")
    importer = FieldImporter()
    field_list: List[np.ndarray] = []
    for lf in project.leadfield_files:
        importer.load(lf, clear_fields=True)
        field_list.append(np.asarray(importer.fields))
    return field_list

def compute_full_voltage_arrays(project: "Project", solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute aggregated voltage and IC matrices across all sources and sensors.

    - Aggregates per device instance within a type by elementwise max of |V|.
    - Concatenates per-type matrices along the sensor axis to preserve full shape.
    - Computes IC per sensor using per-type noise/bandwidth and concatenates similarly.

    Returns (V_all[sources, sensors_total], IC_all[sources, sensors_total]).
    """
    # Basic validations
    if not project.leadfield_files:
        raise ValueError("Project has no leadfields configured.")
    if not project.roi_files:
        raise ValueError("Project has no ROI files configured.")

    # ROI points and normals
    roi_vertices, roi_normals = _ensure_roi_vertices_and_normals(project)
    n_sources = int(roi_vertices.shape[0])

    # Load fields (one per device type)
    fields = _load_leadfields(project)
    if len(fields) != len(project.device_counts):
        # Allow mismatch but warn by trimming to min length
        m = min(len(fields), len(project.device_counts))
        fields = fields[:m]
    # Device type slices over flattened devices (N*6 array)
    slices = _device_type_slices(project.device_counts)
    poses = np.asarray(solution).reshape((-1, 6))
    poses_len = int(poses.shape[0])

    def _voltage_matrix(field: np.ndarray, dippos: np.ndarray, dipvec: np.ndarray) -> np.ndarray:
        nx, ny, nz = int(field.shape[0]), int(field.shape[1]), int(field.shape[2])
        n_vertices = int(dippos.shape[0])
        n_electrodes = int(field.shape[-1])
        V = np.full((n_vertices, n_electrodes), np.nan, dtype=float)
        for i in range(n_vertices):
            p = dippos[i]
            v = dipvec[i]
            if np.isnan(p).any():
                continue
            x = int(p[0] + nx//2)
            y = int(p[1] + ny//2)
            z = int(p[2])
            if (x < 0) or (x >= nx) or (y < 0) or (y >= ny) or (z < 0) or (z >= nz):
                continue
            for e in range(n_electrodes):
                lf_vec = field[x, y, z, :, e]
                if not np.isnan(lf_vec).any():
                    V[i, e] = float(np.dot(lf_vec, v))
        return V

    per_type_V: List[np.ndarray] = []
    per_type_IC: List[np.ndarray] = []

    for t_idx, (start, end) in enumerate(slices):
        if t_idx >= len(fields):
            break
        field = fields[t_idx]
        n_elec = int(field.shape[-1])
        scale = float(project.scale[t_idx]) if t_idx < len(project.scale) and project.scale[t_idx] else 1.0
        magnitude = float(project.magnitude)
        noise = float(project.noise[t_idx]) if t_idx < len(project.noise) and project.noise[t_idx] else 1.0
        bandwidth = float(project.bandwidth[t_idx]) if t_idx < len(project.bandwidth) and project.bandwidth[t_idx] else 1.0

        # Collect per-instance matrices for this type; later flatten device axis into sensor axis
        per_instance_V: List[np.ndarray] = []

        # Clamp instance range to available poses to avoid OOB in multi-device cases
        if poses_len <= start:
            # No instances available for this type in the provided solution
            instance_range = range(0, 0)
        else:
            end = min(end, poses_len)
            instance_range = range(start, end)

        for d_idx in instance_range:
            devpos = poses[d_idx]
            dippos, dipvec = transform_vectorspace(field, scale, magnitude, roi_vertices, roi_normals, devpos)
            dippos, dipvec = trim_data(field, dippos, dipvec)
            V_dev = _voltage_matrix(field, dippos, dipvec)
            # Keep absolute voltages, but DO NOT aggregate across devices; retain each instance
            per_instance_V.append(np.abs(V_dev))

        # If no instances, contribute zero-width; else flatten device axis into sensor axis
        if len(per_instance_V) == 0:
            V_type_flat = np.zeros((n_sources, 0), dtype=float)
        else:
            # Concatenate per-instance electrodes along sensor axis
            V_type_flat = np.concatenate(per_instance_V, axis=1)

        # Compute IC per sensor for this type and append (per-source, per-electrode)
        denom = (noise if noise != 0 else 1.0)
        snr2_type = (V_type_flat / denom) ** 2
        IC_type = bandwidth * np.log2(1.0 + snr2_type)
        # Reader note: See line line 724
        per_type_V.append(V_type_flat)
        per_type_IC.append(IC_type)

    # Concatenate across sensor axis to preserve full shape
    V_all = np.concatenate(per_type_V, axis=1) if per_type_V else np.zeros((n_sources, 0))
    IC_all = np.concatenate(per_type_IC, axis=1) if per_type_IC else np.zeros((n_sources, 0))

    return V_all, IC_all

