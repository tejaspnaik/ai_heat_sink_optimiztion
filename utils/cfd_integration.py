"""
CFD Data Integration Utilities
Load and process OpenFOAM, ANSYS Fluent, and other CFD simulation data
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import json
import struct
import warnings


class OpenFOAMLoader:
    """
    Load OpenFOAM simulation results
    Reads mesh, boundary fields, and internal fields
    """
    
    @staticmethod
    def read_openfoam_mesh(case_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Read OpenFOAM mesh from constant/polyMesh directory
        
        Args:
            case_dir: Path to OpenFOAM case directory
            
        Returns:
            (points, cells, boundaries)
            - points: (N, 3) array of node coordinates
            - cells: (M, variable) array of cell connectivity
            - boundaries: List of boundary dictionaries
        """
        mesh_dir = case_dir / "constant" / "polyMesh"
        
        # Read points
        points_file = mesh_dir / "points"
        points = OpenFOAMLoader._read_openfoam_list(points_file)
        points = np.array(points, dtype=np.float32)
        
        # Read faces
        faces_file = mesh_dir / "faces"
        faces = OpenFOAMLoader._read_openfoam_list(faces_file)
        
        # Read owner and neighbor
        owner_file = mesh_dir / "owner"
        neighbour_file = mesh_dir / "neighbour"
        
        owner = OpenFOAMLoader._read_openfoam_list(owner_file)
        neighbour = OpenFOAMLoader._read_openfoam_list(neighbour_file)
        
        # Parse boundaries
        boundary_file = mesh_dir / "boundary"
        boundaries = OpenFOAMLoader._parse_boundary_file(boundary_file)
        
        return points, faces, boundaries
    
    @staticmethod
    def _read_openfoam_list(file_path: Path) -> List:
        """Parse OpenFOAM list format"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = '\n'.join([line.split('//')[0] for line in content.split('\n')])
        
        # Extract list content between { }
        start_idx = content.find('(')
        end_idx = content.rfind(')')
        
        if start_idx < 0 or end_idx < 0:
            return []
        
        list_content = content[start_idx + 1:end_idx].strip()
        
        # Parse entries
        entries = []
        depth = 0
        current_entry = ""
        
        for char in list_content:
            if char == '(':
                depth += 1
                current_entry += char
            elif char == ')':
                depth -= 1
                current_entry += char
            elif char in (' ', '\n', '\t') and depth == 0:
                if current_entry.strip():
                    entries.append(current_entry.strip())
                current_entry = ""
            else:
                current_entry += char
        
        if current_entry.strip():
            entries.append(current_entry.strip())
        
        return entries
    
    @staticmethod
    def _parse_boundary_file(boundary_file: Path) -> List[Dict]:
        """Parse OpenFOAM boundary definition file"""
        boundaries = []
        
        with open(boundary_file, 'r') as f:
            content = f.read()
        
        # Extract boundary blocks
        import re
        pattern = r'(\w+)\s*\{([^}]+)\}'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            boundary_name = match.group(1)
            boundary_content = match.group(2)
            
            boundary_dict = {'name': boundary_name}
            
            # Extract properties
            prop_pattern = r'(\w+)\s+([^;]+);'
            prop_matches = re.finditer(prop_pattern, boundary_content)
            
            for prop_match in prop_matches:
                key = prop_match.group(1)
                value = prop_match.group(2).strip()
                boundary_dict[key] = value
            
            boundaries.append(boundary_dict)
        
        return boundaries
    
    @staticmethod
    def read_openfoam_field(case_dir: Path, field_name: str, 
                           time_step: float) -> Tuple[np.ndarray, Dict]:
        """
        Read scalar or vector field from OpenFOAM
        
        Args:
            case_dir: Case directory
            field_name: Field name (e.g., 'p', 'U', 'k', 'epsilon')
            time_step: Time directory (e.g., 0, 0.001, 1.0)
            
        Returns:
            (field_data, field_info)
        """
        time_dir = case_dir / str(time_step)
        field_file = time_dir / field_name
        
        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")
        
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Extract header info
        field_info = {}
        header_pattern = r'(\w+)\s+(\w+);'
        
        for match in re.finditer(header_pattern, content.split('internalField')[0]):
            field_info[match.group(1)] = match.group(2)
        
        # Extract internal field
        if 'nonuniform' in content:
            # Extract nonuniform list
            start = content.find('nonuniform');
            if start >= 0:
                start = content.find('(', start)
                end = content.rfind(')')
                field_content = content[start + 1:end].strip()
                
                # Parse vector/scalar values
                values = []
                depth = 0
                current_val = ""
                
                for char in field_content:
                    if char == '(':
                        depth += 1
                        current_val += char
                    elif char == ')':
                        depth -= 1
                        current_val += char
                    elif char in (' ', '\n') and depth == 0:
                        if current_val.strip():
                            values.append(current_val.strip())
                        current_val = ""
                    else:
                        current_val += char
                
                if current_val.strip():
                    values.append(current_val.strip())
                
                # Convert to array
                data = []
                for val in values:
                    if val.startswith('('):
                        # Vector
                        components = val.strip('()').split()
                        data.append([float(c) for c in components])
                    else:
                        # Scalar
                        data.append(float(val))
                
                field_data = np.array(data, dtype=np.float32)
        else:
            # Uniform field
            field_data = np.zeros((1, 1), dtype=np.float32)
        
        return field_data, field_info
    
    @staticmethod
    def load_full_case(case_dir: Path, time_steps: List[float],
                      fields: List[str]) -> Dict[str, np.ndarray]:
        """
        Load complete OpenFOAM case with multiple fields and timesteps
        
        Args:
            case_dir: Case directory path
            time_steps: List of time steps to load
            fields: List of field names to load
            
        Returns:
            Dictionary with structure:
                {
                    'points': (N, 3) point coordinates,
                    'time<t>_<field>': Field arrays,
                    'metadata': Simulation info
                }
        """
        case_dir = Path(case_dir)
        
        result = {}
        
        # Load mesh
        try:
            points, faces, boundaries = OpenFOAMLoader.read_openfoam_mesh(case_dir)
            result['points'] = points
            result['faces'] = faces
            result['boundaries'] = boundaries
        except Exception as e:
            warnings.warn(f"Failed to load mesh: {e}")
        
        # Load fields
        for t in time_steps:
            for field in fields:
                try:
                    field_data, info = OpenFOAMLoader.read_openfoam_field(case_dir, field, t)
                    key = f"time{t}_{field}"
                    result[key] = field_data
                except Exception as e:
                    warnings.warn(f"Failed to load {field} at t={t}: {e}")
        
        return result


class FluentLoader:
    """
    Load ANSYS Fluent case files (.cas/.dat)
    """
    
    @staticmethod
    def read_fluent_cas(cas_file: Path) -> Dict:
        """
        Read Fluent .cas file (ASCII format)
        Basic parser for geometry sections
        
        Args:
            cas_file: Path to .cas file
            
        Returns:
            Dictionary with mesh data
        """
        result = {}
        
        with open(cas_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse sections
            if line.startswith('(0'):  # Section headers
                section_id = int(line[1:].split()[0])
                
                if section_id == 10:  # Dimensions
                    result['dimensions'] = int(lines[i + 1].strip())
                elif section_id == 12:  # Nodes
                    points = []
                    j = i + 2
                    while j < len(lines) and not lines[j].startswith('('):
                        parts = lines[j].split()
                        if len(parts) >= 3:
                            points.append([float(p) for p in parts[:3]])
                        j += 1
                    result['points'] = np.array(points, dtype=np.float32)
            
            i += 1
        
        return result
    
    @staticmethod
    def read_fluent_dat(dat_file: Path) -> Dict:
        """
        Read Fluent .dat file (binary format)
        Parses zone records and data sections
        
        Args:
            dat_file: Path to .dat file
            
        Returns:
            Dictionary with field data
        """
        result = {}
        
        try:
            with open(dat_file, 'rb') as f:
                # Read header
                header = f.read(4)
                
                # Parse zones (simplified)
                zones = []
                while True:
                    zone_header = f.read(12)
                    if len(zone_header) < 12:
                        break
                    
                    zone_id = struct.unpack('i', zone_header[0:4])[0]
                    zone_type = struct.unpack('i', zone_header[4:8])[0]
                    num_nodes = struct.unpack('i', zone_header[8:12])[0]
                    
                    zones.append({
                        'id': zone_id,
                        'type': zone_type,
                        'num_nodes': num_nodes
                    })
            
            result['zones'] = zones
        except Exception as e:
            warnings.warn(f"Error reading Fluent .dat file: {e}")
        
        return result


class CFD_HDF5Loader:
    """
    Load CFD data from HDF5 format (common for large datasets)
    """
    
    @staticmethod
    def load_hdf5(hdf5_file: Path) -> Dict[str, np.ndarray]:
        """
        Load HDF5 CFD data
        
        Args:
            hdf5_file: Path to .h5 file
            
        Returns:
            Dictionary with structure:
                {
                    'points': Coordinates,
                    'cells': Cell connectivity,
                    '<field_name>': Field arrays,
                    'metadata': Case information
                }
        """
        import h5py
        
        result = {}
        
        with h5py.File(hdf5_file, 'r') as f:
            # Recursively load all datasets
            def load_h5_recursive(group, path=""):
                for key in group.keys():
                    dataset_path = f"{path}/{key}" if path else key
                    
                    if isinstance(group[key], h5py.Dataset):
                        result[dataset_path] = np.array(group[key], dtype=np.float32)
                    elif isinstance(group[key], h5py.Group):
                        load_h5_recursive(group[key], dataset_path)
            
            load_h5_recursive(f)
        
        return result


class CFDDataProcessor:
    """
    Post-process CFD data for PINN training
    """
    
    @staticmethod
    def extract_interior_data(cell_centers: np.ndarray, field_data: np.ndarray,
                            boundary_mask: Optional[np.ndarray] = None,
                            exclude_boundaries: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract interior domain data excluding boundaries
        
        Args:
            cell_centers: Cell center coordinates (N, 3)
            field_data: Field values (N, num_vars)
            boundary_mask: Boolean array indicating boundary cells
            exclude_boundaries: Whether to exclude boundary regions
            
        Returns:
            (filtered_centers, filtered_data)
        """
        if boundary_mask is not None and exclude_boundaries:
            mask = ~boundary_mask
            return cell_centers[mask], field_data[mask]
        return cell_centers, field_data
    
    @staticmethod
    def interpolate_to_regular_grid(points: np.ndarray, field: np.ndarray,
                                    grid_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Interpolate unstructured CFD data to regular grid
        
        Args:
            points: Unstructured points (N, 3)
            field: Field values (N, num_vars)
            grid_size: Regular grid dimensions (nx, ny, nz)
            
        Returns:
            Interpolated field on regular grid
        """
        from scipy.interpolate import griddata
        
        # Create regular grid
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        x = np.linspace(x_min, x_max, grid_size[0])
        y = np.linspace(y_min, y_max, grid_size[1])
        z = np.linspace(z_min, z_max, grid_size[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        # Interpolate
        interpolated = griddata(points, field, grid_points, method='linear')
        
        return interpolated.reshape(grid_size + (-1,))
    
    @staticmethod
    def compute_residuals(cfd_solution: np.ndarray, pinn_solution: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute residuals between CFD and PINN solutions
        
        Args:
            cfd_solution: CFD reference solution (N, variables)
            pinn_solution: PINN prediction (N, variables)
            mask: Optional mask for specific region
            
        Returns:
            Dictionary with various error metrics
        """
        if mask is None:
            mask = np.ones(cfd_solution.shape[0], dtype=bool)
        
        diff = cfd_solution[mask] - pinn_solution[mask]
        
        return {
            'mae': np.mean(np.abs(diff), axis=0),
            'rmse': np.sqrt(np.mean(diff ** 2, axis=0)),
            'max_error': np.max(np.abs(diff), axis=0),
            'relative_error': np.mean(np.abs(diff) / (np.abs(cfd_solution[mask]) + 1e-8), axis=0),
        }


class CFDDatasetBuilder:
    """
    Build PINN training dataset from CFD simulations
    """
    
    @staticmethod
    def create_training_set(cfd_data: Dict[str, np.ndarray],
                           training_ratio: float = 0.8,
                           points_key: str = 'points',
                           field_keys: Optional[List[str]] = None) -> Dict:
        """
        Create training, validation, and test sets
        
        Args:
            cfd_data: Dictionary with CFD results from loader
            training_ratio: Fraction for training data
            points_key: Dictionary key for point coordinates
            field_keys: Keys for field data to use
            
        Returns:
            Dictionary with train/val/test splits
        """
        points = cfd_data[points_key]
        N = points.shape[0]
        
        # Generate shuffle indices
        indices = np.random.permutation(N)
        
        n_train = int(N * training_ratio)
        n_val = int(N * (1 - training_ratio) / 2)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        result = {
            'train': {'points': points[train_idx]},
            'val': {'points': points[val_idx]},
            'test': {'points': points[test_idx]},
        }
        
        # Add field data
        if field_keys is None:
            field_keys = [k for k in cfd_data.keys() if k != points_key]
        
        for field_key in field_keys:
            if field_key in cfd_data:
                try:
                    field_data = cfd_data[field_key]
                    if field_data.shape[0] == N:
                        result['train'][field_key] = field_data[train_idx]
                        result['val'][field_key] = field_data[val_idx]
                        result['test'][field_key] = field_data[test_idx]
                except:
                    pass
        
        result['metadata'] = {
            'total_samples': N,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx),
            'training_ratio': training_ratio,
        }
        
        return result
    
    @staticmethod
    def to_torch_dataset(cfd_dataset: Dict) -> Dict[str, torch.utils.data.TensorDataset]:
        """
        Convert CFD dataset to PyTorch tensors
        
        Args:
            cfd_dataset: Dataset from create_training_set
            
        Returns:
            Dictionary with PyTorch TensorDatasets
        """
        result = {}
        
        for split in ['train', 'val', 'test']:
            if split not in cfd_dataset:
                continue
            
            split_data = cfd_dataset[split]
            
            points_tensor = torch.from_numpy(split_data['points']).float()
            
            # Combine other tensors
            field_tensors = []
            for key in split_data.keys():
                if key != 'points':
                    field_tensors.append(torch.from_numpy(split_data[key]).float())
            
            if field_tensors:
                field_combined = torch.cat(field_tensors, dim=1)
                result[split] = torch.utils.data.TensorDataset(points_tensor, field_combined)
            else:
                result[split] = torch.utils.data.TensorDataset(points_tensor)
        
        return result


import re
