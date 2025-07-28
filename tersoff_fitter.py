#!/usr/bin/env python3
import math
import re
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, product
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add
from torch_cluster import radius
from torch_geometric.utils import remove_self_loops

DATA_FILES = {
    'xyz': "PbCsBr_black_700K.xyz",
    'forces': "black_700K_forces.data"
}

GRAPH_PARAMS = {
    'cutoff': 6.0,
    'max_neighbors': 512
}

# Initial Tersoff parameters- same for all interactions
Initial_Parameters = {
    'A': 2000.0, 
    'B': 100.0, 
    'lambda1': 2.5, 
    'lambda2': 1.5,
    'lambda3': 1.0, 
    'beta': 1.0e-3, 
    'n': 1.0, 
    'gamma': 1.0,
    'c': 10.0, 
    'd': 1.0, 
    'R': 5.0, 
    'D': 0.5, 
    'E_ref': 0.0,
}

TRAINING_CONFIG = {
    'h_value': -1.0,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'energy_weight': 0.1,
    'force_weight': 1.0,
    'weight_decay': 0.0,
    'batch_size': 16,
    'max_epochs': 5000,
    'patience': 100,
    'tolerance': 1e-7,
    'learning_rates': {'phase1': 5e-5, 'phase2': 1e-5, 'phase3': 5e-6}
}

# ===== DATA LOADING =====
class DFTDataset(Dataset):
    """Loads and processes DFT trajectory data into graph format."""
    
    def __init__(self, xyz_path, cutoff, forces_path=None, start_frame=0):
        super().__init__()
        self.xyz_path = xyz_path
        self.cutoff = cutoff
        
        print(f"Loading data from '{xyz_path}'...")
        self._scan_trajectory_file(start_frame)
        self._load_forces_if_available(forces_path)
        self._preprocess_all_frames()
        
    def _scan_trajectory_file(self, start_frame):
        """Scan XYZ file to index all frames and extract basic info."""
        self.frame_positions = []
        self.atoms_per_frame = []
        self.total_energies = []
        atom_symbols = set()
        
        with open(self.xyz_path, 'r') as file:
            frame_idx = 0
            while True:
                file_position = file.tell()
                line = file.readline()
                if not line:
                    break
                    
                try:
                    num_atoms = int(line.strip())
                except ValueError:
                    continue
                
                # Extract energy from comment line
                comment_line = file.readline()
                try:
                    energy_match = re.search(r'E\s*=\s*([-\d\.eE]+)', comment_line)
                    energy = float(energy_match.group(1))
                except (AttributeError, ValueError):
                    # Skip invalid frames
                    for _ in range(num_atoms):
                        file.readline()
                    continue

                if frame_idx >= start_frame:
                    self.frame_positions.append(file_position)
                    self.atoms_per_frame.append(num_atoms)
                    self.total_energies.append(energy)
                    
                    # Collect atom types
                    for _ in range(num_atoms):
                        atom_line = file.readline()
                        symbol = atom_line.strip().split()[0]
                        atom_symbols.add(symbol)
                else:
                    # Skip unwanted frames
                    for _ in range(num_atoms):
                        file.readline()
                        
                frame_idx += 1

        if not self.frame_positions:
            raise ValueError(f"No valid frames found in {self.xyz_path}")

        # Setup atom type mapping
        self.atom_types = sorted(list(atom_symbols))
        self.type_to_index = {symbol: i for i, symbol in enumerate(self.atom_types)}
        self.num_atom_features = len(self.atom_types)
        
        # Calculate energy statistics
        self.atoms_per_frame = np.array(self.atoms_per_frame)
        self.total_energies = np.array(self.total_energies)
        self.energy_per_atom = self.total_energies / self.atoms_per_frame
        self.energy_mean = np.mean(self.energy_per_atom)
        self.energy_std = np.std(self.energy_per_atom)
        
        print(f"Loaded {len(self.frame_positions)} frames with atom types: {self.atom_types}")
        print(f"Energy/atom statistics: mean={self.energy_mean:.4f}, std={self.energy_std:.4f}")

    def _load_forces_if_available(self, forces_path):
        """Load force data from separate file if provided."""
        self.force_data = None
        if not forces_path or not os.path.exists(forces_path):
            return
            
        print(f"Loading forces from '{forces_path}'...")
        BOHR_TO_ANGSTROM = 0.529177210903
        FORCE_CONVERSION = 1.0 / BOHR_TO_ANGSTROM
        
        self.force_data = []
        with open(forces_path, 'r') as file:
            lines = file.readlines()
            
        line_idx = 0
        frame_idx = 0
        
        while frame_idx < len(self.frame_positions) and line_idx < len(lines):
            if '# Atom' in lines[line_idx]:
                num_atoms = self.atoms_per_frame[frame_idx]
                frame_forces = []
                
                for atom_idx in range(num_atoms):
                    force_line = lines[line_idx + 1 + atom_idx].strip().split()
                    raw_forces = [float(force_line[i]) for i in range(3, 6)]
                    converted_forces = [f * FORCE_CONVERSION for f in raw_forces]
                    frame_forces.append(converted_forces)
                    
                self.force_data.append(frame_forces)
                line_idx += num_atoms + 1
                frame_idx += 1
            else:
                line_idx += 1
                
        if self.force_data:
            print(f"Loaded forces for {len(self.force_data)} frames")

    def _preprocess_all_frames(self):
        """Convert all frames to graph format and store in memory."""
        print("Converting structures to graph format...")
        self.graph_data = []
        
        for frame_idx in range(len(self.frame_positions)):
            if (frame_idx + 1) % 500 == 0 or frame_idx == 0 or frame_idx + 1 == len(self.frame_positions):
                print(f"  Processing frame {frame_idx + 1}/{len(self.frame_positions)}")
            self.graph_data.append(self._create_graph_from_frame(frame_idx))
            
        print("Preprocessing complete")

    def _create_graph_from_frame(self, frame_idx):
        """Convert a single frame to PyTorch Geometric Data object."""
        # Read atomic coordinates and types
        with open(self.xyz_path, 'r') as file:
            file.seek(self.frame_positions[frame_idx])
            num_atoms = int(file.readline().strip())
            file.readline()  # Skip comment
            
            atom_data = []
            for _ in range(num_atoms):
                parts = file.readline().split()
                symbol = parts[0]
                coords = [float(parts[i]) for i in range(1, 4)]
                atom_data.append((symbol, coords))

        # Convert to tensors
        positions = torch.tensor([coords for _, coords in atom_data], dtype=torch.float)
        type_indices = torch.tensor([self.type_to_index[symbol] for symbol, _ in atom_data], dtype=torch.long)
        node_features = torch.nn.functional.one_hot(type_indices, num_classes=self.num_atom_features).float()

        # Build graph edges
        edge_index = radius(positions, positions, self.cutoff, max_num_neighbors=GRAPH_PARAMS['max_neighbors'])
        edge_index = remove_self_loops(edge_index)[0]
        
        # Normalized energy is no longer used for loss, but can be kept for other analyses
        normalized_energy = (self.energy_per_atom[frame_idx] - self.energy_mean) / self.energy_std
        
        # Create data object
        graph = Data(
            x=node_features,
            pos=positions,
            edge_index=edge_index,
            y=torch.tensor([normalized_energy], dtype=torch.float),
            y_total=torch.tensor([self.total_energies[frame_idx]], dtype=torch.float)
        )

        # Add forces if available
        if self.force_data and frame_idx < len(self.force_data):
            graph.forces = torch.tensor(self.force_data[frame_idx], dtype=torch.float)
        
        return graph

    def len(self):
        return len(self.graph_data)

    def get(self, idx):
        return self.graph_data[idx]

# ===== TERSOFF MODEL =====
class TersoffGNN(nn.Module):
    """Graph Neural Network implementing Tersoff potential."""
    
    def __init__(self, initial_params, interaction_map):
        super().__init__()
        self.training_phase = 1  # Controls which terms are active
        self.register_buffer('interaction_map', interaction_map)
        self.eps = 1e-15  # Small value to prevent division by zero
        
        # Initialize learnable parameters (in log space for positivity)
        param_names = ['A', 'B', 'lambda1', 'lambda2', 'lambda3', 'beta', 'n', 'gamma', 'c', 'd']
        for name in param_names:
            values = initial_params[name]
            log_values = torch.tensor([math.log(v) if v > self.eps else -20.0 for v in values], dtype=torch.float)
            setattr(self, f"log_{name}", nn.Parameter(log_values))

        # Reference energies (not in log space)
        self.E_ref = nn.Parameter(torch.tensor(initial_params['E_ref'], dtype=torch.float))
        
        # Fixed parameters
        self.register_buffer('h_values', torch.tensor(initial_params['h'], dtype=torch.float))
        self.register_buffer('R_cutoff', torch.tensor(initial_params['R'], dtype=torch.float))
        self.register_buffer('D_width', torch.tensor(initial_params['D'], dtype=torch.float))

    def _get_parameters_for_edges(self, edge_param_indices):
        """Extract parameters for specific edge types."""
        params = {}
        param_names = ['A', 'B', 'lambda1', 'lambda2', 'beta', 'n', 'gamma', 'c', 'd', 'lambda3']
        
        for name in param_names:
            log_param = getattr(self, f"log_{name}")
            params[name] = torch.exp(log_param)[edge_param_indices]
        
        params['h'] = self.h_values[edge_param_indices]
        params['R'] = self.R_cutoff[edge_param_indices]
        params['D'] = self.D_width[edge_param_indices]
        return params

    def _smooth_cutoff_function(self, distances, R, D):
        """Smooth cutoff function for Tersoff potential."""
        cutoff_values = torch.ones_like(distances)
        
        transition_start = R - D
        transition_end = R + D
        
        # Apply smooth transition
        in_transition = (distances >= transition_start) & (distances < transition_end)
        beyond_cutoff = distances >= transition_end
        
        if torch.any(in_transition):
            r_trans = distances[in_transition]
            R_trans = R[in_transition]
            D_trans = D[in_transition]
            
            argument = torch.pi * (r_trans - R_trans + D_trans) / (2 * D_trans + self.eps)
            cutoff_values[in_transition] = 0.5 - 0.5 * torch.sin(argument)
            
        cutoff_values[beyond_cutoff] = 0.0
        return cutoff_values

    def _angular_function(self, cos_theta, gamma, c, d, h):
        """Angular dependence function for three-body interactions."""
        c_squared = c**2
        d_squared = d**2
        h_minus_cos = h - cos_theta
        
        return gamma * (1 + c_squared / (d_squared + self.eps) - 
                        c_squared / (d_squared + h_minus_cos**2 + self.eps))

    def _compute_bond_order(self, data, edge_param_indices):
        """Calculate bond order terms for multi-body interactions."""
        positions = data.pos
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            return torch.tensor([], device=positions.device)

        source_nodes, target_nodes = edge_index
        
        # Efficiently find angle triplets (i-j-k) to avoid memory explosion
        source_pos = source_nodes.view(-1, 1).float()
        edge_pairs = radius(source_pos, source_pos, r=0, max_num_neighbors=GRAPH_PARAMS['max_neighbors'])
        
        edge_ik_indices, edge_ij_indices = edge_pairs
        
        # Filter out self-pairs and duplicates
        mask = edge_ij_indices < edge_ik_indices
        edge_ij_indices = edge_ij_indices[mask]
        edge_ik_indices = edge_ik_indices[mask]

        if edge_ij_indices.shape[0] == 0:
            return torch.ones(num_edges, device=positions.device)
            
        # Calculate vectors and angles
        vec_ij = positions[target_nodes[edge_ij_indices]] - positions[source_nodes[edge_ij_indices]]
        vec_ik = positions[target_nodes[edge_ik_indices]] - positions[source_nodes[edge_ik_indices]]
        
        dist_ij = torch.norm(vec_ij, dim=1)
        dist_ik = torch.norm(vec_ik, dim=1)
        
        cos_theta = torch.sum(vec_ij * vec_ik, dim=1) / (dist_ij * dist_ik + self.eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Get atom types for parameter lookup
        atom_types = torch.argmax(data.x, dim=1)
        type_i = atom_types[source_nodes[edge_ij_indices]]
        type_j = atom_types[target_nodes[edge_ij_indices]]
        type_k = atom_types[target_nodes[edge_ik_indices]]
        
        # Get parameters for each interaction
        params_ik = self._get_parameters_for_edges(self.interaction_map[type_i, type_k])
        params_ij = self._get_parameters_for_edges(self.interaction_map[type_i, type_j])

        # Calculate angular and cutoff contributions
        angular_term = self._angular_function(cos_theta, params_ik['gamma'], params_ik['c'], params_ik['d'], params_ik['h'])
        cutoff_ik = self._smooth_cutoff_function(dist_ik, params_ik['R'], params_ik['D'])
        cutoff_ij = self._smooth_cutoff_function(dist_ij, params_ij['R'], params_ij['D'])
        
        # Exponential distance dependence
        exp_term_ij = torch.clamp(params_ik['lambda3'] * (dist_ij - dist_ik), max=35.0)
        zeta_contrib_ij = cutoff_ik * angular_term * torch.exp(exp_term_ij)
        
        exp_term_ik = torch.clamp(params_ij['lambda3'] * (dist_ik - dist_ij), max=35.0)
        zeta_contrib_ik = cutoff_ij * angular_term * torch.exp(exp_term_ik)
        
        # Accumulate contributions
        zeta_values = torch.zeros(num_edges, device=positions.device)
        zeta_values.scatter_add_(0, edge_ij_indices, zeta_contrib_ij)
        zeta_values.scatter_add_(0, edge_ik_indices, zeta_contrib_ik)
        
        # Calculate final bond order
        all_edge_params = self._get_parameters_for_edges(edge_param_indices)
        n_exp, beta = all_edge_params['n'], all_edge_params['beta']
        bond_order = (1 + (beta * zeta_values)**n_exp)**(-1 / (2 * n_exp + self.eps))
        
        return bond_order

    def forward(self, data):
        """Forward pass computing energy and forces."""
        positions = data.pos
        edge_index = data.edge_index
        
        # Get atom types and corresponding parameter indices
        # *** FIX 1: Must get atom types for the entire batch ***
        batch_atom_types = torch.argmax(data.x, dim=1)
        edge_param_indices = self.interaction_map[batch_atom_types[edge_index[0]], batch_atom_types[edge_index[1]]]

        # Calculate interatomic distances
        bond_vectors = positions[edge_index[1]] - positions[edge_index[0]]
        bond_distances = torch.norm(bond_vectors, dim=1)
        
        # Get parameters for all edges
        params = self._get_parameters_for_edges(edge_param_indices)
        
        # Calculate repulsive and attractive terms
        repulsive_term = params['A'] * torch.exp(-params['lambda1'] * bond_distances)
        attractive_term = -params['B'] * torch.exp(-params['lambda2'] * bond_distances)
        
        # Bond order calculation (depends on training phase)
        if self.training_phase == 1:
            bond_order = torch.ones_like(bond_distances)
        else:
            bond_order = self._compute_bond_order(data, edge_param_indices)

        # Apply cutoff function
        cutoff_weights = self._smooth_cutoff_function(bond_distances, params['R'], params['D'])
        
        # Total pair potential
        pair_energies = cutoff_weights * (repulsive_term + bond_order * attractive_term)
        
        # Sum to get atomic energies (factor of 0.5 to avoid double counting)
        atomic_potential = scatter_add(0.5 * pair_energies, edge_index[0], dim=0, dim_size=data.num_nodes)
        atomic_total_energy = atomic_potential + self.E_ref[batch_atom_types]
        
        # Sum energies for all atoms in the batch to get per-graph energies
        # The 'data.batch' attribute tells us which atom belongs to which graph in the batch
        total_system_energy = global_add_pool(atomic_total_energy, data.batch)

        # Calculate forces if needed
        forces = None
        if self.training and positions.requires_grad:
            # *** FIX 2: Removed grad_outputs for scalar output to fix shape mismatch ***
            forces = -torch.autograd.grad(
                outputs=total_system_energy.sum(), # Sum all energies in the batch to a single scalar
                inputs=positions,
                create_graph=True,
                retain_graph=True
            )[0]
        
        # The model now outputs total energy for each graph in the batch
        return total_system_energy, forces

    def get_current_parameters(self):
        """Extract current parameter values for output."""
        with torch.no_grad():
            result = {}
            param_names = ['A', 'B', 'lambda1', 'lambda2', 'lambda3', 'beta', 'n', 'gamma', 'c', 'd']
            
            for name in param_names:
                result[name] = torch.exp(getattr(self, f"log_{name}")).tolist()
                
            result['h'] = self.h_values.tolist()
            result['E_ref'] = self.E_ref.tolist()
            result['R'] = self.R_cutoff.tolist()
            result['D'] = self.D_width.tolist()
            
        return result

# ===== TRAINING FUNCTIONS =====
def setup_initial_parameters(atom_types, interaction_pairs):
    """Generate initial parameter arrays for all interaction types."""
    num_interactions = len(interaction_pairs)
    num_atom_types = len(atom_types)
    
    params = {}
    for param_name, default_value in Initial_Parameters.items():
        if param_name == 'E_ref':
            params[param_name] = [default_value] * num_atom_types
        else:
            params[param_name] = [default_value] * num_interactions
            
    params['h'] = [TRAINING_CONFIG['h_value']] * num_interactions
    return params

def train_model_phase(model, optimizer, dataloader, max_epochs, patience, tolerance, phase_name, energy_weight=1.0, force_weight=0.0):
    """Train model for one phase with early stopping."""
    loss_function = nn.MSELoss()
   
   #SCHEDULER
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #optimizer, 'min', factor=0.5, patience=15, min_lr=1e-9
    #)
    
    training_losses = []
    force_rmse_history = []
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    checkpoint_path = f'best_model_{phase_name}.pt'

    print(f"\n--- Training {phase_name} (Device: {TRAINING_CONFIG['device']}) ---")
    print(f"Criteria: Patience={patience}, Tolerance={tolerance}, Max Epochs={max_epochs}")
    print(f"Dataloader size: {len(dataloader)} batches of size {TRAINING_CONFIG['batch_size']}")
    
    total_time = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        
        total_energy_loss = 0.0
        total_force_loss = 0.0
        num_batches = 0
        
        # *** MODIFIED: Loop over the DataLoader ***
        for batch in dataloader:
            batch = batch.to(TRAINING_CONFIG['device'])
            
            if force_weight > 0:
                batch.pos.requires_grad_(True)

            optimizer.zero_grad()
            
            # Forward pass
            predicted_total_energy, predicted_forces = model(batch)
            
            # Energy loss on total system energy (physical units)
            energy_loss = loss_function(predicted_total_energy, batch.y_total.view_as(predicted_total_energy))
            
            # Force loss (if applicable)
            force_loss = torch.tensor(0.0, device=TRAINING_CONFIG['device'])
            if force_weight > 0 and predicted_forces is not None and hasattr(batch, 'forces'):
                force_loss = loss_function(predicted_forces, batch.forces)

            # Total loss balances weighted energy and force contributions
            total_loss = (energy_weight * energy_loss) + (force_weight * force_loss)
            
            # Check for NaN
            if torch.isnan(total_loss):
                print(f"Epoch {epoch}: NaN loss detected. Restoring best model.")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path))
                return model, training_losses, force_rmse_history
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_energy_loss += energy_loss.item()
            if force_weight > 0:
                total_force_loss += force_loss.item()
            num_batches += 1
        
        # Calculate average losses
        epoch_duration = time.time() - epoch_start
        total_time += epoch_duration
        
        avg_energy_loss = total_energy_loss / num_batches
        avg_force_loss = total_force_loss / num_batches
        current_loss = (energy_weight * avg_energy_loss) + (force_weight * avg_force_loss)
        training_losses.append(current_loss)
        
        # Calculate and store RMSE values for plotting
        energy_rmse = math.sqrt(avg_energy_loss)
        force_rmse = math.sqrt(avg_force_loss) if force_weight > 0 and avg_force_loss > 0 else 0.0
        force_rmse_history.append(force_rmse)

        #scheduler.step(current_loss)
        
        # Progress reporting
        if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
            learning_rate = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{max_epochs} | E_RMSE: {energy_rmse:.4f} Ha | "
                  f"F_RMSE: {force_rmse:.6f} Ha/Å | Loss: {current_loss:.6f} | "
                  f"Time: {total_time:.2f}s | LR: {learning_rate:.2e}")
            total_time = 0.0

        # Early stopping check
        if current_loss < best_loss - tolerance:
            best_loss = current_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Convergence reached after {epoch} epochs")
            break
    
    # Restore best model
    if os.path.exists(checkpoint_path):
        print(f"Restoring best model (loss: {best_loss:.6f})")
        model.load_state_dict(torch.load(checkpoint_path))
        os.remove(checkpoint_path)
    
    return model, training_losses, force_rmse_history

# ===== OUTPUT FUNCTIONS =====
# (These functions remain unchanged)
def write_tersoff_potential_file(filename, atom_types, interaction_to_index, final_params):
    """Write parameters in LAMMPS Tersoff format."""
    print(f"Writing potential file: {filename}")
    
    # Standard Tersoff parameter order for LAMMPS
    header_params = ['m', 'gamma', 'lambda3', 'c', 'd', 'h', 'n', 'beta',
                     'lambda2', 'B', 'R', 'D', 'lambda1', 'A']
    
    with open(filename, 'w') as file:
        file.write("# el1 el2 el3 " + " ".join(header_params) + "\n")
        file.write("#" + "-" * 100 + "\n")
        
        for el1, el2, el3 in product(atom_types, repeat=3):
            pair_key = '-'.join(sorted([el1, el3]))
            param_index = interaction_to_index[pair_key]
            
            line_values = [el1, el2, el3]
            for param in header_params:
                if param == 'm':
                    line_values.append("1.0")
                else:
                    line_values.append(f"{final_params[param][param_index]:g}")
                    
            file.write(" ".join(line_values) + "\n")

def create_energy_parity_plot(model, dataset, phase_tag):
    """Generate predicted vs true energy scatter plot."""
    model.eval()
    predictions = []
    true_values = []
    
    # Use a dataloader for evaluation to be consistent
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'])

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(TRAINING_CONFIG['device'])
            predicted_total, _ = model(batch)
            
            predictions.extend(predicted_total.cpu().numpy())
            true_values.extend(batch.y_total.cpu().numpy())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    rmse = np.sqrt(np.mean((predictions - true_values)**2))
    
    plt.figure(figsize=(7, 7))
    plt.scatter(true_values, predictions, s=15, alpha=0.5, label='Data Points')
    
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label=f'RMSE = {rmse:.4f} Ha')
    
    plt.xlabel("True Total Energy (Ha)")
    plt.ylabel("Predicted Total Energy (Ha)")
    plt.title(f"Energy Parity Plot ({phase_tag})")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("parity_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: parity_plot.png")

def create_force_parity_plot(model, dataset, phase_tag):
    """Generate predicted vs true force component scatter plot."""
    if not hasattr(dataset, 'force_data') or dataset.force_data is None:
        return

    model.eval()
    all_predicted_forces = []
    all_true_forces = []

    # Use a dataloader for evaluation
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'])
    
    for batch in dataloader:
        batch = batch.to(TRAINING_CONFIG['device'])
        batch.pos.requires_grad_(True)
        _, predicted_forces = model(batch)
        
        if predicted_forces is not None:
            all_predicted_forces.append(predicted_forces.detach().cpu().numpy())
            # Forces are stored per-atom, need to handle the batch correctly
            all_true_forces.append(batch.forces.cpu().numpy())
    
    if not all_predicted_forces:
        return

    predictions = np.concatenate(all_predicted_forces, axis=0).flatten()
    true_values = np.concatenate(all_true_forces, axis=0).flatten()
    rmse = np.sqrt(np.mean((predictions - true_values)**2))

    plt.figure(figsize=(8, 8))
    plt.hexbin(true_values, predictions, gridsize=50, cmap='viridis', bins='log', mincnt=1)
    
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
             label=f'Component RMSE = {rmse:.5f} Ha/Å')
    
    plt.xlabel("True Force Component (Ha/Å)")
    plt.ylabel("Predicted Force Component (Ha/Å)")
    plt.title(f"Force Parity Plot ({phase_tag})")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.colorbar(label='Log Count')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("force_parity_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: force_parity_plot.png")

def create_training_curve_plot(all_losses, phase_labels):
    """Plot training loss curves for all phases."""
    plt.figure(figsize=(10, 6))
    current_epoch = 0
    
    for phase_idx, losses in enumerate(all_losses):
        if not losses:
            continue
            
        epochs = np.arange(current_epoch + 1, current_epoch + len(losses) + 1)
        plt.plot(epochs, losses, label=phase_labels[phase_idx])
        current_epoch += len(losses)
        
        # Add phase separator line
        if phase_idx < len(all_losses) - 1 and losses:
            plt.axvline(x=current_epoch + 0.5, color='gray', linestyle='--')
    
    plt.yscale('log')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("training_curve.png", dpi=300)
    plt.close()
    print("Saved: training_curve.png")

def create_force_rmse_plot(all_force_rmses, phase_labels):
    """Plot Force RMSE curves for all phases."""
    plt.figure(figsize=(10, 6))
    current_epoch = 0
    
    for phase_idx, phase_rmses in enumerate(all_force_rmses):
        if not phase_rmses:
            continue
            
        epochs = np.arange(current_epoch + 1, current_epoch + len(phase_rmses) + 1)
        plt.plot(epochs, phase_rmses, label=phase_labels[phase_idx])
        current_epoch += len(phase_rmses)
        
        # Add phase separator line
        if phase_idx < len(all_force_rmses) - 1 and phase_rmses:
            plt.axvline(x=current_epoch + 0.5, color='gray', linestyle='--')
    
    plt.yscale('log')
    plt.title('Force RMSE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Force RMSE (Ha/Å, log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("force_rmse_curve.png", dpi=300)
    plt.close()
    print("Saved: force_rmse_curve.png")

def create_potential_curves_plot(fitted_params, interaction_pairs, phase_tag=""):
    """Plot fitted pair potential curves."""
    def pair_potential(r, A, B, lambda1, lambda2):
        return A * np.exp(-lambda1 * r) - B * np.exp(-lambda2 * r)
    
    plt.figure(figsize=(12, 6))
    distance_range = np.linspace(1.5, GRAPH_PARAMS['cutoff'], 200)
    
    for pair_idx, pair_label in enumerate(interaction_pairs):
        # Extract parameters for this pair type
        pair_params = {}
        for param_name, param_values in fitted_params.items():
            if isinstance(param_values, list) and len(param_values) == len(interaction_pairs):
                pair_params[param_name] = param_values[pair_idx]
        
        # Check if we have all required parameters
        required_params = ['A', 'B', 'lambda1', 'lambda2']
        if all(param in pair_params for param in required_params):
            potential_values = pair_potential(
                distance_range,
                pair_params['A'],
                pair_params['B'],
                pair_params['lambda1'],
                pair_params['lambda2']
            )
            plt.plot(distance_range, potential_values, label=f"Pair Potential {pair_label}")
    
    plt.title(f'Fitted Pair Potentials ({phase_tag})')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Potential Energy (Ha)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.axhline(0, color='k', linestyle=':', linewidth=0.7)
    plt.ylim(-0.1, 0.15)
    plt.savefig("potentials_comparison.png", dpi=300)
    plt.close()
    print("Saved: potentials_comparison.png")

def generate_all_outputs(model, dataset, atom_types, interaction_to_index, interaction_pairs, phase_tag):
    """Generate all output files for current model state."""
    print(f"\n--- Generating outputs for {phase_tag} ---")
    
    current_params = model.get_current_parameters()
    write_tersoff_potential_file("predicted_potential.tersoff", atom_types, interaction_to_index, current_params)
    create_energy_parity_plot(model, dataset, phase_tag)
    create_potential_curves_plot(current_params, interaction_pairs, phase_tag)
    create_force_parity_plot(model, dataset, phase_tag)

# ===== MAIN EXECUTION =====
def main():
    """Main training workflow."""
    print(f"Using device: {TRAINING_CONFIG['device']}")

    # Load and prepare data
    dataset = DFTDataset(
        xyz_path=DATA_FILES['xyz'],
        forces_path=DATA_FILES['forces'],
        cutoff=GRAPH_PARAMS['cutoff']
    )
    
    # Create DataLoader for mini-batching
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)

    # Setup interaction mapping
    atom_types = dataset.atom_types
    interaction_pairs = sorted([
        '-'.join(sorted(pair))
        for pair in combinations_with_replacement(atom_types, 2)
    ])
    interaction_to_index = {pair: idx for idx, pair in enumerate(interaction_pairs)}
    
    # Create interaction map tensor
    interaction_map = torch.zeros((len(atom_types), len(atom_types)), dtype=torch.long)
    for i, type1 in enumerate(atom_types):
        for j, type2 in enumerate(atom_types):
            pair_key = '-'.join(sorted([type1, type2]))
            interaction_map[i, j] = interaction_to_index[pair_key]
    
    print(f"Found {len(interaction_pairs)} interaction types: {interaction_pairs}")

    # Initialize model
    initial_params = setup_initial_parameters(atom_types, interaction_pairs)
    initial_params['E_ref'] = [dataset.energy_mean] * len(atom_types)
    
    model = TersoffGNN(
        initial_params=initial_params,
        interaction_map=interaction_map
    ).to(TRAINING_CONFIG['device'])

    # ===== PHASE 1: FIT PAIR POTENTIAL =====
    print("\n" + "="*50)
    print("PHASE 1: Fitting pair potential parameters")
    print("="*50)
    
    model.training_phase = 1
    pair_param_names = ['log_A', 'log_B', 'log_lambda1', 'log_lambda2', 'E_ref']
    
    for name, param in model.named_parameters():
        param.requires_grad = name in pair_param_names
    
    optimizer_phase1 = torch.optim.Adam(
        [p for name, p in model.named_parameters() if name in pair_param_names],
        lr=TRAINING_CONFIG['learning_rates']['phase1'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    model, losses_phase1, force_rmses_phase1 = train_model_phase(
        model, optimizer_phase1, dataloader,
        TRAINING_CONFIG['max_epochs'], TRAINING_CONFIG['patience'],
        TRAINING_CONFIG['tolerance'], "Phase_1",
        energy_weight=TRAINING_CONFIG['energy_weight'],
        force_weight=TRAINING_CONFIG['force_weight']
    )
    
    generate_all_outputs(model, dataset, atom_types, interaction_to_index, interaction_pairs, "Phase 1")

    # ===== PHASE 2: FIT ANGULAR TERMS =====
    print("\n" + "="*50)
    print("PHASE 2: Fitting angular interaction parameters")
    print("="*50)
    
    model.training_phase = 2
    
    pair_param_names = ['log_A', 'log_B', 'log_lambda1', 'log_lambda2', 'E_ref']
    angular_param_names = ['log_lambda3', 'log_beta', 'log_gamma', 'log_c', 'log_d']
    phase2_trainable_names = pair_param_names + angular_param_names

    for name, param in model.named_parameters():
        param.requires_grad = any(trainable_name in name for trainable_name in phase2_trainable_names)
    
    optimizer_phase2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAINING_CONFIG['learning_rates']['phase2'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    model, losses_phase2, force_rmses_phase2 = train_model_phase(
        model, optimizer_phase2, dataloader,
        TRAINING_CONFIG['max_epochs'], TRAINING_CONFIG['patience'],
        TRAINING_CONFIG['tolerance'], "Phase_2",
        energy_weight=TRAINING_CONFIG['energy_weight'],
        force_weight=TRAINING_CONFIG['force_weight']
    )
    
    generate_all_outputs(model, dataset, atom_types, interaction_to_index, interaction_pairs, "Phase 2")

    # ===== PHASE 3: FINE-TUNING ALL PARAMETERS =====
    print("\n" + "="*50)
    print("PHASE 3: Fine-tuning all parameters")
    print("="*50)
    
    model.training_phase = 3
    
    # *** MODIFIED: Train 'n' along with all other parameters for final tuning ***
    pair_param_names = ['log_A', 'log_B', 'log_lambda1', 'log_lambda2', 'E_ref']
    angular_param_names = ['log_lambda3', 'log_beta', 'log_gamma', 'log_c', 'log_d']
    n_param_name = ['log_n']
    phase3_trainable_names = pair_param_names + angular_param_names + n_param_name

    for name, param in model.named_parameters():
        param.requires_grad = any(trainable_name in name for trainable_name in phase3_trainable_names)
    
    optimizer_phase3 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAINING_CONFIG['learning_rates']['phase3'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    model, losses_phase3, force_rmses_phase3 = train_model_phase(
        model, optimizer_phase3, dataloader,
        TRAINING_CONFIG['max_epochs'], TRAINING_CONFIG['patience'],
        TRAINING_CONFIG['tolerance'], "Phase_3",
        energy_weight=TRAINING_CONFIG['energy_weight'],
        force_weight=TRAINING_CONFIG['force_weight']
    )
    
    generate_all_outputs(model, dataset, atom_types, interaction_to_index, interaction_pairs, "Phase 3")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*50)
    print("TRAINING COMPLETE - Generating final summary")
    print("="*50)
    
    all_losses = [losses_phase1, losses_phase2, losses_phase3]
    all_force_rmses = [force_rmses_phase1, force_rmses_phase2, force_rmses_phase3]
    phase_labels = ['Phase 1: Pair Potential', 'Phase 2: Angular Terms', 'Phase 3: Fine-tuning']
    create_training_curve_plot(all_losses, phase_labels)
    create_force_rmse_plot(all_force_rmses, phase_labels)
    
    print("\nTraining completed successfully!")
    print("Generated files:")
    print("  - predicted_potential.tersoff (LAMMPS format)")
    print("  - parity_plot.png (energy predictions)")
    print("  - force_parity_plot.png (force predictions)")
    print("  - potentials_comparison.png (fitted curves)")
    print("  - training_curve.png (loss evolution)")
    print("  - force_rmse_curve.png (Force RMSE evolution)")

if __name__ == "__main__":
    main()
