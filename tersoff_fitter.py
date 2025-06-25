#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add
from torch_cluster import radius


#input files, parameters to be adjusted
LAMMPS_DUMP_PATH = "tersoff_ML_450K_50atom.lammpstrj"
ENERGY_TXT_PATH  = "extracted_energies.txt"

GRAPH_CUTOFF = 6
START_TIMESTEP = 100
ENERGY_SKIP_LINES = 0

TYPE_TO_INDEX = {1: 0, 2: 1} #1 is Pb, 2 is S
N_ATOM_FEATURE = 2

#hyperparameters
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
NUM_EPOCHS = 200
LR_TERSOFF_PARAMS = 5e-4
WEIGHT_DECAY = 0

INITIAL_PARAMS = {
    'A':         [1500.0,    3295.0,    1500.0], # Pb-Pb, Pb-S, S-S
    'B':         [20.0,      332.1,     20.0],
    'lambda1':   [2.5,       2.7,       2.5],
    'lambda2':   [1.5,       1.8,       1.5],
    'beta':      [1.0e-6,    1.5724e-6, 1.0e-6],
    'n':         [1.0,       0.78734,   1.0]
}
#true parameters just used for plotting as well as setting the values not being parameterized
TRUE_PARAMS = {
    'A':         [1500.0,    3295.0,    1500.0], # Pb-Pb, Pb-S, S-S
    'B':         [20.0,      332.1,     20.0],
    'lambda1':   [2.5,       2.7,       2.5],
    'lambda2':   [1.5,       1.8,       1.5],
    'beta':      [1.0e-6,    1.5724e-6, 1.0e-6],
    'n':         [1.0,       0.78734,   1.0],
    'R':         [4.0,       3.5,       4.0],
    'D':         [0.2,       0.2,       0.2],
}

#reading lammps and energy files, making any calculations necessary
class RawDFTLammpsDataset(Dataset):
    def __init__(self, dump_path, energy_path, cutoff, start_timestep, energy_skip_lines):
        super().__init__()

        print(f"Loading energies from '{energy_path}'...")
        raw_total_energies = np.loadtxt(energy_path, skiprows=energy_skip_lines)

        self.frames = []
        self.num_atoms_list = []

        print(f"Scanning trajectory '{dump_path}'...")
        with open(dump_path, 'r') as f:
            is_first_frame = True
            while True:
                line = f.readline()
                if not line: break

                if line.strip() == "ITEM: TIMESTEP":
                    current_timestep = int(f.readline().strip())

                    if current_timestep < start_timestep:
                        f.readline()
                        N = int(f.readline().strip())
                        for _ in range(7 + N): f.readline()
                        continue

                    if is_first_frame:
                        print(f"Found start. Processing from timestep {current_timestep}.")
                        is_first_frame = False

                    f.readline()
                    N = int(f.readline().strip())
                    f.readline()
                    bounds_x = list(map(float, f.readline().split()))
                    bounds_y = list(map(float, f.readline().split()))
                    bounds_z = list(map(float, f.readline().split()))

                    header_line = f.readline()
                    tokens = header_line.split()
                    t_type_idx = tokens.index("type") - 2
                    is_scaled = 'xs' in tokens
                    if is_scaled:
                        pos_indices = (tokens.index("xs")-2, tokens.index("ys")-2, tokens.index("zs")-2)
                    else:
                        pos_indices = (tokens.index("x")-2, tokens.index("y")-2, tokens.index("z")-2)

                    types_list, coords_list = [], []
                    for _ in range(N):
                        parts = f.readline().split()
                        types_list.append(int(parts[t_type_idx]))
                        coords = [float(parts[i]) for i in pos_indices]
                        #if xs ys zs, rescale to true xyz values using boundary box dimensions
                        if is_scaled:
                            coords[0] = bounds_x[0] + coords[0] * (bounds_x[1] - bounds_x[0])
                            coords[1] = bounds_y[0] + coords[1] * (bounds_y[1] - bounds_y[0])
                            coords[2] = bounds_z[0] + coords[2] * (bounds_z[1] - bounds_z[0])
                        coords_list.append(coords)

                    self.frames.append((np.array(types_list), np.array(coords_list, dtype=np.float32), bounds_x + bounds_y + bounds_z))
                    self.num_atoms_list.append(N)

        #make sure number of lines in energy file matches number of timesteps
        n_f, n_e = len(self.frames), len(raw_total_energies)
        if n_f != n_e:
            L = min(n_f, n_e)
            print(f"Warning: Frame count ({n_f}) and energy count ({n_e}) mismatch. Truncating to {L}.")
            self.frames, self.num_atoms_list, raw_total_energies = self.frames[:L], self.num_atoms_list[:L], raw_total_energies[:L]

        self.num_atoms_array = np.array(self.num_atoms_list)
        self.energies_per_atom = raw_total_energies / self.num_atoms_array #energies per atom
        self.raw_total_energies = raw_total_energies
        self.cutoff = cutoff

    def len(self):
        return len(self.frames)

    def get(self, idx):
        types_np, pos_np, box_bounds = self.frames[idx]

        #create the nodes
        x = torch.zeros((len(types_np), N_ATOM_FEATURE), dtype=torch.float)
        for i, t in enumerate(types_np):
            x[i, TYPE_TO_INDEX[t]] = 1.0

        #setup recognizing periodic conditions
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        Lx, Ly, Lz = xhi - xlo, yhi - ylo, zhi - zlo
        pos_tensor = torch.tensor(pos_np - np.array([xlo, ylo, zlo]), dtype=torch.float)
        N = len(types_np)

        all_pos_images, all_idx_images = [], []
        for sx in [-1, 0, 1]:
            for sy in [-1, 0, 1]:
                for sz in [-1, 0, 1]:
                    shift = torch.tensor([sx*Lx, sy*Ly, sz*Lz], dtype=torch.float)
                    all_pos_images.append(pos_tensor + shift)
                    all_idx_images.append(torch.arange(N, dtype=torch.long))

        #create edges
        pos_images_tensor = torch.cat(all_pos_images, dim=0)
        idx_images_tensor = torch.cat(all_idx_images, dim=0)

        img_all, src_all = radius(pos_tensor, pos_images_tensor, self.cutoff, max_num_neighbors=512)
        dst_all = idx_images_tensor[img_all]

        #make sure atoms aren't counted twice or inside eachother (any interaction of distence x10-6 is ignored)
        mask = (src_all != dst_all) | (img_all >= N)
        src, dst, img = src_all[mask], dst_all[mask], img_all[mask]

        dists = torch.norm(pos_images_tensor[img] - pos_tensor[src], dim=1)

        valid_dist_mask = dists > 1e-6

        edge_index = torch.stack([src[valid_dist_mask], dst[valid_dist_mask]], dim=0)
        edge_attr = dists[valid_dist_mask].view(-1, 1)

        return Data(
            x=x,
            pos=pos_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([self.energies_per_atom[idx]], dtype=torch.float),
            y_total=torch.tensor([self.raw_total_energies[idx]], dtype=torch.float)
        )

class TersoffGNN(nn.Module):
    def __init__(self, init_params):
        super().__init__()
        for param_name in ['A', 'B', 'lambda1', 'lambda2', 'beta', 'n']:
            values = init_params[param_name]
            log_values = [nn.Parameter(torch.tensor(math.log(v), dtype=torch.float)) for v in values]
            setattr(self, f"log_{param_name}_PbPb", log_values[0])
            setattr(self, f"log_{param_name}_PbS",  log_values[1])
            setattr(self, f"log_{param_name}_SS",   log_values[2])
        self.R_params = torch.tensor(init_params['R'], dtype=torch.float)
        self.D_params = torch.tensor(init_params['D'], dtype=torch.float)

    def f_c(self, r, R, D):
        fc_values = torch.zeros_like(r)
        cond1 = r < (R - D)
        fc_values[cond1] = 1.0
        cond2 = (r >= (R - D)) & (r < (R + D))
        R_broad = R.expand_as(r)
        D_broad = D.expand_as(r)
        arg = math.pi * (r[cond2] - (R_broad[cond2] - D_broad[cond2])) / (2 * D_broad[cond2])
        fc_values[cond2] = 0.5 * (1 + torch.cos(arg))
        return fc_values

    def forward(self, data):
        params = {p: torch.exp(torch.stack([getattr(self, f"log_{p}_{t}") for t in ["PbPb", "PbS", "SS"]])) for p in ['A','B','lambda1','lambda2','beta','n']}
        i, j = data.edge_index
        r_ij = data.edge_attr[:, 0]
        mask_PbPb=data.x[i,0]*data.x[j,0]; mask_PbS=(data.x[i,0]*data.x[j,1])+(data.x[i,1]*data.x[j,0]); mask_SS=data.x[i,1]*data.x[j,1]
        edge_param_indices = (mask_PbPb * 0 + mask_PbS * 1 + mask_SS * 2).long()
        A_ij,B_ij,l1_ij,l2_ij,beta_ij,n_ij = [params[p][edge_param_indices] for p in ['A','B','lambda1','lambda2','beta','n']]
        R_edge = self.R_params.to(r_ij.device)[edge_param_indices]
        D_edge = self.D_params.to(r_ij.device)[edge_param_indices]
        f_R_raw = A_ij * torch.exp(-l1_ij * r_ij)
        f_A_raw = -B_ij * torch.exp(-l2_ij * r_ij)
        fc_ij = self.f_c(r_ij, R_edge, D_edge)
        sum_fc_per_atom_i = scatter_add(fc_ij, i, dim=0, dim_size=data.num_nodes)
        zeta_ij = torch.clamp(sum_fc_per_atom_i[i] - fc_ij, min=0)
        eps = 1e-15
        term_b = torch.pow(beta_ij * zeta_ij + eps, n_ij)
        b_ij = torch.pow(1.0 + term_b, -1.0 / (2.0 * n_ij))
        V_ij = fc_ij * (f_R_raw + b_ij * f_A_raw)
        total_energy = global_add_pool(0.5 * V_ij, data.batch[i])
        _, n_atoms_per_graph = torch.unique(data.batch, return_counts=True)
        return total_energy.squeeze(-1) / n_atoms_per_graph


def train_model(model, train_loader, num_epochs, lr, weight_decay, true_params):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    #LR divided by 2 if MSE does not change after 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-7)

    train_losses = []

    log_bounds = {
        'A': (math.log(100), math.log(5000)), 
        'B': (math.log(1), math.log(1000)),
        'lambda1': (math.log(1.0), math.log(5.0)), 
        'lambda2': (math.log(0.5), math.log(3.0)),
        'beta': (math.log(1e-8), math.log(1)), 
        'n': (math.log(0.1), math.log(10))
    }

    # --- Helper function to calculate energy using a given parameter set ---
    def calculate_true_energy(batch, true_params_dict):
        true_tensors = {k: torch.tensor(v, device=DEVICE) for k, v in true_params_dict.items()}

        i, j = batch.edge_index
        r_ij = batch.edge_attr[:, 0]

        mask_PbPb = batch.x[i, 0] * batch.x[j, 0]
        mask_PbS  = (batch.x[i, 0] * batch.x[j, 1]) + (batch.x[i, 1] * batch.x[j, 0])
        mask_SS   = batch.x[i, 1] * batch.x[j, 1]
        edge_param_indices = (mask_PbPb * 0 + mask_PbS * 1 + mask_SS * 2).long()

        A_ij, B_ij = true_tensors['A'][edge_param_indices], true_tensors['B'][edge_param_indices]
        lambda1_ij, lambda2_ij = true_tensors['lambda1'][edge_param_indices], true_tensors['lambda2'][edge_param_indices]
        beta_ij, n_ij = true_tensors['beta'][edge_param_indices], true_tensors['n'][edge_param_indices]
        R_edge, D_edge = true_tensors['R'][edge_param_indices], true_tensors['D'][edge_param_indices]

        fc_ij = model.f_c(r_ij, R_edge, D_edge)
        f_R = (A_ij * torch.exp(-lambda1_ij * r_ij)) * fc_ij
        f_A = (-B_ij * torch.exp(-lambda2_ij * r_ij)) * fc_ij

        sum_fc_per_atom_i = scatter_add(fc_ij, i, dim=0, dim_size=batch.num_nodes)
        zeta_ij_per_edge = torch.clamp(sum_fc_per_atom_i[i] - fc_ij, min=0)

        eps = 1e-8
        term_b = torch.pow(beta_ij * zeta_ij_per_edge + eps, n_ij)
        b_ij = torch.pow(1.0 + term_b, -1.0 / (2.0 * n_ij))

        V_ij = f_R + b_ij * f_A
        total_pair_energy = 0.5 * V_ij
        total_energy = global_add_pool(total_pair_energy, batch.batch[i])

        _, n_atoms_per_graph = torch.unique(batch.batch, return_counts=True)
        return total_energy.squeeze(-1) / n_atoms_per_graph

    # Get the number of atoms for reporting. Assumes it's constant across the dataset.
    N_atoms = train_loader.dataset.num_atoms_array[0] # <-- ADDED

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, count = 0.0, 0

        if epoch == 1:
            print("\n--- Epoch 1: Verifying Inherent Model Error ---")
            true_param_losses = []
            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(DEVICE)
                    true_pred = calculate_true_energy(batch, true_params)
                    true_loss = loss_fn(true_pred, batch.y.view_as(true_pred))
                    true_param_losses.append(true_loss.item())
            avg_true_loss = np.mean(true_param_losses)
            inherent_rmse = N_atoms * np.sqrt(avg_true_loss) # <-- ADDED
            print(f"  -> Inherent Model MSE/atom (using TRUE_PARAMS): {avg_true_loss:.6f}")
            print(f"  -> Inherent Model Total RMSE (using TRUE_PARAMS): {inherent_rmse:.4f} eV") # <-- ADDED
            print("  -> This is the BEST possible performance the model can achieve.")
            print("--- Starting Training ---")

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view_as(pred))

            if torch.isnan(loss):
                print(f"Epoch {epoch}: Loss is NaN. Stopping training.")
                return model, train_losses

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p_name in ['A', 'B', 'lambda1', 'lambda2', 'beta', 'n']:
                    min_b, max_b = log_bounds[p_name]
                    getattr(model, f"log_{p_name}_PbPb").clamp_(min_b, max_b)
                    getattr(model, f"log_{p_name}_PbS").clamp_(min_b, max_b)
                    getattr(model, f"log_{p_name}_SS").clamp_(min_b, max_b)

            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

        avg_train_loss = total_loss / count
        train_losses.append(avg_train_loss)
        scheduler.step(avg_train_loss)

        if epoch % 10 == 0 or epoch == 1:
            # Report both the per-atom loss (for scheduler) and the total RMSE (for intuition)
            avg_total_rmse = N_atoms * math.sqrt(avg_train_loss) # <-- CHANGED
            print(f"Epoch {epoch}/{num_epochs}   Train MSE/atom={avg_train_loss:.6f}   Total RMSE={avg_total_rmse:.4f} eV   LR={optimizer.param_groups[0]['lr']:.2e}") # <-- CHANGED

    return model, train_losses

def plot_pred_vs_true(model, loader, tag):
    model.eval()
    preds_total, trues_total = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            p_per_atom = model(batch).cpu().numpy()

            _, n_atoms_per_graph = torch.unique(batch.batch, return_counts=True)
            n_atoms_per_graph = n_atoms_per_graph.cpu().numpy()

            preds_total.extend(p_per_atom * n_atoms_per_graph)
            trues_total.extend(batch.y_total.cpu().numpy())

    preds, trues = np.array(preds_total), np.array(trues_total)
    valid_indices = ~np.isnan(preds) & ~np.isnan(trues)
    preds, trues = preds[valid_indices], trues[valid_indices]

    if len(preds) == 0:
        print("Could not generate parity plot: no valid (non-NaN) data points.")
        return

    rmse = np.sqrt(np.mean((preds - trues)**2))

    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, s=10, alpha=0.5, label=f'RMSE = {rmse:.4f} eV')
    mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='y=x')
    plt.xlabel("True Total Energy (eV)"), plt.ylabel("Predicted Total Energy (eV)")
    plt.title(f"Parity Plot ({tag})"), plt.legend(), plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"parity_plot_{tag.lower()}.png", dpi=300)
    plt.close()
    print(f"Saved parity plot to parity_plot_{tag.lower()}.png")

def plot_training_curve(train_losses):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    valid_losses = [l for l in train_losses if not np.isnan(l)]
    valid_epochs = [e for e, l in zip(epochs, train_losses) if not np.isnan(l)]

    if not valid_losses:
        print("Could not generate training curve: no valid (non-NaN) loss values.")
        return

    plt.plot(valid_epochs, valid_losses)
    plt.yscale('log'), plt.title('Training Loss Curve (MSE/atom)')
    plt.xlabel('Epoch'), plt.ylabel('Mean Squared Error (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.savefig("training_curve.png", dpi=300)
    plt.close()
    print("Saved training curve to training_curve.png")

def plot_potentials(predicted_params):
    def two_body_tersoff(r, A, B, lambda1, lambda2):
        return A * np.exp(-lambda1 * r) - B * np.exp(-lambda2 * r)

    def fc_np(r, R, D):
        cond1 = (r < R - D)
        cond2 = (r >= R - D) & (r <= R)

        fc_values = np.zeros_like(r)
        fc_values[cond1] = 1.0
        fc_values[cond2] = 0.5 - 0.5 * np.sin(np.pi / 2 * (r[cond2] - R) / D)
        return fc_values

    interaction_types = ['Pb-Pb', 'Pb-S', 'S-S']
    r_range = np.linspace(1.8, 6.0, 200)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    true_R = TRUE_PARAMS['R']
    true_D = TRUE_PARAMS['D']
    pred_R = predicted_params.get('R', true_R) # Use true if not in pred
    pred_D = predicted_params.get('D', true_D)

    for i, label in enumerate(interaction_types):
        try:
            A_t, B_t = TRUE_PARAMS['A'][i], TRUE_PARAMS['B'][i]
            l1_t, l2_t = TRUE_PARAMS['lambda1'][i], TRUE_PARAMS['lambda2'][i]
            R_t, D_t = true_R[i], true_D[i]

            A_p, B_p = predicted_params['A'][i], predicted_params['B'][i]
            l1_p, l2_p = predicted_params['lambda1'][i], predicted_params['lambda2'][i]
            R_p, D_p = pred_R[i], pred_D[i]

            V_true_raw = two_body_tersoff(r_range, A_t, B_t, l1_t, l2_t)
            fc_true = fc_np(r_range, R_t, D_t)
            V_true = V_true_raw * fc_true

            V_pred_raw = two_body_tersoff(r_range, A_p, B_p, l1_p, l2_p)
            fc_pred = fc_np(r_range, R_p, D_p)
            V_pred = V_pred_raw * fc_pred

            ax = axes[i]
            ax.plot(r_range, V_true, 'r-', lw=2, label='True (2-body with cutoff)')
            ax.plot(r_range, V_pred, 'b--', lw=2, label='Predicted (2-body with cutoff)')

            ax.set_title(f'Tersoff 2-Body Potential: {label}'), ax.set_xlabel('Distance (Å)')
            if i == 0: ax.set_ylabel('Potential Energy (eV)')
            ax.grid(True), ax.legend()
            ax.axhline(0, color='k', linestyle=':', linewidth=0.5)

            min_well = min(np.nanmin(V_true), np.nanmin(V_pred), 0)
            max_height = max(np.nanmax(V_true), np.nanmax(V_pred), 0)
            ax.set_ylim(min_well * 1.2 - 0.1, max_height * 1.2 + 0.1)
        except (ValueError, TypeError) as e:
             print(f"Could not plot potential for {label} due to error: {e}")

    plt.tight_layout()
    plt.savefig("potentials_comparison.png", dpi=300)
    print("Saved potentials comparison plot to potentials_comparison.png")

def main():
    full_initial_params = {**INITIAL_PARAMS, 'R': TRUE_PARAMS['R'], 'D': TRUE_PARAMS['D']}

    dataset = RawDFTLammpsDataset(
        dump_path=LAMMPS_DUMP_PATH,
        energy_path=ENERGY_TXT_PATH,
        cutoff=GRAPH_CUTOFF,
        start_timestep=START_TIMESTEP,
        energy_skip_lines=ENERGY_SKIP_LINES
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"\nLoaded {len(dataset)} frames.")

    print("\n--- Using initial guess for Tersoff training ---")
    print("Initial Physical Parameters (Trainable):")
    for name, values in INITIAL_PARAMS.items():
        print(f"  {name:<8s} (Pb-Pb, Pb-S, S-S): {[f'{v:.4g}' for v in values]}")
    print("\nFixed Tersoff Parameters (Non-Trainable, from TRUE_PARAMS):")
    print(f"  R        (Pb-Pb, Pb-S, S-S): {[f'{v:.4g}' for v in TRUE_PARAMS['R']]}")
    print(f"  D        (Pb-Pb, Pb-S, S-S): {[f'{v:.4g}' for v in TRUE_PARAMS['D']]}")

    model = TersoffGNN(init_params=full_initial_params).to(DEVICE)

    trained_model, train_losses = train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR_TERSOFF_PARAMS,
        weight_decay=WEIGHT_DECAY,
        true_params=TRUE_PARAMS
    )

    print("\n--- Training Finished. Final Physical Parameters: ---")
    final_params = {}
    with torch.no_grad():
        for p_name in INITIAL_PARAMS.keys():
            final_params[p_name] = [
                torch.exp(getattr(trained_model, f"log_{p_name}_PbPb")).item(),
                torch.exp(getattr(trained_model, f"log_{p_name}_PbS")).item(),
                torch.exp(getattr(trained_model, f"log_{p_name}_SS")).item()
            ]
        final_params['R'] = model.R_params.tolist()
        final_params['D'] = model.D_params.tolist()

    for name, values in final_params.items():
        if isinstance(values, list):
            print(f"  {name:<8s} (Pb-Pb, Pb-S, S-S): {[f'{v:.4g}' for v in values]}")
        else:
            print(f"  {name:<8s}: {values:.4g}")

    plot_training_curve(train_losses)
    plot_pred_vs_true(trained_model, train_loader, "Training")
    plot_potentials(final_params)

if __name__ == "__main__":
    main()
