import numpy as np
import ot
from scipy.optimize import nnls
from scipy import sparse
from scipy.spatial import KDTree


def _to_numpy(data):
    if hasattr(data, "X"):
        data = data.X
    if sparse.issparse(data):
        data = data.toarray()
    return np.asarray(data)


def map_rna_cells(
    adata_sc_emb,
    adata_st_emb,
    gene_emb,
    label="Ground_Truth",
    k_cells_per_spot=5,
    alpha=0.5,
):
    """Map single-cell embeddings to spatial spots.

    The workflow combines FGW optimal transport for spot-cell matching,
    NNLS-based cell type quotas per spot, and local geometric spreading to
    assign sub-spot spatial coordinates.
    """
    sc_emb = _to_numpy(adata_sc_emb.X)
    st_emb = _to_numpy(adata_st_emb.X)
    gene_emb = _to_numpy(gene_emb)

    cell_types = adata_sc_emb.obs[label].values
    unique_types = np.unique(cell_types)

    type_centroids = {}
    for cell_type in unique_types:
        indices = np.where(cell_types == cell_type)[0]
        type_centroids[cell_type] = np.mean(sc_emb[indices], axis=0)

    n_spots = adata_st_emb.shape[0]
    n_cells = adata_sc_emb.shape[0]
    spot_coords = adata_st_emb.obsm["spatial"]
    print(f"Mapping {n_cells} single cells to {n_spots} spatial spots.")

    print("Step 1: Preparing FGW input matrices.")
    spot_cell_sim = st_emb @ sc_emb.T
    spot_cell_sim_min, spot_cell_sim_max = spot_cell_sim.min(), spot_cell_sim.max()
    spot_cell_sim_norm = (
        (spot_cell_sim - spot_cell_sim_min)
        / (spot_cell_sim_max - spot_cell_sim_min + 1e-8)
    )
    cross_domain_cost = np.asarray(1.0 - spot_cell_sim_norm, dtype=np.float64)

    spot_sim = st_emb @ st_emb.T
    spot_sim_min, spot_sim_max = spot_sim.min(), spot_sim.max()
    spot_sim_norm = (spot_sim - spot_sim_min) / (spot_sim_max - spot_sim_min + 1e-8)
    spot_cost = np.asarray(1.0 - spot_sim_norm, dtype=np.float64)

    cell_sim = sc_emb @ sc_emb.T
    cell_sim_min, cell_sim_max = cell_sim.min(), cell_sim.max()
    cell_sim_norm = (cell_sim - cell_sim_min) / (cell_sim_max - cell_sim_min + 1e-8)
    cell_cost = np.asarray(1.0 - cell_sim_norm, dtype=np.float64)

    spot_mass = np.ones((n_spots,)) / n_spots
    cell_mass = np.ones((n_cells,)) / n_cells

    print(f"Step 2: Running Fused Gromov-Wasserstein OT (alpha={alpha}).")
    transport = ot.gromov.fused_gromov_wasserstein(
        cross_domain_cost,
        spot_cost,
        cell_cost,
        spot_mass,
        cell_mass,
        loss_fun="square_loss",
        alpha=alpha,
        verbose=True,
    )

    print("Step 3: Estimating spot-level cell type quotas with NNLS.")
    type_profiles = np.array(
        [type_centroids[cell_type] @ gene_emb.T for cell_type in unique_types]
    ).T

    spot_quotas = []
    for spot_idx in range(n_spots):
        spot_expr = st_emb[spot_idx] @ gene_emb.T
        type_ratio, _ = nnls(type_profiles, spot_expr)

        type_ratio_sum = np.sum(type_ratio)
        if type_ratio_sum == 0:
            type_ratio = np.zeros(len(type_ratio))
        else:
            type_ratio = type_ratio / type_ratio_sum

        quota = {
            unique_types[type_idx]: int(
                np.round(type_ratio[type_idx] * k_cells_per_spot)
            )
            for type_idx in range(len(unique_types))
        }
        spot_quotas.append(quota)

    print("Step 4: Assigning cells by OT probability and spot quotas.")
    flat_indices = np.argsort(transport.flatten())[::-1]
    assigned_cell_to_spot = {}
    current_quotas = [quota.copy() for quota in spot_quotas]

    for flat_idx in flat_indices:
        spot_idx, cell_idx = divmod(flat_idx, n_cells)
        cell_type = cell_types[cell_idx]

        if (
            cell_idx not in assigned_cell_to_spot
            and current_quotas[spot_idx][cell_type] > 0
        ):
            assigned_cell_to_spot[cell_idx] = spot_idx
            current_quotas[spot_idx][cell_type] -= 1

    unassigned_cells = set(range(n_cells)) - set(assigned_cell_to_spot.keys())
    print(f"Step 5: Assigning remaining cells ({len(unassigned_cells)} cells).")

    for cell_idx in unassigned_cells:
        cell_type = cell_types[cell_idx]
        cell_probs = transport[:, cell_idx]
        available_spots = [
            spot_idx
            for spot_idx in range(n_spots)
            if current_quotas[spot_idx][cell_type] > 0
        ]

        if len(available_spots) > 0:
            best_spot_idx = available_spots[np.argmax(cell_probs[available_spots])]
            assigned_cell_to_spot[cell_idx] = best_spot_idx
            current_quotas[best_spot_idx][cell_type] -= 1

    assigned_cells = sorted(assigned_cell_to_spot.keys())
    print(f"Assigned {len(assigned_cells)} cells.")

    spot_to_cells = {spot_idx: [] for spot_idx in range(n_spots)}
    for cell_idx, spot_idx in assigned_cell_to_spot.items():
        spot_to_cells[spot_idx].append(cell_idx)

    print("Step 6: Computing sub-spot coordinates.")
    mapped_spot_emb = np.zeros_like(st_emb)
    for spot_idx in range(n_spots):
        cells_in_spot = spot_to_cells[spot_idx]
        if len(cells_in_spot) > 0:
            mapped_spot_emb[spot_idx] = np.mean(sc_emb[cells_in_spot], axis=0)
        else:
            mapped_spot_emb[spot_idx] = st_emb[spot_idx]

    spatial_tree = KDTree(spot_coords)
    distances, _ = spatial_tree.query(spot_coords, k=2)
    mean_radius = np.mean(distances[:, 1]) / 2.0
    final_cell_coords = np.zeros((n_cells, 2))

    for spot_idx in range(n_spots):
        cells_in_spot = spot_to_cells[spot_idx]
        if len(cells_in_spot) == 0:
            continue

        spot_coord = spot_coords[spot_idx]
        distances, neighbor_indices = spatial_tree.query(spot_coord, k=7)
        valid_neighbors = [
            neighbor_idx
            for distance, neighbor_idx in zip(distances, neighbor_indices)
            if distance > 1e-5
        ][:6]

        spot_cell_coords = []
        for cell_idx in cells_in_spot:
            cell_emb = sc_emb[cell_idx]
            neighbor_similarities = []
            vectors_to_neighbors = []

            for neighbor_idx in valid_neighbors:
                neighbor_emb = mapped_spot_emb[neighbor_idx]
                neighbor_coord = spot_coords[neighbor_idx]
                neighbor_similarities.append(np.dot(cell_emb, neighbor_emb))
                vectors_to_neighbors.append(neighbor_coord - spot_coord)

            neighbor_similarities = np.array(neighbor_similarities)
            vectors_to_neighbors = np.array(vectors_to_neighbors)

            if len(neighbor_similarities) > 2:
                sim_min = neighbor_similarities.min()
                sim_max = neighbor_similarities.max()
                if sim_max > sim_min:
                    sim_weights = (
                        (neighbor_similarities - sim_min)
                        / (sim_max - sim_min)
                    )
                else:
                    sim_weights = np.zeros_like(neighbor_similarities)
            else:
                sim_weights = neighbor_similarities

            if np.sum(sim_weights) > 0:
                shift_vector = np.average(
                    vectors_to_neighbors,
                    axis=0,
                    weights=sim_weights,
                )
            elif len(vectors_to_neighbors) > 0:
                shift_vector = np.mean(vectors_to_neighbors, axis=0)
            else:
                shift_vector = np.zeros(2)

            spot_cell_coords.append(spot_coord + shift_vector)

        spot_cell_coords = np.array(spot_cell_coords)

        if len(spot_cell_coords) > 1:
            midpoint = np.mean(spot_cell_coords, axis=0)
            adjusted_coords = spot_cell_coords + spot_coord - midpoint

            diffs = adjusted_coords - spot_coord
            sq_dists = np.sum(diffs**2, axis=1)
            max_dist = np.sqrt(np.max(sq_dists))

            if max_dist > 0:
                ratio = mean_radius / max_dist
                adjusted_coords = diffs * ratio + spot_coord
        else:
            adjusted_coords = np.array([spot_coord])

        for idx, cell_idx in enumerate(cells_in_spot):
            final_cell_coords[cell_idx] = adjusted_coords[idx]

    adata_sc_mapped = adata_sc_emb[assigned_cells].copy()
    adata_sc_mapped.obsm["spatial"] = final_cell_coords[assigned_cells]
    adata_sc_mapped.obs["assigned_spot_id"] = [
        adata_st_emb.obs_names[assigned_cell_to_spot[cell_idx]]
        for cell_idx in assigned_cells
    ]

    print(f"Finished mapping. Output contains {adata_sc_mapped.shape[0]} cells.")
    return adata_sc_mapped, transport, spot_quotas




def map_atac_cells(
    adata_st_emb,
    adata_rna_emb,
    adata_atac_emb,
    pi_spot_rna,
    spot_quotas,
    rna_celltype_key="celltype",
    atac_celltype_key="celltype",
    neighbor_k=6,
    weighted_average=True,
    scale_similarity=True,
):
    """Map ATAC cells to spatial spots through RNA-guided transport.

    RNA-to-ATAC transport is computed within matched cell types only, then
    composed with the spot-to-RNA transport matrix. Spot quotas guide the final
    ATAC-to-spot assignment, followed by local coordinate refinement.
    """

    def scale01(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            return (x - xmin) / (xmax - xmin)
        return np.zeros_like(x)

    st_emb = _to_numpy(adata_st_emb.X)
    rna_emb = _to_numpy(adata_rna_emb.X)
    atac_emb = _to_numpy(adata_atac_emb.X)
    spot_coords = np.asarray(adata_st_emb.obsm["spatial"])

    spot_names = np.asarray(adata_st_emb.obs_names).astype(str)
    rna_names = np.asarray(adata_rna_emb.obs_names).astype(str)
    atac_names = np.asarray(adata_atac_emb.obs_names).astype(str)

    n_spots = len(spot_names)
    n_rna = len(rna_names)
    n_atac = len(atac_names)

    expected_shape = (n_spots, n_rna)
    if pi_spot_rna.shape != expected_shape:
        raise ValueError(
            f"pi_spot_rna shape mismatch: expected {expected_shape}, "
            f"got {pi_spot_rna.shape}."
        )

    rna_ct = adata_rna_emb.obs[rna_celltype_key].astype(str).values
    atac_ct = adata_atac_emb.obs[atac_celltype_key].astype(str).values

    unique_types = np.intersect1d(np.unique(rna_ct), np.unique(atac_ct))

    print("Step 1: Computing cell type-constrained RNA-to-ATAC OT.")
    pi_rna_atac = np.zeros((n_rna, n_atac), dtype=float)

    for ct in unique_types:
        idx_r = np.where(rna_ct == ct)[0]
        idx_a = np.where(atac_ct == ct)[0]

        if len(idx_r) == 0 or len(idx_a) == 0:
            continue

        sub_rna = rna_emb[idx_r]
        sub_atac = atac_emb[idx_a]

        similarity = sub_rna @ sub_atac.T
        sim_min, sim_max = similarity.min(), similarity.max()
        if sim_max > sim_min:
            similarity_norm = (similarity - sim_min) / (sim_max - sim_min + 1e-8)
        else:
            similarity_norm = similarity - sim_min
        cost = 1.0 - similarity_norm

        rna_mass = np.ones(len(idx_r)) / len(idx_r)
        atac_mass = np.ones(len(idx_a)) / len(idx_a)

        pi_rna_atac[np.ix_(idx_r, idx_a)] = ot.emd(rna_mass, atac_mass, cost)

    print("Step 2: Composing spot-to-RNA and RNA-to-ATAC transport.")
    pi_spot_atac = pi_spot_rna @ pi_rna_atac

    print("Step 3: Assigning ATAC cells by transport score and spot quotas.")
    flat_indices = np.argsort(pi_spot_atac.flatten())[::-1]
    assigned_cell_to_spot = {}
    current_quotas = [q.copy() for q in spot_quotas]

    for idx in flat_indices:
        spot_idx, atac_idx = divmod(idx, n_atac)
        cell_type = atac_ct[atac_idx]

        if (
            atac_idx not in assigned_cell_to_spot
            and current_quotas[spot_idx].get(cell_type, 0) > 0
        ):
            assigned_cell_to_spot[atac_idx] = spot_idx
            current_quotas[spot_idx][cell_type] -= 1

    unassigned_cells = set(range(n_atac)) - set(assigned_cell_to_spot.keys())
    if len(unassigned_cells) > 0:
        for atac_idx in unassigned_cells:
            cell_type = atac_ct[atac_idx]
            cell_probs = pi_spot_atac[:, atac_idx]
            available_spots = [
                spot_idx
                for spot_idx in range(n_spots)
                if current_quotas[spot_idx].get(cell_type, 0) > 0
            ]

            if len(available_spots) > 0:
                best_spot_idx = available_spots[np.argmax(cell_probs[available_spots])]
                assigned_cell_to_spot[atac_idx] = best_spot_idx
                current_quotas[best_spot_idx][cell_type] -= 1

    assigned_cells = sorted(assigned_cell_to_spot.keys())
    print(f"Assigned {len(assigned_cells)} ATAC cells.")

    spot_to_cells = {spot_idx: [] for spot_idx in range(n_spots)}
    for atac_idx, spot_idx in assigned_cell_to_spot.items():
        spot_to_cells[spot_idx].append(atac_idx)

    print("Step 4: Computing sub-spot coordinates.")
    mapped_spot_emb = np.zeros_like(st_emb)
    for spot_idx in range(n_spots):
        cells_in_spot = spot_to_cells[spot_idx]
        if len(cells_in_spot) > 0:
            mapped_spot_emb[spot_idx] = np.mean(atac_emb[cells_in_spot], axis=0)
        else:
            mapped_spot_emb[spot_idx] = st_emb[spot_idx]

    spatial_tree = KDTree(spot_coords)
    if n_spots >= 2:
        distances, _ = spatial_tree.query(spot_coords, k=2)
        mean_radius = np.mean(distances[:, 1]) / 2.0
    else:
        mean_radius = 1.0

    final_cell_coords = np.zeros((n_atac, 2))

    for spot_idx in range(n_spots):
        cells_in_spot = spot_to_cells[spot_idx]
        if len(cells_in_spot) == 0:
            continue

        spot_coord = spot_coords[spot_idx]
        dists, all_neighbor_indices = spatial_tree.query(spot_coord, k=neighbor_k + 1)
        valid_neighbors = [
            neighbor_idx
            for distance, neighbor_idx in zip(dists, all_neighbor_indices)
            if distance > 1e-5
        ][:neighbor_k]

        spot_cell_coords = []

        for cell_idx in cells_in_spot:
            cell_emb = atac_emb[cell_idx]

            neighbor_similarities = []
            vectors_to_neighbors = []

            for neighbor_idx in valid_neighbors:
                neighbor_emb = mapped_spot_emb[neighbor_idx]
                neighbor_coord = spot_coords[neighbor_idx]

                neighbor_similarities.append(np.dot(cell_emb, neighbor_emb))
                vectors_to_neighbors.append(neighbor_coord - spot_coord)

            neighbor_similarities = np.array(neighbor_similarities)
            vectors_to_neighbors = np.array(vectors_to_neighbors)

            if scale_similarity and len(neighbor_similarities) > 2:
                sim_weights = scale01(neighbor_similarities)
            else:
                sim_weights = neighbor_similarities

            if weighted_average and np.sum(sim_weights) > 0:
                shift_vector = np.average(
                    vectors_to_neighbors,
                    axis=0,
                    weights=sim_weights,
                )
            elif len(vectors_to_neighbors) > 0:
                shift_vector = np.mean(vectors_to_neighbors, axis=0)
            else:
                shift_vector = np.zeros(2)

            spot_cell_coords.append(spot_coord + shift_vector)

        spot_cell_coords = np.array(spot_cell_coords)

        if len(spot_cell_coords) > 1:
            midpoint = np.mean(spot_cell_coords, axis=0)
            adjusted_coords = spot_cell_coords + spot_coord - midpoint

            diffs = adjusted_coords - spot_coord
            sq_dists = np.sum(diffs**2, axis=1)
            max_dist = np.sqrt(np.max(sq_dists))

            if max_dist > 0:
                ratio = mean_radius / max_dist
                adjusted_coords = diffs * ratio + spot_coord
        else:
            adjusted_coords = np.array([spot_coord])

        for idx, cell_idx in enumerate(cells_in_spot):
            final_cell_coords[cell_idx] = adjusted_coords[idx]

    adata_atac_mapped = adata_atac_emb[assigned_cells].copy()
    adata_atac_mapped.obsm["spatial"] = final_cell_coords[assigned_cells]
    adata_atac_mapped.obs["assigned_spot_id"] = [
        spot_names[assigned_cell_to_spot[cell_idx]]
        for cell_idx in assigned_cells
    ]

    print(f"Finished mapping. Output contains {adata_atac_mapped.shape[0]} ATAC cells.")
    return adata_atac_mapped, pi_spot_atac
