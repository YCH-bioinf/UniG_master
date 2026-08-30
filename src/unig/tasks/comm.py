from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def install_numpy_compat_aliases():
    """Add NumPy aliases needed by older dependencies."""
    if "object" not in np.__dict__:
        np.object = object
    if "float" not in np.__dict__:
        np.float = float


def register_anndata_null_reader():
    """Register a reader for legacy AnnData null datasets."""
    import h5py
    from anndata._io.specs import IOSpec, _REGISTRY

    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def read_null(elem, *, _reader):
            return None
    except Exception as exc:
        message = str(exc).lower()
        if "already" not in message and "duplicate" not in message:
            raise
        read_null = None

    return read_null


def load_stcase_source(stcase_source, reload_ccci=True):
    """Load the local UniG STCase source package."""
    stcase_source = Path(stcase_source)
    package_parent = stcase_source.parent if stcase_source.name == "UniG_STCase_src" else stcase_source
    path_str = str(package_parent)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    import UniG_STCase_src as st

    if reload_ccci:
        st.ccci = importlib.reload(st.ccci)
    return st


def load_stcase_databases(stcase_root, species="Human"):
    """Load STCase interaction and complex databases."""
    db_dir = Path(stcase_root) / "STCaseDB"
    interaction = pd.read_csv(db_dir / f"STCaseDB_{species}.csv", index_col=0)
    complex_table = pd.read_csv(db_dir / f"STCaseDB_{species}_Complex.csv", index_col=0)
    return interaction, complex_table


def attach_unig_clusters_to_spatial_adata(
    st_adata,
    unig_adata_dict,
    entity_key="C",
    cluster_key="mclust",
    output_key="type",
):
    """Attach UniG cluster labels to a spatial RNA AnnData object."""
    adata_c = unig_adata_dict[entity_key]
    adata_c = adata_c[st_adata.obs_names, :]
    st_adata = st_adata.copy()
    st_adata.obs[output_key] = adata_c.obs[cluster_key].values
    return st_adata


def run_stcase_communication(
    st_module,
    st_adata,
    db_interaction,
    db_complex,
    ct_key="type",
    method="Hill",
    cell_type=None,
    if_hvg=False,
    if_filter=False,
    if_self=True,
    if_intra=True,
    if_stringent=False,
    intra_method="unig",
    unig_network_path=None,
    unig_tf_activity_path=None,
    min_regulon_targets=10,
    background_number=1000,
    threads=10,
    scope=6,
    min_exp=0.1,
    cutoff=0.01,
):
    """Run STCase spatial cell-cell communication."""
    return st_module.ccci.spatial_cell_communication_run(
        st_adata,
        db_interaction,
        db_complex,
        method=method,
        ct_key=ct_key,
        cell_type=cell_type,
        if_hvg=if_hvg,
        if_filter=if_filter,
        if_self=if_self,
        if_intra=if_intra,
        if_stringent=if_stringent,
        intra_method=intra_method,
        unig_network_path=unig_network_path,
        unig_tf_activity_path=unig_tf_activity_path,
        min_regulon_targets=min_regulon_targets,
        background_number=background_number,
        threads=threads,
        scope=scope,
        min_exp=min_exp,
        cutoff=cutoff,
    )


def lr_cell_level_score(adata, lr_pair, lr_weight_key="LR_cell_weight"):
    """Return a spot-by-spot communication matrix for one LR pair."""
    matrix = adata.uns[lr_weight_key][lr_pair]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, index=adata.obs_names, columns=adata.obs_names)


def lr_celltype_weight(adata, lr_pair, weight_key="LR_celltype_weight"):
    """Return sender-by-receiver cell-type communication for one LR pair."""
    type_list = [str(x) for x in adata.uns["type_list"]]
    return pd.DataFrame(adata.uns[weight_key][lr_pair], index=type_list, columns=type_list)


def lr_celltype_mean_weight(adata, lr_pair, weight_key="LR_celltype_mean_weight"):
    """Return sender-by-receiver mean communication for one LR pair."""
    type_list = [str(x) for x in adata.uns["type_list"]]
    return pd.DataFrame(adata.uns[weight_key][lr_pair], index=type_list, columns=type_list)


def celltype_communication_matrix(adata, weight_key="LR_celltype_weight"):
    """Aggregate all LR cell-type matrices into one sender-by-receiver matrix."""
    type_list = [str(x) for x in adata.uns["type_list"]]
    total_mat = np.zeros((len(type_list), len(type_list)), dtype=float)
    for matrix in adata.uns[weight_key].values():
        total_mat += np.asarray(matrix, dtype=float)
    return pd.DataFrame(total_mat, index=type_list, columns=type_list)


def plot_celltype_communication_heatmap(
    comm_mtx,
    out_pdf=None,
    title="Cell type communication",
    cmap="coolwarm",
    figsize=(6.5, 5.8),
):
    """Plot a log1p cell-type communication heatmap."""
    import matplotlib.pyplot as plt

    plot_mtx = np.log1p(comm_mtx)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_mtx.values, cmap=cmap, aspect="equal")

    ax.set_xticks(np.arange(comm_mtx.shape[1]))
    ax.set_yticks(np.arange(comm_mtx.shape[0]))
    ax.set_xticklabels(comm_mtx.columns)
    ax.set_yticklabels(comm_mtx.index)
    ax.set_xlabel("Receiver cluster")
    ax.set_ylabel("Sender cluster")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log1p(total communication abundance)")
    fig.tight_layout()

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_chord_diagram_nature(
    comm_mtx,
    out_pdf=None,
    out_svg=None,
    title="Cell type communication",
    min_weight=None,
    top_n_edges=None,
    remove_self=True,
    sector_size="strength",
    gap_deg=3.0,
    ribbon_alpha=0.55,
    figsize=(6.2, 6.2),
    label_size=8.5,
    title_size=11,
    ring_width=0.055,
    font_family="Arial",
    color_adata=None,
    color_key="type",
    fallback_cmap="tab20",
):
    """Plot a chord diagram for aggregated cell-type communication."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    plt.rcParams["font.family"] = font_family
    mat = comm_mtx.copy().astype(float)
    labels = [str(x) for x in mat.index]
    if remove_self:
        shared = mat.index.intersection(mat.columns)
        for node in shared:
            mat.loc[node, node] = 0.0

    edges = _prepare_comm_edges(mat, min_weight=min_weight, top_n_edges=top_n_edges, remove_self=False)
    if not edges:
        raise ValueError("No positive communication edges remain after filtering.")

    edge_df = pd.DataFrame(edges, columns=["sender", "receiver", "weight"])
    displayed = pd.DataFrame(0.0, index=labels, columns=labels)
    for sender, receiver, weight in edges:
        if sender in displayed.index and receiver in displayed.columns:
            displayed.loc[sender, receiver] = weight

    out_strength = displayed.sum(axis=1)
    in_strength = displayed.sum(axis=0)
    total_strength = out_strength + in_strength
    if sector_size == "equal":
        sector_weights = pd.Series(1.0, index=labels)
    elif sector_size == "outgoing":
        sector_weights = out_strength
    elif sector_size == "incoming":
        sector_weights = in_strength
    else:
        sector_weights = total_strength
    fallback = max(float(total_strength.max()) * 0.02, 1e-12)
    sector_weights = sector_weights.replace(0, np.nan).fillna(fallback)

    n_labels = len(labels)
    available = 360.0 - gap_deg * n_labels
    sector_angles = sector_weights / sector_weights.sum() * available

    sectors = {}
    current = 90.0
    for label, width in zip(labels, sector_angles):
        start = current
        end = current - width
        sectors[label] = (start, end)
        current = end - gap_deg

    color_map = _scanpy_category_color_map(color_adata, color_key, labels, fallback_cmap=fallback_cmap)
    source_seg, target_seg = _allocate_chord_segments(labels, sectors, edge_df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    for _, row in edge_df.sort_values("weight", ascending=True).iterrows():
        sender = row["sender"]
        receiver = row["receiver"]
        s0, s1 = source_seg[(sender, receiver)]
        t0, t1 = target_seg[(sender, receiver)]
        ax.add_patch(
            _make_ribbon(
                s0,
                s1,
                t0,
                t1,
                r=0.90,
                facecolor=color_map[sender],
                alpha=ribbon_alpha,
            )
        )

    outer_r = 1.00
    for label in labels:
        start, end = sectors[label]
        ax.add_patch(
            Wedge(
                (0, 0),
                outer_r,
                end,
                start,
                width=ring_width,
                facecolor=color_map[label],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
        )
        mid = (start + end) / 2
        x_pos, y_pos = _pol2cart(mid, 1.10)
        rot, ha = _text_rotation(mid)
        ax.text(
            x_pos,
            y_pos,
            label,
            ha=ha,
            va="center",
            rotation=rot,
            rotation_mode="anchor",
            fontsize=label_size,
            color="#222222",
        )

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    if title:
        ax.set_title(title, fontsize=title_size, pad=12)

    fig.tight_layout()
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.show()
    return fig, ax


def receiver_lr_spot_matrix(adata, lr_weight_key="LR_cell_weight", dtype=np.float32):
    """Build an LR-pair-by-receiver-spot matrix."""
    return receiver_feature_spot_matrix(adata, lr_weight_key, dtype=dtype)


def receiver_pathway_spot_matrix(
    adata,
    pathway_weight_key="LR_pathway_cell_weight",
    dtype=np.float32,
):
    """Build a pathway-by-receiver-spot matrix."""
    return receiver_feature_spot_matrix(adata, pathway_weight_key, dtype=dtype)


def receiver_feature_spot_matrix(adata, weight_key, dtype=np.float32):
    """Build a feature-by-receiver-spot matrix from spot-by-spot weights."""
    feature_names = list(adata.uns[weight_key].keys())
    spot_names = adata.obs_names.astype(str).to_list()
    receiver_rows = np.zeros((len(feature_names), adata.n_obs), dtype=dtype)

    for i, feature in enumerate(feature_names):
        matrix = adata.uns[weight_key][feature]
        if sparse.issparse(matrix):
            receiver_total = np.asarray(matrix.sum(axis=0)).ravel()
        else:
            receiver_total = np.asarray(matrix, dtype=dtype).sum(axis=0)
        receiver_rows[i, :] = receiver_total.astype(dtype, copy=False)

    return pd.DataFrame(receiver_rows, index=feature_names, columns=spot_names)


def make_receiver_lr_adata(lr_spot_mtx, adata, cluster_key="type"):
    """Represent receiver LR abundance as spots by LR pairs."""
    return make_receiver_feature_adata(
        lr_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="LR_pair",
        label_func=format_lr_name,
        label_col="LR_label",
    )


def make_receiver_pathway_adata(pathway_spot_mtx, adata, cluster_key="type"):
    """Represent receiver pathway abundance as spots by pathways."""
    return make_receiver_feature_adata(
        pathway_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="pathway",
    )


def make_receiver_feature_adata(
    feature_spot_mtx,
    adata,
    cluster_key="type",
    feature_col="feature",
    label_func=None,
    label_col=None,
):
    """Represent receiver feature abundance as an AnnData matrix."""
    receiver_adata = sc.AnnData(feature_spot_mtx.T.astype(np.float32))
    receiver_adata.obs_names = feature_spot_mtx.columns.astype(str)
    receiver_adata.var_names = feature_spot_mtx.index.astype(str)
    receiver_adata.obs[cluster_key] = adata.obs.loc[receiver_adata.obs_names, cluster_key].astype("category")
    receiver_adata.var[feature_col] = receiver_adata.var_names
    if label_func is not None and label_col is not None:
        receiver_adata.var[label_col] = [label_func(x) for x in receiver_adata.var_names]
    return receiver_adata


def summarize_receiver_by_cluster(feature_spot_mtx, adata, cluster_key="type"):
    """Average receiver feature abundance for each cluster."""
    cluster_labels = adata.obs[cluster_key]
    if hasattr(cluster_labels, "cat"):
        cluster_order = [str(x) for x in cluster_labels.cat.categories]
    else:
        cluster_order = [str(x) for x in pd.Index(cluster_labels.astype(str)).drop_duplicates()]

    cluster_labels = cluster_labels.astype(str)
    mean_rows = []
    pct_rows = []
    for cluster in cluster_order:
        spots = adata.obs_names[cluster_labels.to_numpy() == cluster].astype(str).to_list()
        if not spots:
            continue
        sub = feature_spot_mtx.loc[:, spots]
        mean_rows.append(sub.mean(axis=1).rename(cluster))
        pct_rows.append((sub > 0).mean(axis=1).rename(cluster))

    return pd.concat(mean_rows, axis=1), pd.concat(pct_rows, axis=1)


def summarize_lr_receiver_by_cluster(lr_spot_mtx, adata, cluster_key="type"):
    """Average LR receiver abundance for each cluster."""
    return summarize_receiver_by_cluster(lr_spot_mtx, adata, cluster_key=cluster_key)


def summarize_pathway_receiver_by_cluster(pathway_spot_mtx, adata, cluster_key="type"):
    """Average pathway receiver abundance for each cluster."""
    return summarize_receiver_by_cluster(pathway_spot_mtx, adata, cluster_key=cluster_key)


def rank_cluster_specific_lr_pairs_scanpy(
    lr_spot_mtx,
    adata,
    cluster_key="type",
    top_n=20,
    method="wilcoxon",
    min_pct=0.05,
    max_pval_adj=0.05,
    min_logfoldchange=0.0,
    fill_top_n_with_unfiltered=False,
):
    """Rank receiver LR markers with Scanpy group-vs-rest marker logic."""
    return rank_cluster_specific_features_scanpy(
        lr_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="LR_pair",
        top_n=top_n,
        method=method,
        min_pct=min_pct,
        max_pval_adj=max_pval_adj,
        min_logfoldchange=min_logfoldchange,
        fill_top_n_with_unfiltered=fill_top_n_with_unfiltered,
        label_func=format_lr_name,
        label_col="LR_label",
    )


def rank_cluster_specific_pathways_scanpy(
    pathway_spot_mtx,
    adata,
    cluster_key="type",
    top_n=20,
    method="wilcoxon",
    min_pct=0.05,
    max_pval_adj=0.05,
    min_logfoldchange=0.0,
    fill_top_n_with_unfiltered=False,
):
    """Rank receiver pathway markers with Scanpy group-vs-rest marker logic."""
    return rank_cluster_specific_features_scanpy(
        pathway_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="pathway",
        top_n=top_n,
        method=method,
        min_pct=min_pct,
        max_pval_adj=max_pval_adj,
        min_logfoldchange=min_logfoldchange,
        fill_top_n_with_unfiltered=fill_top_n_with_unfiltered,
    )


def rank_cluster_specific_features_scanpy(
    feature_spot_mtx,
    adata,
    cluster_key="type",
    feature_col="feature",
    top_n=20,
    method="wilcoxon",
    min_pct=0.05,
    max_pval_adj=0.05,
    min_logfoldchange=0.0,
    fill_top_n_with_unfiltered=False,
    label_func=None,
    label_col=None,
):
    """Rank receiver communication markers with Scanpy group-vs-rest logic."""
    receiver_adata = make_receiver_feature_adata(
        feature_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col=feature_col,
        label_func=label_func,
        label_col=label_col,
    )
    sc.pp.log1p(receiver_adata)
    sc.tl.rank_genes_groups(
        receiver_adata,
        groupby=cluster_key,
        reference="rest",
        method=method,
        pts=True,
        use_raw=False,
        n_genes=receiver_adata.n_vars,
    )

    marker_df = sc.get.rank_genes_groups_df(receiver_adata, group=None)
    marker_df = marker_df.rename(
        columns={
            "group": "cluster",
            "names": feature_col,
            "pct_nz_group": "pct_receiver_spots_positive",
            "pct_nz_reference": "pct_receiver_spots_positive_rest",
        }
    )
    marker_df["cluster"] = marker_df["cluster"].astype(str)
    marker_df[feature_col] = marker_df[feature_col].astype(str)
    if label_func is not None and label_col is not None:
        marker_df[label_col] = marker_df[feature_col].map(label_func)

    raw_mean, raw_pct = summarize_receiver_by_cluster(
        feature_spot_mtx,
        adata,
        cluster_key=cluster_key,
    )
    marker_df = _merge_receiver_mean_and_pct(marker_df, raw_mean, raw_pct, feature_col)
    marker_df = marker_df.sort_values(
        ["cluster", "scores", "logfoldchanges", "mean_receiver_abundance"],
        ascending=[True, False, False, False],
    )

    keep = marker_df["pct_receiver_spots_positive"].fillna(1.0) >= min_pct
    if np.isfinite(max_pval_adj):
        keep &= marker_df["pvals_adj"].fillna(0.0) <= max_pval_adj
    if np.isfinite(min_logfoldchange):
        keep &= marker_df["logfoldchanges"].fillna(0.0) >= min_logfoldchange

    if fill_top_n_with_unfiltered:
        selected = []
        for _cluster, cluster_df in marker_df.groupby("cluster", sort=False):
            filtered = cluster_df.loc[keep.loc[cluster_df.index]].head(top_n)
            if len(filtered) < top_n:
                filler = cluster_df.loc[
                    ~cluster_df[feature_col].isin(filtered[feature_col])
                ].head(top_n - len(filtered))
                filtered = pd.concat([filtered, filler], axis=0)
            selected.append(filtered)
        marker_df = pd.concat(selected, ignore_index=True)
    else:
        marker_df = (
            marker_df.loc[keep]
            .groupby("cluster", group_keys=False, sort=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    return marker_df, receiver_adata, raw_mean, raw_pct


def format_lr_name(lr_name):
    """Format an STCase LR name for display."""
    return str(lr_name).replace("COMPLEX:", "").replace("|", "->")


def plot_cluster_specific_lr_marker_heatmap(
    mean_by_cluster,
    marker_table,
    adata,
    cluster_key="type",
    out_pdf=None,
    out_svg=None,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
):
    """Plot cluster-specific receiver LR markers."""
    selected_table = marker_table.drop_duplicates(["cluster", "LR_pair"]).copy()
    selected = selected_table["LR_pair"].drop_duplicates().to_list()
    if not selected:
        raise ValueError("No cluster-specific LR pairs passed the current filters.")

    heatmap_raw = np.log1p(mean_by_cluster.loc[selected])
    heatmap_z = _row_zscore(heatmap_raw).clip(vmin, vmax)
    label_map = selected_table.drop_duplicates("LR_pair").set_index("LR_pair")["LR_label"].to_dict()
    row_labels = [label_map.get(x, format_lr_name(x)) for x in heatmap_z.index]
    row_group_map = selected_table.drop_duplicates("LR_pair").set_index("LR_pair")["cluster"].to_dict()
    row_groups = [row_group_map.get(x, "") for x in heatmap_z.index]

    return _draw_marker_heatmap(
        heatmap_z,
        row_labels=row_labels,
        row_groups=row_groups,
        adata=adata,
        cluster_key=cluster_key,
        title="Cluster-specific receiver L-R communication markers",
        ylabel="Receiver L-R pair",
        colorbar_label="Row z-score of log1p(mean receiver abundance)",
        out_pdf=out_pdf,
        out_svg=out_svg,
        fig_width=7.0,
        row_height=0.125,
        min_fig_height=4.2,
        max_fig_height=11.0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        row_label_size=5.2,
    )


def plot_cluster_specific_pathway_marker_heatmap(
    mean_by_cluster,
    marker_table,
    adata,
    cluster_key="type",
    out_pdf=None,
    out_svg=None,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
):
    """Plot cluster-specific receiver pathway markers."""
    selected_table = marker_table.drop_duplicates(["cluster", "pathway"]).copy()
    selected = selected_table["pathway"].drop_duplicates().to_list()
    if not selected:
        raise ValueError("No cluster-specific pathways passed the current filters.")

    heatmap_raw = np.log1p(mean_by_cluster.loc[selected])
    heatmap_z = _row_zscore(heatmap_raw).clip(vmin, vmax)
    row_group_map = selected_table.drop_duplicates("pathway").set_index("pathway")["cluster"]
    row_groups = row_group_map.reindex(heatmap_z.index).fillna("").to_list()

    return _draw_marker_heatmap(
        heatmap_z,
        row_labels=heatmap_z.index.to_list(),
        row_groups=row_groups,
        adata=adata,
        cluster_key=cluster_key,
        title="Cluster-specific receiver pathway communication markers",
        ylabel="Receiver pathway",
        colorbar_label="Row z-score of log1p(mean receiver abundance)",
        out_pdf=out_pdf,
        out_svg=out_svg,
        fig_width=6.8,
        row_height=0.28,
        min_fig_height=3.6,
        max_fig_height=9.0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        row_label_size=7.2,
    )


def extract_cluster_markers(
    marker_df,
    cluster_value="1",
    feature_col="LR_pair",
    label_col=None,
    top_n=20,
):
    """Extract the top receiver communication markers for one cluster."""
    df = marker_df.copy()
    df["cluster"] = df["cluster"].astype(str)
    out = df.loc[df["cluster"] == str(cluster_value)].copy()
    if out.empty:
        available = sorted(df["cluster"].astype(str).unique())
        raise ValueError(f"No markers found for cluster {cluster_value!r}. Available clusters: {available}")

    sort_cols = [c for c in ["scores", "logfoldchanges", "mean_receiver_abundance"] if c in out.columns]
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(top_n).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    if label_col is None:
        label_col = feature_col
    if "label" not in out.columns:
        out["label"] = out[label_col].astype(str)
    if "communication_abundance" not in out.columns and "mean_receiver_abundance" in out.columns:
        out["communication_abundance"] = out["mean_receiver_abundance"]
    return out


def plot_cluster_marker_score_bars(
    lr_markers,
    pathway_markers,
    cluster_value="1",
    top_n=20,
    out_pdf=None,
    cmap="YlGnBu",
):
    """Plot LR and pathway receiver marker scores for one cluster."""
    import matplotlib.pyplot as plt

    lr_logfc = lr_markers["logfoldchanges"].astype(float).to_numpy()
    pathway_logfc = pathway_markers["logfoldchanges"].astype(float).to_numpy()
    all_logfc = np.concatenate([lr_logfc[np.isfinite(lr_logfc)], pathway_logfc[np.isfinite(pathway_logfc)]])
    if len(all_logfc) > 0 and np.nanmax(all_logfc) > np.nanmin(all_logfc):
        norm = plt.Normalize(vmin=np.nanmin(all_logfc), vmax=np.nanmax(all_logfc))
    else:
        norm = plt.Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4), constrained_layout=True)
    _plot_marker_score_bar(
        lr_markers,
        label_col="label",
        title=f"Cluster {cluster_value} receiver top {top_n} L-R markers",
        ax=axes[0],
        top_n=top_n,
        norm=norm,
        cmap=cmap,
    )
    _plot_marker_score_bar(
        pathway_markers,
        label_col="label",
        title=f"Cluster {cluster_value} receiver top {top_n} pathway markers",
        ax=axes[1],
        top_n=top_n,
        norm=norm,
        cmap=cmap,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.82, pad=0.015)
    cbar.set_label("logfoldchanges", rotation=90)

    if out_pdf is not None:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.show()
    return fig, axes


def add_lr_receiver_total(adata, lr_pair, score_key=None, lr_weight_key="LR_cell_weight"):
    """Add receiver total score for one LR pair to adata.obs."""
    if score_key is None:
        score_key = f"{_safe_obs_key(lr_pair)}_receiver_total"
    return add_receiver_total_score(adata, lr_pair, score_key, lr_weight_key)


def add_pathway_receiver_total(
    adata,
    pathway,
    score_key=None,
    pathway_weight_key="LR_pathway_cell_weight",
):
    """Add receiver total score for one pathway to adata.obs."""
    if score_key is None:
        score_key = f"{_safe_obs_key(pathway)}_pathway_receiver_total"
    return add_receiver_total_score(adata, pathway, score_key, pathway_weight_key)


def add_receiver_total_score(adata, feature, score_key, weight_key):
    """Add the receiver total score for one communication feature to adata.obs."""
    if feature not in adata.uns[weight_key]:
        candidates = [key for key in adata.uns[weight_key].keys() if str(feature) in str(key)]
        raise KeyError(f"{feature!r} not found in {weight_key}. Candidates: {candidates[:10]}")

    matrix = adata.uns[weight_key][feature]
    if sparse.issparse(matrix):
        receiver_total = np.asarray(matrix.sum(axis=0)).ravel()
    else:
        receiver_total = np.asarray(matrix).sum(axis=0)
    adata.obs[score_key] = receiver_total
    return score_key


def plot_spatial_receiver_score(
    adata,
    score_key,
    basis="spatial",
    size=75,
    title=None,
    cmap="coolwarm",
    save=None,
):
    """Plot a receiver communication score on spatial coordinates."""
    return sc.pl.embedding(
        adata,
        basis=basis,
        color=[score_key],
        size=size,
        title=title or score_key,
        cmap=cmap,
        save=save,
    )


def _merge_receiver_mean_and_pct(marker_df, raw_mean, raw_pct, feature_col):
    long_mean = (
        raw_mean.stack()
        .rename("mean_receiver_abundance")
        .reset_index()
        .rename(columns={"level_0": feature_col, "level_1": "cluster"})
    )
    long_mean[feature_col] = long_mean[feature_col].astype(str)
    long_mean["cluster"] = long_mean["cluster"].astype(str)
    marker_df = marker_df.merge(long_mean, on=[feature_col, "cluster"], how="left")

    if "pvals_adj" not in marker_df.columns:
        marker_df["pvals_adj"] = np.nan
    if "logfoldchanges" not in marker_df.columns:
        marker_df["logfoldchanges"] = np.nan
    if "pct_receiver_spots_positive" not in marker_df.columns:
        long_pct = (
            raw_pct.stack()
            .rename("pct_receiver_spots_positive")
            .reset_index()
            .rename(columns={"level_0": feature_col, "level_1": "cluster"})
        )
        long_pct[feature_col] = long_pct[feature_col].astype(str)
        long_pct["cluster"] = long_pct["cluster"].astype(str)
        marker_df = marker_df.merge(long_pct, on=[feature_col, "cluster"], how="left")
    return marker_df


def _prepare_comm_edges(comm_mtx, min_weight=None, top_n_edges=None, remove_self=True):
    mat = comm_mtx.copy().astype(float)
    if remove_self:
        shared = mat.index.intersection(mat.columns)
        for node in shared:
            mat.loc[node, node] = 0.0

    edges = []
    for sender in mat.index:
        for receiver in mat.columns:
            weight = float(mat.loc[sender, receiver])
            if np.isfinite(weight) and weight > 0:
                edges.append((str(sender), str(receiver), weight))

    if min_weight is not None:
        edges = [edge for edge in edges if edge[2] >= min_weight]
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    if top_n_edges is not None:
        edges = edges[: int(top_n_edges)]
    return edges


def _row_zscore(df):
    values = df.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    z = (values - mean) / np.where(std == 0, 1.0, std)
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _draw_marker_heatmap(
    heatmap_z,
    row_labels,
    row_groups,
    adata,
    cluster_key,
    title,
    ylabel,
    colorbar_label,
    out_pdf=None,
    out_svg=None,
    fig_width=7.0,
    row_height=0.135,
    min_fig_height=4.0,
    max_fig_height=11.0,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
    row_label_size=5.6,
    col_label_size=8.0,
):
    import matplotlib.pyplot as plt

    n_rows, n_cols = heatmap_z.shape
    fig_height = min(max_fig_height, max(min_fig_height, 1.15 + n_rows * row_height))
    left = 0.24 if n_rows > 45 else 0.22
    right = 0.89
    bottom = 0.075
    top = 0.93
    strip_h = 0.018
    strip_gap = 0.010

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([left, bottom, right - left, top - bottom - strip_h - strip_gap])
    ax_colors = fig.add_axes([left, top - strip_h, right - left, strip_h])
    cax = fig.add_axes([0.925, bottom, 0.030, top - bottom - strip_h - strip_gap])

    cluster_order = heatmap_z.columns.to_list()
    color_map = _scanpy_category_color_map(adata, cluster_key, cluster_order, fallback_cmap="tab20")
    rgb = np.array([plt.matplotlib.colors.to_rgb(color_map[c]) for c in cluster_order])[np.newaxis, :, :]
    ax_colors.imshow(rgb, aspect="auto", interpolation="nearest")
    ax_colors.set_xlim(-0.5, n_cols - 0.5)
    ax_colors.set_xticks(np.arange(n_cols))
    ax_colors.set_xticklabels(cluster_order, rotation=45, ha="right", fontsize=col_label_size)
    ax_colors.set_yticks([])
    ax_colors.tick_params(axis="x", length=0, pad=1)
    for spine in ax_colors.spines.values():
        spine.set_visible(False)

    im = ax.imshow(
        heatmap_z.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cluster_order, rotation=45, ha="right", fontsize=col_label_size)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=row_label_size)
    ax.set_xlabel("Receiver cell type", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(length=0, pad=1)

    group_counts = pd.Series(row_groups).value_counts(sort=False).to_numpy()
    for boundary in np.cumsum(group_counts)[:-1]:
        ax.axhline(boundary - 0.5, color="white", lw=1.0)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, rotation=90, fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=2)

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.show()
    return fig, ax


def _plot_marker_score_bar(df, label_col, title, ax, top_n=20, norm=None, cmap="YlGnBu"):
    import matplotlib.pyplot as plt

    plot_df = df.head(top_n).copy()
    values = plot_df["scores"].astype(float).to_numpy()
    logfc = plot_df.get("logfoldchanges", pd.Series(np.nan, index=plot_df.index)).astype(float).to_numpy()

    cmap_obj = plt.get_cmap(cmap)
    if norm is None:
        if np.isfinite(logfc).any() and np.nanmax(logfc) > np.nanmin(logfc):
            norm = plt.Normalize(vmin=np.nanmin(logfc), vmax=np.nanmax(logfc))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
    colors = cmap_obj(norm(logfc)) if np.isfinite(logfc).any() else cmap_obj(np.linspace(0.25, 0.85, len(plot_df)))

    x_pos = np.arange(len(plot_df))
    ax.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.5, width=0.78)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df[label_col].astype(str), rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Scanpy marker score")
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    return ax


def _scanpy_category_color_map(adata, color_key, labels, fallback_cmap="tab20"):
    import matplotlib.pyplot as plt

    labels = [str(label) for label in labels]
    fallback_palette = plt.get_cmap(fallback_cmap)(np.linspace(0, 1, max(len(labels), 3)))
    fallback = {label: fallback_palette[i] for i, label in enumerate(labels)}

    if adata is None or color_key not in getattr(adata, "obs", {}):
        return fallback

    color_key_uns = f"{color_key}_colors"
    if color_key_uns not in adata.uns:
        return fallback

    values = adata.obs[color_key]
    if hasattr(values, "cat"):
        categories = [str(x) for x in values.cat.categories]
    else:
        categories = [str(x) for x in pd.Categorical(values).categories]

    colors = list(adata.uns[color_key_uns])
    scanpy_map = {category: colors[i] for i, category in enumerate(categories) if i < len(colors)}
    return {label: scanpy_map.get(label, fallback[label]) for label in labels}


def _allocate_chord_segments(labels, sectors, edge_df):
    source_seg = {}
    target_seg = {}
    for label in labels:
        start, end = sectors[label]
        width = start - end

        outgoing = edge_df.loc[edge_df["sender"] == label]
        out_total = outgoing["weight"].sum()
        cursor = start
        for _, row in outgoing.iterrows():
            span = width * row["weight"] / out_total if out_total > 0 else 0
            source_seg[(row["sender"], row["receiver"])] = (cursor, cursor - span)
            cursor -= span

        incoming = edge_df.loc[edge_df["receiver"] == label]
        in_total = incoming["weight"].sum()
        cursor = start
        for _, row in incoming.iterrows():
            span = width * row["weight"] / in_total if in_total > 0 else 0
            target_seg[(row["sender"], row["receiver"])] = (cursor, cursor - span)
            cursor -= span
    return source_seg, target_seg


def _make_ribbon(theta1a, theta1b, theta2a, theta2b, r=0.92, alpha=0.55, facecolor="gray"):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    p1 = _pol2cart(theta1a, r)
    p2 = _pol2cart(theta1b, r)
    p3 = _pol2cart(theta2a, r)
    p4 = _pol2cart(theta2b, r)
    center = np.array([0.0, 0.0])
    verts = [p1, center, center, p4, p3, center, center, p2, p1]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(
        MplPath(verts, codes),
        facecolor=facecolor,
        edgecolor="none",
        lw=0.0,
        alpha=alpha,
        zorder=1,
    )


def _pol2cart(theta_deg, r):
    theta = np.deg2rad(theta_deg)
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def _text_rotation(theta_deg):
    if 90 < theta_deg < 270:
        return theta_deg + 180, "right"
    return theta_deg, "left"


def _safe_obs_key(value):
    return (
        str(value)
        .replace("COMPLEX:", "")
        .replace("|", "_")
        .replace("->", "_")
        .replace(":", "_")
        .replace("-", "_")
    )
