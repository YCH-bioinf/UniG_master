"""plotting functions"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.api.types import (
    is_string_dtype,
    is_categorical_dtype,
)
import json
import warnings
# import plotly.express as px
# import plotly.graph_objects as go

from .palettes import (
    default_20,
    default_28,
    default_102
)

from .settings import settings



def generate_palette(arr):
    """Generate a color palette for a given array
    """

    if not isinstance(arr, (pd.Series, np.ndarray)):
        raise TypeError("`arr` must be pd.Series or np.ndarray")
    colors = []
    if is_string_dtype(arr) or is_categorical_dtype(arr):
        categories = np.unique(arr)
        length = len(categories)
        # check if default matplotlib palette has enough colors
        # mpl.style.use('default')
        if len(mpl.rcParams['axes.prop_cycle'].by_key()['color']) >= length:
            cc = mpl.rcParams['axes.prop_cycle']()
            palette = [mpl.colors.rgb2hex(next(cc)['color'])
                       for _ in range(length)]
        else:
            if length <= 20:
                palette = default_20
            elif length <= 28:
                palette = default_28
            elif length <= len(default_102):  # 103 colors
                palette = default_102
            else:
                rgb_rainbow = mpl.cm.rainbow(np.linspace(0, 1, length))
                palette = [mpl.colors.rgb2hex(rgb_rainbow[i, :-1])
                           for i in range(length)]
        colors = pd.Series(['']*len(arr))
        for i, x in enumerate(categories):
            ids = np.where(arr == x)[0]
            colors[ids] = palette[i]
        colors = list(colors)
    else:
        raise TypeError("unsupported data type for `arr`")
    dict_palette = dict(zip(arr, colors))
    return dict_palette


def discretize(adata,
               kde=None,
               fig_size=(6, 6),
               pad=1.08,
               w_pad=None,
               h_pad=None,
               save_fig=None,
               fig_path=None,
               fig_name='plot_discretize.pdf',
               **kwargs):
    """Plot original data VS discretized data

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    kde : `bool`, optional (default: None)
        If True, compute a kernel density estimate to smooth the distribution
        and show on the plot. Invalid as of v0.2.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5,8))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_discretize.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``plt.hist()``

    Returns
    -------
    None
    """
    if kde is not None:
        warnings.warn("kde is not supported as of v0.2", DeprecationWarning)
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    assert 'disc' in adata.uns_keys(), \
        "please run `si.tl.discretize()` first"
    if kde is not None:
        warnings.warn("kde is no longer supported as of v1.1",
                      DeprecationWarning)

    hist_edges = adata.uns['disc']['hist_edges']
    hist_count = adata.uns['disc']['hist_count']
    bin_edges = adata.uns['disc']['bin_edges']
    bin_count = adata.uns['disc']['bin_count']

    fig, ax = plt.subplots(2, 1, figsize=fig_size)
    _ = ax[0].hist(hist_edges[:-1],
                   hist_edges,
                   weights=hist_count,
                   linewidth=0,
                   **kwargs)
    _ = ax[1].hist(bin_edges[:-1],
                   bin_edges,
                   weights=bin_count,
                   **kwargs)
    ax[0].set_xlabel('Non-zero values')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Original')
    ax[1].set_xlabel('Non-zero values')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Discretized')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)



def pbg_metrics(metrics=['mrr'],
                path_emb=None,
                fig_size=(5, 3),
                fig_ncol=1,
                save_fig=None,
                fig_path=None,
                fig_name='pbg_metrics.pdf',
                pad=1.08,
                w_pad=None,
                h_pad=None,
                **kwargs):
    """Plot PBG training metrics

    Parameters
    ----------
    metrics: `list`, optional (default: ['mrr])
        Evalulation metrics for PBG training. Possible metrics:

        - 'pos_rank' : the average of the ranks of all positives
          (lower is better, best is 1).
        - 'mrr' : the average of the reciprocal of the ranks of all positives
          (higher is better, best is 1).
        - 'r1' : the fraction of positives that rank better than
           all their negatives, i.e., have a rank of 1
           (higher is better, best is 1).
        - 'r10' : the fraction of positives that rank in the top 10
           among their negatives
           (higher is better, best is 1).
        - 'r50' : the fraction of positives that rank in the top 50
           among their negatives
           (higher is better, best is 1).
        - 'auc' : Area Under the Curve (AUC)
    path_emb: `str`, optional (default: None)
        Path to directory for pbg embedding model.
        If None, .settings.pbg_params['checkpoint_path'] will be used.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5, 3))
        figure size.
    fig_ncol: `int`, optional (default: 1)
        the number of columns of the figure panel
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_umap.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    assert isinstance(metrics, list), "`metrics` must be list"
    for x in metrics:
        if x not in ['pos_rank', 'mrr', 'r1',
                     'r10', 'r50', 'auc']:
            raise ValueError(f'unrecognized metric {x}')
    pbg_params = settings.pbg_params
    if path_emb is None:
        path_emb = pbg_params['checkpoint_path']
    training_loss = []
    eval_stats_before = dict()
    with open(os.path.join(path_emb, 'training_stats.json'), 'r') as f:
        for line in f:
            line_json = json.loads(line)
            if 'stats' in line_json.keys():
                training_loss.append(line_json['stats']['metrics']['loss'])
                line_stats_before = line_json['eval_stats_before']['metrics']
                for x in line_stats_before.keys():
                    if x not in eval_stats_before.keys():
                        eval_stats_before[x] = [line_stats_before[x]]
                    else:
                        eval_stats_before[x].append(line_stats_before[x])
    n_epochs = len(training_loss)
    if n_epochs == 0:
        raise ValueError(f"No training stats found in {path_emb!r}.")
    if 'loss' not in eval_stats_before:
        raise ValueError(f"No validation loss found in {path_emb!r}/training_stats.json.")

    metric_lengths = [len(training_loss)]
    metric_lengths.extend(
        len(eval_stats_before[x])
        for x in ['loss', *metrics]
        if x in eval_stats_before
    )
    n_epochs = min(metric_lengths)

    df_metrics = pd.DataFrame(index=range(n_epochs))
    df_metrics['epoch'] = range(n_epochs)
    training_loss = training_loss[:n_epochs]
    df_metrics['training_loss'] = training_loss
    df_metrics['validation_loss'] = eval_stats_before['loss'][:n_epochs]
    for x in metrics:
        if x in eval_stats_before:
            df_metrics[x] = eval_stats_before[x][:n_epochs]

    fig_nrow = int(np.ceil((df_metrics.shape[1]-1)/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                              fig_size[1]*fig_nrow))
    dict_palette = generate_palette(df_metrics.columns[1:].values)
    for i, metric in enumerate(df_metrics.columns[1:]):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        ax_i.scatter(df_metrics['epoch'],
                     df_metrics[metric],
                     c=dict_palette[metric],
                     **kwargs)
        ax_i.set_title(metric)
        ax_i.set_xlabel('epoch')
        ax_i.set_ylabel(metric)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


from scipy.sparse import find
def node_similarity(adata,
                    bins=20,
                    log=True,
                    show_cutoff=True,
                    cutoff=None,
                    n_edges=5000,
                    fig_size=(5, 3),
                    pad=1.08,
                    w_pad=None,
                    h_pad=None,
                    save_fig=None,
                    fig_path=None,
                    fig_name='plot_node_similarity.pdf',
                    ):
    """Plot similarity scores of nodes

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    bins : `int`, optional (default: 20)
        The number of equal-width bins in the given range for histogram plot.
    log : `bool`, optional (default: True)
        If True, log scale will be used for y axis.
    show_cutoff : `bool`, optional (default: True)
        If True, cutoff on scores will be shown
    cutoff: `int`, optional (default: None)
        Cutoff used to select edges
    n_edges: `int`, optional (default: 5000)
        The number of edges to select.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5,8))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_node_similarity.pdf')
        if `save_fig` is True, specify figure name.

    Returns
    -------
    None
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    mat_sim = adata.X

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.hist(mat_sim.data, bins=bins)
    if log:
        ax.set_yscale('log')
    if show_cutoff:
        if cutoff is None:
            if n_edges is None:
                raise ValueError('"cutoff" or "n_edges" has to be specified')
            else:
                cutoff = \
                    np.partition(mat_sim.data,
                                 (mat_sim.size-n_edges))[mat_sim.size-n_edges]
        id_x, id_y, _ = find(mat_sim > cutoff)
        print(f'#selected edges: {len(id_x)}')
        plt.axvline(cutoff, ls='--', c='red')
    ax.set_xlabel('similariy scores')
    ax.set_title('Node similarity')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from scipy import sparse
import seaborn as sns
import os




def plotRegion(chrm, start, end, ax, gtf_file):
    """Plot gene annotations.
    
    Parameters
    ----------
    chrm : str
        Chromosome number.
    start : int
        Start coordinate.
    end : int
        End coordinate.
    ax : axis
        Axis to plot on.
    gtf_file : str
        Path to GTF file.
    """
    
    a = pd.read_csv(gtf_file, sep = "\t", header = None)
    a = a[(a[0] == chrm) & (a[3] <= end) & (a[4] >= start)]

    transcripts = {}
    for i, r in a.iterrows():
        if r[2] == 'transcript':
            gene = r[8].split(";")[0][9:-1]
            if gene[:3] == 'MIR': continue # ignore microRNA
            if r[6] == '+':
                  ax.add_patch(patches.Rectangle((r[3], 0.4), r[4] - r[3] + 1, 0.3, color = 'skyblue'))
            else:
                  ax.add_patch(patches.Rectangle((r[3], -0.7), r[4] - r[3] + 1, 0.3, color = 'lightcoral'))
            if gene not in transcripts:
                transcripts[gene] = (r[3], r[4], r[6])
            else:
                transcripts[gene] = (min(r[3], transcripts[gene][0]), max(r[4], transcripts[gene][1]), r[6])
        elif r[2] == 'exon': 
            gene = r[8].split(";")[0][9:-1] # ignore microRNA
            if gene[:3] == 'MIR': continue
            if r[6] == '+':
                  ax.add_patch(patches.Rectangle((r[3], 0.1), r[4] - r[3] + 1, 0.9, color = 'skyblue'))
            else:
                  ax.add_patch(patches.Rectangle((r[3], -1), r[4] - r[3] + 1, 0.9, color = 'lightcoral'))
        
    for t in transcripts:
        if transcripts[t][2] == '+':
            ax.text(transcripts[t][0] + 10, 1.2, '$\it{'+t+'}$')
        else:
            ax.text(transcripts[t][0] + 10, -1.6, '$\it{'+t+'}$')

    ax.set_xlim((start, end))
    ax.set_ylim((-1.9, 1.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False) # new
    ax.spines['right'].set_visible(False) # new
    ax.spines['left'].set_visible(False) # new
    ax.spines['bottom'].set_visible(False) # new
    
    
def _parse_peak_str(peak_str):
    try:
        if ':' in peak_str and '-' in peak_str:
            chrom, coords = peak_str.split(':')
            start, end = map(int, coords.split('-'))
            return chrom, start, end
    except: pass
    return None, None, None

def get_gene_info(gtf_file, target_gene):
    if not gtf_file: return None, None, None, None
    try:
        df = pd.read_csv(gtf_file, sep="\t", header=None, comment='#', low_memory=False)
        df_gene = df[df[8].str.contains(f'"{target_gene}"', regex=False)]
        if df_gene.empty: return None, None, None, None
        gene_rows = df_gene[df_gene[2] == 'gene']
        if gene_rows.empty: gene_rows = df_gene[df_gene[2] == 'transcript']
        if not gene_rows.empty:
            row = gene_rows.iloc[0]
            chrom, start, end, strand = row[0], int(row[3]), int(row[4]), row[6]
            return chrom, start, end, strand
        return None, None, None, None
    except Exception as e:
        print(f"Error reading GTF: {e}")
        return None, None, None, None
    
def plot_regulatory_links(
    adata_peak,
    adata_rna,
    gene,
    gtf_file,
    regulatory_df=None,
    cell_types=None,
    cell_type_col='label',
    padding=250000,
    figsize=None,
    cmap='tab20',
    plot_region_fn=None, 
    do_zscore=True,
    cutoff=0.5,
    save_path=None
):
    """Plot SCARlink-style ATAC tracks, gene expression, and regulatory links."""
    
    chrom, g_start, g_end, strand = get_gene_info(gtf_file, gene)
    if chrom is None:
        print(f"Gene {gene} not found in GTF.")
        return
    
    tss = g_start  # if strand == '+' else g_end
    x_min = int(tss - padding)
    x_max = int(tss + padding)
    print(f"Plotting {gene} ({chrom}:{x_min}-{x_max})")

    # Prepare gene expression once, independent of the selected cell types.
    expr_series = None
    expr_min, expr_max = -5, 10 
    
    if adata_rna is not None and gene in adata_rna.var_names:
        all_expr = adata_rna[:, gene].X
        if sparse.issparse(all_expr):
            all_expr = np.array(all_expr.todense()).flatten()
        else:
            all_expr = np.array(all_expr).flatten()
            
        if do_zscore:
            mean = np.mean(all_expr)
            std = np.std(all_expr)
            if std == 0:
                gene_expr_vec = all_expr - mean
            else:
                gene_expr_vec = (all_expr - mean) / std
            expr_min, expr_max = -5, 10
        else:
            gene_expr_vec = all_expr
            vmin, vmax = gene_expr_vec.min(), gene_expr_vec.max()
            pad = (vmax - vmin) * 0.1 if vmax != vmin else 1.0
            expr_min, expr_max = vmin - pad, vmax + pad
            
        expr_series = pd.Series(gene_expr_vec, index=adata_rna.obs_names)
        
        if cell_types is None:
            cell_types = sorted(adata_rna.obs[cell_type_col].unique().astype(str))
    else:
        print(f"Warning: {gene} not found in RNA adata.")
        if cell_types is None:
            cell_types = sorted(adata_peak.obs[cell_type_col].unique().astype(str))
        expr_series = None

    # Prepare ATAC signal tracks.
    atac_signals = {}
    global_atac_max = 0
    valid_peak_names = []
    valid_peak_coords = [] 
    
    if adata_peak is not None:
        for p_name in adata_peak.var_names:
            pc, ps, pe = _parse_peak_str(p_name)
            if pc == chrom and pe >= x_min and ps <= x_max:
                valid_peak_names.append(p_name)
                valid_peak_coords.append((ps, pe))

    if valid_peak_names:
        try:
            sub_adata = adata_peak[:, valid_peak_names]
            for ct in cell_types:
                cells = adata_peak.obs[adata_peak.obs[cell_type_col].astype(str) == str(ct)].index
                if len(cells) == 0: continue
                
                sub_X = sub_adata[cells, :].X
                if sparse.issparse(sub_X):
                    means = np.array(sub_X.mean(axis=0)).flatten()
                else:
                    means = np.mean(sub_X, axis=0)
                
                step = 10
                x_vec = np.arange(x_min, x_max, step)
                y_vec = np.zeros_like(x_vec, dtype=float)
                
                for i, val in enumerate(means):
                    if val > 0:
                        ps, pe = valid_peak_coords[i]
                        s_idx = max(0, (ps - x_min) // step)
                        e_idx = min(len(y_vec), (pe - x_min) // step)
                        if s_idx < len(y_vec):
                            y_vec[s_idx:e_idx] = max(y_vec[s_idx:e_idx].max(), val)
                        
                atac_signals[ct] = (x_vec, y_vec)
                if len(y_vec) > 0:
                    global_atac_max = max(global_atac_max, y_vec.max())
        except Exception as e:
            print(f"Error processing peaks: {e}")

    n_tracks = len(cell_types)
    if figsize is None:
        figsize = (14, n_tracks * 1.2 + 2)
        
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_tracks + 1, 2, 
                           width_ratios=[3, 1], 
                           height_ratios=[1] * n_tracks + [0.6], 
                           wspace=0.05, hspace=0.1)
    
    try:
        cmap_obj = plt.get_cmap(cmap)
    except:
        cmap_obj = plt.cm.get_cmap(cmap)

    ylim_atac = global_atac_max * 1.1 if global_atac_max > 0 else 1

    for i, ct in enumerate(cell_types):
        # --- Left: ATAC --- 
        ax_track = fig.add_subplot(gs[i, 0])
        color = cmap_obj(i % 20)
        
        if ct in atac_signals:
            x_vec, y_vec = atac_signals[ct]
            ax_track.plot(x_vec, y_vec, color=color, lw=1, zorder=1)
            ax_track.fill_between(x_vec, 0, y_vec, color=color, alpha=0.4, zorder=1)
        
        # --- Add Links (Cicero style) --- 
        if regulatory_df is not None:
            ct_links = regulatory_df[regulatory_df['celltype'].astype(str) == str(ct)]
            
            if not ct_links.empty:
                gene_links = ct_links[ct_links['gene'] == gene]
                
                gene_links_filtered = gene_links[gene_links['score'] > cutoff]
                
                if not gene_links_filtered.empty:
                    base_rgb = mcolors.to_rgb(color)
                    darker_color = tuple([max(0, c * 0.5) for c in base_rgb])
                    
                    for idx, row in gene_links_filtered.iterrows():
                        peak_str = row['peak']
                        score_col = 'normalized_score' if 'normalized_score' in row else 'score'
                        score = row[score_col]

                        pc, ps, pe = _parse_peak_str(peak_str)
                        if pc == chrom:
                            p_center = (ps + pe) / 2
                            
                            # Square the score to emphasize stronger links.
                            h = (score ** 2) * ylim_atac
                            
                            verts = [
                                (p_center, 0),            
                                ((p_center + tss) / 2, h), 
                                (tss, 0)                  
                            ]
                            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                            
                            path = Path(verts, codes)
                            patch = patches.PathPatch(path, facecolor='none', edgecolor=darker_color, lw=1, alpha=0.8, zorder=2)
                            ax_track.add_patch(patch)

        ax_track.set_xlim(x_min, x_max)
        ax_track.set_ylim(0, ylim_atac)
        ax_track.axis('off')
        ax_track.text(-0.02, 0.5, ct, ha='right', va='center', 
                      transform=ax_track.transAxes, fontsize=12, fontweight='bold')

         # --- Right: RNA Violin ---
        ax_violin = fig.add_subplot(gs[i, 1])
        
        if expr_series is not None:
            try:
                cells = adata_rna.obs[adata_rna.obs[cell_type_col].astype(str) == str(ct)].index
                valid_cells = cells.intersection(expr_series.index)
                if len(valid_cells) > 0:
                    ct_expr = expr_series.loc[valid_cells].values
                    
                    sns.violinplot(x=ct_expr, ax=ax_violin, color=color, orient='h', 
                                   cut=0.5, linewidth=1, density_norm='width', inner="box")
                                   
                    mean_val = np.median(ct_expr)
                    ax_violin.scatter(mean_val, 0, color='white', s=10, zorder=3, edgecolors='black', linewidths=0.5)
            except Exception as e: pass
        
        ax_violin.axis('off')
        ax_violin.set_xlim(expr_min, expr_max)
        
        if i == 0:
            title_suffix = "(Z-score)" if do_zscore else "(Expression)"
            ax_violin.set_title(f"Expression {title_suffix}", fontsize=10)

    # Bottom tracks (Gene & Axis)
    ax_gene = fig.add_subplot(gs[-1, 0])
    if plot_region_fn and gtf_file:
        try:
            plot_region_fn(chrom, x_min, x_max, ax_gene, gtf_file)
        except:
            ax_gene.plot([g_start, g_end], [0, 0], color='black', lw=2)
            ax_gene.text((g_start+g_end)/2, -0.5, gene, ha='center')
            ax_gene.set_xlim(x_min, x_max)
            ax_gene.axis('off')
    else:
        ax_gene.plot([g_start, g_end], [0, 0], color='black', lw=2)
        ax_gene.text((g_start+g_end)/2, -0.5, gene, ha='center')
        ax_gene.set_xlim(x_min, x_max)
        ax_gene.axis('off')

    ax_empty = fig.add_subplot(gs[-1, 1])
    if expr_series is not None:
        ax_empty.set_xlim(expr_min, expr_max)
        ax_empty.spines['top'].set_visible(False)
        ax_empty.spines['right'].set_visible(False)
        ax_empty.spines['left'].set_visible(False)
        ax_empty.get_yaxis().set_visible(False)
    else:
        ax_empty.axis('off')
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")

    return fig



