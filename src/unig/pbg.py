"""PyTorch-BigGraph (PBG) for learning graph embeddings"""

from ._warnings import configure_runtime_warnings

configure_runtime_warnings()

from random import shuffle
from typing import Dict, Optional
import numpy as np
import pandas as pd
import anndata as ad
import os
import json
import tempfile
import shutil
from tqdm.auto import tqdm

from pathlib import Path
import attr
from torchbiggraph.config import (
    add_to_sys_path,
    ConfigFileLoader
)
from torchbiggraph.converters.importers import (
    convert_input_data,
    TSVEdgelistReader
)
from torchbiggraph.train import train as train_tbg
from torchbiggraph.util import (
    set_logging_verbosity,
    setup_logging,
    SubprocessInitializer,
)

from .settings import settings
 


def gen_graph(list_CP=None,
              list_PM=None,
              list_CG=None,
              list_CC_Spatial=None,
              list_CC_Anchor=None,
              list_GP=None,
              prefix_C='C',
              prefix_P='P',
              prefix_M='M',
              prefix_G='G',
              weight_CP = 1,
              weight_PM = 1,
              weight_CC_Spatial = 5,  # weight for spatial constraints
              weight_CC_Anchor = 10,  # weight for MNN pairs
              weight_GP = 1,
              layer='unig',
              copy=False,
              dirname='graph0',
              ):
    """Generate graph for PBG training.

    Observations and variables of each Anndata object will be encoded
    as nodes (entities). The non-zero values in `.layers['unig']` (by default)
    or `.X` (if `.layers['unig']` does not exist) indicate the edges
    between nodes.
    Nodes between different anndata objects will be automatically matched
    based on `.obs_names` and `.var_names`. Each anndata object indicates one
    or more relation types.

    It also generates an accompanying file 'entity_alias.tsv' to map
    the indices to the aliases used in the graph.

    Parameters
    ----------
    list_CP: `list`, optional (default: None)
        A list of anndata objects that store ATAC-seq data (Cells by Peaks)
        The default weight of cell-peak relation type is 1.0.
    list_PM: `list`, optional (default: None)
        A list of anndata objects that store relation between Peaks and Motifs
    list_CG: `list`, optional (default: None)
        A list of anndata objects that store RNA-seq data (Cells by Genes)
    list_CC_Spatial: `list`, optional (default: None)
        A list of anndata objects that store relation between Cells
        from two conditions (Spatial constraints)
    list_CC_Anchor: `list`, optional (default: None)
        A list of anndata objects that store relation between Cells
        from two conditions (MNN pairs)
    list_GP: `list`, optional (default: None)
        A list of AnnData objects encoding gene-peak relations.
        Rows are genes (obs), columns are peaks (var).
        Non-zero values indicate connection strength or presence.
    prefix_C: `str`, optional (default: 'C')
        Prefix to indicate the entity type of cells
    prefix_G: `str`, optional (default: 'G')
        Prefix to indicate the entity type of genes
    layer: `str`, optional (default: 'unig')
        The layer in AnnData to use for constructing the graph.
        If `layer` is None or the specificed layer does not exist,
        `.X` in AnnData will be used instead.
    dirname: `str`, (default: 'graph0')
        The name of the directory in which each graph will be stored

    copy: `bool`, optional (default: False)
        If True, it returns the graph file as a data frame
    Returns
    -------
    If `copy` is True,
    edges: `pd.DataFrame`
        The edges of the graph used for PBG training.
        Each line contains information about one edge.
        Using tabs as separators, each line contains the identifiers of
        the source entities, the relation types and the target entities.

    updates `.settings.pbg_params` with the following parameters.
    entity_path: `str`
        The path of the directory containing entity count files.
    edge_paths: `list`
        A list of paths to directories containing (partitioned) edgelists.
        Typically a single path is provided.
    entities: `dict`
        The entity types.
    relations: `list`
        The relation types.

    updates `.settings.graph_stats` with the following parameters.
    `dirname`: `dict`
        Statistics of input graph
    """

    if sum(list(map(lambda x: x is None,
                    [list_CP,
                     list_PM,
                     list_CG,
                     list_CC_Spatial,
                     list_CC_Anchor,
                     list_GP]))) == 6:
        return 'No graph is generated'
    
    filepath = os.path.join(settings.workdir, 'pbg', dirname)
    settings.pbg_params['entity_path'] = \
        os.path.join(filepath, "input/entity")
    settings.pbg_params['edge_paths'] = \
        [os.path.join(filepath, "input/edge"), ]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Collect the indices of entities
    dict_cells = dict()  # unique cell indices from all anndata objects
    ids_genes = pd.Index([])
    ids_peaks = pd.Index([])
    ids_motifs = pd.Index([])

    if list_CG is not None:
        for adata_ori in list_CG:
            adata = adata_ori.copy()
            ids_cells_i = adata.obs.index
            if len(dict_cells) == 0:
                dict_cells[prefix_C] = ids_cells_i
            else:
                # check if cell indices are included in dict_cells
                flag_included = False
                for k in dict_cells.keys():
                    ids_cells_k = dict_cells[k]
                    if set(ids_cells_i) <= set(ids_cells_k):
                        flag_included = True
                        break
                if not flag_included:
                    # create a new set of entities
                    # when not all indices are included
                    dict_cells[
                        f'{prefix_C}{len(dict_cells)+1}'] = \
                            ids_cells_i
            ids_genes = ids_genes.union(adata.var.index)
    
    
    if list_CP is not None:
        for adata_ori in list_CP:
            adata = adata_ori.copy()
            ids_cells_i = adata.obs.index
            if len(dict_cells) == 0:
                dict_cells[prefix_C] = ids_cells_i
            else:
                # check if cell indices are included in dict_cells
                flag_included = False
                for k in dict_cells.keys():
                    ids_cells_k = dict_cells[k]
                    if set(ids_cells_i) <= set(ids_cells_k):
                        flag_included = True
                        break
                if not flag_included:
                    # create a new set of entities
                    # when not all indices are included
                    dict_cells[
                        f'{prefix_C}{len(dict_cells)+1}'] = \
                            ids_cells_i
            ids_peaks = ids_peaks.union(adata.var.index)
       
    if list_PM is not None:
        for adata_ori in list_PM:
            adata = adata_ori.copy()
            ids_peaks = ids_peaks.union(adata.obs.index)
            ids_motifs = ids_motifs.union(adata.var.index)
    
    if list_GP is not None:
        for adata_ori in list_GP:
            adata = adata_ori.copy()
            ids_peaks = ids_peaks.union(adata.var.index)
            ids_genes = ids_genes.union(adata.obs.index)
       
       
    entity_alias = pd.DataFrame(columns=['alias'])
    dict_df_cells = dict()  # unique cell dataframes
    for k in dict_cells.keys():
        dict_df_cells[k] = pd.DataFrame(
            index=dict_cells[k],
            columns=['alias'],
            data=[f'{k}.{x}' for x in range(len(dict_cells[k]))])
        settings.pbg_params['entities'][k] = {'num_partitions': 1}
        entity_alias = pd.concat(
            [entity_alias, dict_df_cells[k]],
            ignore_index=False)
    if len(ids_genes) > 0:
        df_genes = pd.DataFrame(
                index=ids_genes,
                columns=['alias'],
                data=[f'{prefix_G}.{x}' for x in range(len(ids_genes))])
        settings.pbg_params['entities'][prefix_G] = {'num_partitions': 1}
        entity_alias = pd.concat(
            [entity_alias, df_genes],
            ignore_index=False)
       
    if len(ids_peaks) > 0:
        df_peaks = pd.DataFrame(
                index=ids_peaks,
                columns=['alias'],
                data=[f'{prefix_P}.{x}' for x in range(len(ids_peaks))])
        settings.pbg_params['entities'][prefix_P] = {'num_partitions': 1}
        entity_alias = pd.concat(
            [entity_alias, df_peaks],
            ignore_index=False)
       
    if len(ids_motifs) > 0:
        df_motifs = pd.DataFrame(
            index=ids_motifs,
            columns=['alias'],
            data=[f'{prefix_M}.{x}' for x in range(len(ids_motifs))])
        settings.pbg_params['entities'][prefix_M] = {'num_partitions': 1}
        entity_alias = pd.concat(
            [entity_alias, df_motifs],
            ignore_index=False)

    # generate edges
    dict_graph_stats = dict()
    col_names = ["source", "relation", "destination", "weight"]
    df_edges = pd.DataFrame(columns=col_names)
    id_r = 0
    settings.pbg_params['relations'] = []

    if list_CP is not None:
        for i, adata_ori in enumerate(list_CP):
            adata = adata_ori.copy()
            # select reference of cells
            for key, df_cells in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells.index):
                    break
            if layer is not None:
                if layer in adata.layers.keys():
                    arr_unig = adata.layers[layer]
                else:
                    print(f'`{layer}` does not exist in anndata {i} '
                            'in `list_CP`.`.X` is being used instead.')
                    arr_unig = adata.X
            else:
                arr_unig = adata.X
           
            _row, _col = arr_unig.nonzero()
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_cells.loc[
                adata.obs_names[_row], 'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_peaks.loc[
                adata.var_names[_col], 'alias'].values
            # Add actual feature values as edge weights for MSE loss
            if hasattr(arr_unig, 'toarray'):  # sparse matrix
                weight_values = arr_unig[_row, _col].A1  # extract as 1D array
            else:  # dense matrix
                weight_values = arr_unig[_row, _col]
            df_edges_x['weight'] = weight_values
            
            settings.pbg_params['relations'].append({
                'name': f'r{id_r}',
                'lhs': f'{key}',
                'rhs': f'{prefix_P}',
                'operator': 'none',
                'weight': weight_CP
                })
            dict_graph_stats[f'relation{id_r}'] = {
                'source': key,
                'destination': prefix_P,
                'n_edges': df_edges_x.shape[0]}
            print(
                f'relation{id_r}: '
                f'source: {key}, '
                f'destination: {prefix_P}\n'
                f'#edges: {df_edges_x.shape[0]}')
            id_r += 1
            df_edges = pd.concat(
                [df_edges, df_edges_x],
                ignore_index=True)
            adata_ori.obs['pbg_id'] = ""
            adata_ori.var['pbg_id'] = ""
            adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                df_cells.loc[adata.obs_names, 'alias'].copy()
            adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                df_peaks.loc[adata.var_names, 'alias'].copy()

    if list_PM is not None:
        for i, adata_ori in enumerate(list_PM):
            adata = adata_ori.copy()
            if layer is not None:
                if layer in adata.layers.keys():
                    arr_unig = adata.layers[layer]
                else:
                    print(f'`{layer}` does not exist in anndata {i} '
                            'in `list_PM`.`.X` is being used instead.')
                    arr_unig = adata.X
            else:
                arr_unig = adata.X
        
            _row, _col = arr_unig.nonzero()
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_peaks.loc[
                adata.obs_names[_row], 'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_motifs.loc[
                adata.var_names[_col], 'alias'].values
            # Add default weight for Peak-Motif relations
            df_edges_x['weight'] = weight_PM
            
            settings.pbg_params['relations'].append({
                'name': f'r{id_r}',
                'lhs': f'{prefix_P}',
                'rhs': f'{prefix_M}',
                'operator': 'none',
                'weight': weight_PM
                })
            dict_graph_stats[f'relation{id_r}'] = {
                'source': prefix_P,
                'destination': prefix_M,
                'n_edges': df_edges_x.shape[0]}
            print(
                f'relation{id_r}: '
                f'source: {prefix_P}, '
                f'destination: {prefix_M}\n'
                f'#edges: {df_edges_x.shape[0]}')

            id_r += 1
            df_edges = pd.concat(
                [df_edges, df_edges_x],
                ignore_index=True)
            adata_ori.obs['pbg_id'] = ""
            adata_ori.var['pbg_id'] = ""
            adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                df_peaks.loc[adata.obs_names, 'alias'].copy()
            adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                df_motifs.loc[adata.var_names, 'alias'].copy()

    if list_CG is not None:
        for i, adata_ori in enumerate(list_CG):
            adata = adata_ori.copy()
            # select reference of cells
            for key, df_cells in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells.index):
                    break
            if layer is not None:
                if layer in adata.layers.keys():
                    arr_unig = adata.layers[layer]
                else:
                    print(f'`{layer}` does not exist in anndata {i} '
                            'in `list_CG`.`.X` is being used instead.')
                    arr_unig = adata.X
            else:
                arr_unig = adata.X
            
            expr_level = np.unique(arr_unig.data)
            expr_weight = np.linspace(
                start=1, stop=5, num=len(expr_level))
            for i_lvl, lvl in enumerate(expr_level):
                _row, _col = (arr_unig == lvl).astype(int).nonzero()
                df_edges_x = pd.DataFrame(columns=col_names)
                df_edges_x['source'] = df_cells.loc[
                    adata.obs_names[_row], 'alias'].values
                df_edges_x['relation'] = f'r{id_r}'
                df_edges_x['destination'] = df_genes.loc[
                    adata.var_names[_col], 'alias'].values
                # Add actual expression values as edge weights for MSE loss  
                df_edges_x['weight'] = [lvl] * len(_row)
                settings.pbg_params['relations'].append({
                    'name': f'r{id_r}',
                    'lhs': f'{key}',
                    'rhs': f'{prefix_G}',
                    'operator': 'none',
                    'weight': round(expr_weight[i_lvl], 2),
                    })
                print(
                    f'relation{id_r}: '
                    f'source: {key}, '
                    f'destination: {prefix_G}\n'
                    f'#edges: {df_edges_x.shape[0]}')
                dict_graph_stats[f'relation{id_r}'] = {
                    'source': key,
                    'destination': prefix_G,
                    'n_edges': df_edges_x.shape[0]}
                id_r += 1
                df_edges = pd.concat(
                    [df_edges, df_edges_x], ignore_index=True)
                

            adata_ori.obs['pbg_id'] = ""
            adata_ori.var['pbg_id'] = ""
            adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                df_cells.loc[adata.obs_names, 'alias'].copy()
            adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                df_genes.loc[adata.var_names, 'alias'].copy()

    if list_CC_Spatial is not None:
        for i, adata in enumerate(list_CC_Spatial):
            # select reference of cells
            for key_obs, df_cells_obs in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells_obs.index):
                    break
            for key_var, df_cells_var in dict_df_cells.items():
                if set(adata.var_names) <= set(df_cells_var.index):
                    break
            if layer is not None:
                if layer in adata.layers.keys():
                    arr_unig = adata.layers[layer]
                else:
                    print(f'`{layer}` does not exist in anndata {i} '
                            'in `list_CC_Spatial`.`.X` is being used instead.')
                    arr_unig = adata.X
            else:
                arr_unig = adata.X
            _row, _col = arr_unig.nonzero()
            #  edges between ref and query
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_cells_obs.loc[
                adata.obs_names[_row], 'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_cells_var.loc[
                adata.var_names[_col], 'alias'].values
            # Add actual feature values as edge weights for Cell-Cell relations
            if hasattr(arr_unig, 'toarray'):  # sparse matrix
                weight_values = arr_unig[_row, _col].A1  # extract as 1D array
            else:  # dense matrix
                weight_values = arr_unig[_row, _col]
            df_edges_x['weight'] = weight_values
            
            settings.pbg_params['relations'].append({
                'name': f'r{id_r}',
                'lhs': f'{key_obs}',
                'rhs': f'{key_var}',
                'operator': 'none',
                'weight': weight_CC_Spatial
                })
            print(
                f'relation{id_r} (Spatial): '
                f'source: {key_obs}, '
                f'destination: {key_var}\n'
                f'#edges: {df_edges_x.shape[0]}')
            dict_graph_stats[f'relation{id_r}'] = {
                'source': key_obs,
                'destination': key_var,
                'n_edges': df_edges_x.shape[0]}

            id_r += 1
            df_edges = pd.concat(
                [df_edges, df_edges_x],
                ignore_index=True)
            adata.obs['pbg_id'] = df_cells_obs.loc[
                adata.obs_names, 'alias'].copy()
            adata.var['pbg_id'] = df_cells_var.loc[
                adata.var_names, 'alias'].copy()

    if list_CC_Anchor is not None:
        for i, adata in enumerate(list_CC_Anchor):
            # select reference of cells
            for key_obs, df_cells_obs in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells_obs.index):
                    break
            for key_var, df_cells_var in dict_df_cells.items():
                if set(adata.var_names) <= set(df_cells_var.index):
                    break
            if layer is not None:
                if layer in adata.layers.keys():
                    arr_unig = adata.layers[layer]
                else:
                    print(f'`{layer}` does not exist in anndata {i} '
                            'in `list_CC_Anchor`.`.X` is being used instead.')
                    arr_unig = adata.X
            else:
                arr_unig = adata.X
            _row, _col = arr_unig.nonzero()
            #  edges between ref and query
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_cells_obs.loc[
                adata.obs_names[_row], 'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_cells_var.loc[
                adata.var_names[_col], 'alias'].values
            # Add actual feature values as edge weights for Cell-Cell relations
            if hasattr(arr_unig, 'toarray'):  # sparse matrix
                weight_values = arr_unig[_row, _col].A1  # extract as 1D array
            else:  # dense matrix
                weight_values = arr_unig[_row, _col]
            df_edges_x['weight'] = weight_values
            
            settings.pbg_params['relations'].append({
                'name': f'r{id_r}',
                'lhs': f'{key_obs}',
                'rhs': f'{key_var}',
                'operator': 'none',
                'weight': weight_CC_Anchor
                })
            print(
                f'relation{id_r} (Achor): '
                f'source: {key_obs}, '
                f'destination: {key_var}\n'
                f'#edges: {df_edges_x.shape[0]}')
            dict_graph_stats[f'relation{id_r}'] = {
                'source': key_obs,
                'destination': key_var,
                'n_edges': df_edges_x.shape[0]}

            id_r += 1
            df_edges = pd.concat(
                [df_edges, df_edges_x],
                ignore_index=True)
            adata.obs['pbg_id'] = df_cells_obs.loc[
                adata.obs_names, 'alias'].copy()
            adata.var['pbg_id'] = df_cells_var.loc[
                adata.var_names, 'alias'].copy()
                
    if list_GP is not None:
        for i, adata_ori in enumerate(list_GP):
            adata = adata_ori.copy()
            if layer is not None:
                if layer in adata.layers.keys():
                    arr_unig = adata.layers[layer]
                else:
                    print(f'`{layer}` does not exist in anndata {i} '
                            'in `list_GP`.`.X` is being used instead.')
                    arr_unig = adata.X
            else:
                arr_unig = adata.X
            _row, _col = arr_unig.nonzero()

            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_genes.loc[
                adata.obs_names[_row], 'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_peaks.loc[
                adata.var_names[_col], 'alias'].values
            # Add actual feature values as edge weights for Gene-Peak relations
            if hasattr(arr_unig, 'toarray'):  # sparse matrix
                weight_values = arr_unig[_row, _col].A1  # extract as 1D array
            else:  # dense matrix
                weight_values = arr_unig[_row, _col]
            df_edges_x['weight'] = weight_values

            settings.pbg_params['relations'].append({
                'name': f'r{id_r}',
                'lhs': prefix_G,
                'rhs': prefix_P,
                'operator': 'none',
                'weight': weight_GP
            })

            print(
                f'relation{id_r}: '
                f'source: {prefix_G}, '
                f'destination: {prefix_P}\n'
                f'#edges: {df_edges_x.shape[0]}')

            dict_graph_stats[f'relation{id_r}'] = {
                'source': prefix_G,
                'destination': prefix_P,
                'n_edges': df_edges_x.shape[0]
            }

            id_r += 1
            df_edges = pd.concat([df_edges, df_edges_x], ignore_index=True)

            adata_ori.obs['pbg_id'] = df_genes.loc[
                adata.obs_names, 'alias'].copy()
            adata_ori.var['pbg_id'] = df_peaks.loc[
                adata.var_names, 'alias'].copy()
    print(f'Total number of edges: {df_edges.shape[0]}')
    dict_graph_stats['n_edges'] = df_edges.shape[0]
    settings.graph_stats[dirname] = dict_graph_stats

    print(f'Writing graph file "pbg_graph.txt" to "{filepath}" ...')
    df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
                    header=False,
                    index=False,
                    sep='\t')
    
    # Save index-to-alias mapping files for MNN validation
    for entity_prefix in dict_df_cells.keys():
        if 'C' in entity_prefix: # Covers C, C2, C3 etc.
            # These are already sorted by PBG's internal processing order
            aliases = dict_df_cells[entity_prefix]['alias']
            aliases.to_csv(
                os.path.join(filepath, f'{entity_prefix}_idx_to_alias.tsv'),
                header=False,
                index=False,
                sep='\t'
            )

    entity_alias.to_csv(os.path.join(filepath, 'entity_alias.txt'),
                        header=True,
                        index=True,
                        sep='\t')
    with open(os.path.join(filepath, 'graph_stats.json'), 'w') as fp:
        json.dump(dict_graph_stats,
                  fp,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))
    print("Finished.")
    settings.graph_stats[dirname]['entities'] = settings.pbg_params['entities']
    settings.graph_stats[dirname]['relations'] = settings.pbg_params['relations']
    
    if copy:
        return df_edges
    else:
        return None

def pbg_train(dirname=None,
              pbg_params=None,
              output='model',
              auto_wd=True,
              save_wd=False,
              seed=42):
    """PBG training
    (docstring remains the same)
    """
    configure_runtime_warnings()
    if pbg_params is None:
        pbg_params = settings.pbg_params.copy()
    else:
        assert isinstance(pbg_params, dict),\
            "`pbg_params` must be dict"

    # Set the random seed
    pbg_params['seed'] = seed
    
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA runs more reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed} for reproducibility")

    if dirname is None:
        filepath = Path(pbg_params['entity_path']).parent.parent.as_posix()
        dirname = os.path.basename(filepath)
    else:
        filepath = os.path.join(settings.workdir, 'pbg', dirname)
    pbg_params['checkpoint_path'] = os.path.join(filepath, output)
    settings.pbg_params['checkpoint_path'] = pbg_params['checkpoint_path']
  
    if auto_wd:
        # empirical numbers from simulation experiments
        if settings.graph_stats[
                os.path.basename(filepath)]['n_edges'] < 5e7:  
            # optimial wd (0.013) for sample size (2725781)
            wd = 0.013 * 2725781 / settings.graph_stats[
                    os.path.basename(filepath)]['n_edges']
        else:
            # optimial wd (0.0004) for sample size (59103481)
            wd = 0.0004 * 59103481 / settings.graph_stats[
                    os.path.basename(filepath)]['n_edges']
        print(f'Auto-estimated weight decay: {wd:.6E}')
        pbg_params['wd'] = wd
        if save_wd:
            settings.pbg_params['wd'] = pbg_params['wd']
            print(f"Updated weight decay to {wd:.6E}")


    # to avoid oversubscription issues in workloads
    # that involve nested parallelism
    os.environ["OMP_NUM_THREADS"] = "1"

    # Remove custom joint-loss keys before loading the TBG config schema.
    EXTRA_KEYS = [
        'alpha_ot_rna',
        'alpha_ot_atac',
        'ot_swd_num_projections',
        'z_softmax_temp',
        's',  # ATAC dropout probability
        'rna_noise_level', # RNA noise level
        'alpha_triplet',
        'begin_triplet_epoch',
        'triplet_update_freq',
        'triplet_num_samples',
        'seed',
        'alpha_ot_gp', # Sliced Wasserstein Distance on connected Gene-Peak pairs
        'rna_aggregation', # Whether to aggregate RNA data
        'atac_aggregation', # Whether to aggregate ATAC data
    ]
    extra_kwargs = {k: pbg_params[k] for k in EXTRA_KEYS if k in pbg_params}
    pbg_params_filtered = {k: v for k, v in pbg_params.items() if k not in EXTRA_KEYS}

    use_ext = False
    # Use the extended trainer when any joint-loss term is enabled.
    if extra_kwargs.get('alpha_ot_rna', 0) > 0 or extra_kwargs.get('alpha_ot_atac', 0) > 0 or extra_kwargs.get('alpha_triplet', 0) > 0 or extra_kwargs.get('alpha_ot_gp', 0) > 0:
        use_ext = True
    if use_ext:
        enabled_losses = []
        if extra_kwargs.get('alpha_ot_rna', 0) > 0:
            enabled_losses.append(f"RNA-OT={extra_kwargs['alpha_ot_rna']:g}")
        if extra_kwargs.get('alpha_ot_atac', 0) > 0:
            enabled_losses.append(f"ATAC-OT={extra_kwargs['alpha_ot_atac']:g}")
        if extra_kwargs.get('alpha_triplet', 0) > 0:
            enabled_losses.append(f"triplet={extra_kwargs['alpha_triplet']:g}")
        if extra_kwargs.get('alpha_ot_gp', 0) > 0:
            enabled_losses.append(f"GP-OT={extra_kwargs['alpha_ot_gp']:g}")
        print(f"Training mode: joint ({', '.join(enabled_losses)})")
    else:
        print("Training mode: standard")

    tbg_tmpdir = os.path.join(filepath, "tmp")
    os.makedirs(tbg_tmpdir, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = tbg_tmpdir
    try:
        loader = ConfigFileLoader()
    finally:
        tempfile.tempdir = previous_tempdir
    config = loader.load_config_simba(pbg_params_filtered)
    set_logging_verbosity(config.verbose)

    list_filenames = [os.path.join(filepath, "pbg_graph.txt")]
    input_edge_paths = [Path(name) for name in list_filenames]
    print("Converting input data...")
    
    # Always read the 4th column as per-edge value for downstream losses (e.g., MSE/OT),
    # while ranking loss in TBG still uses relation weights from config.
    if use_ext:
        print("  Reading edge values from 4th column for downstream losses...")
    edge_reader = TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1, weight_col=3)
        
    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        edge_reader,
        dynamic_relations=config.dynamic_relations,
        )
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(configure_runtime_warnings)
    subprocess_init.register(setup_logging, config.verbose)
    if loader.config_dir is not None:
        subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    train_config = attr.evolve(config, edge_paths=config.edge_paths)
    
    print(f"Training config: {len(config.entities)} entities, {len(config.relations)} relations")
    print(f"Output directory: {pbg_params['checkpoint_path']}")
    
    print("Starting training...")
    if use_ext:
        try:
            from .tbg_ext.train_ext import train as train_ext
            print("Trainer: extended")
            # Pass the seed through to the extended trainer.
            extra_kwargs['seed'] = seed
            train_ext(
                config=config,
                subprocess_init=subprocess_init,
                alpha_ot_rna=pbg_params.get('alpha_ot_rna', 5e4),
                alpha_ot_atac=pbg_params.get('alpha_ot_atac', 5e4),
                ot_swd_num_projections=pbg_params.get('ot_swd_num_projections', 50),
                z_softmax_temp=pbg_params.get('z_softmax_temp', 1.0),
                s=pbg_params.get('s', 0.95),
                rna_noise_level=pbg_params.get('rna_noise_level', 0.05),
                alpha_triplet=pbg_params.get('alpha_triplet', 0),
                begin_triplet_epoch=pbg_params.get('begin_triplet_epoch', 10),
                triplet_update_freq=pbg_params.get('triplet_update_freq', 2),
                triplet_num_samples=pbg_params.get('triplet_num_samples', 512),
                seed=pbg_params.get('seed', 42),
                alpha_ot_gp=pbg_params.get('alpha_ot_gp', 100),
                rna_aggregation=pbg_params.get('rna_aggregation', True),
                atac_aggregation=pbg_params.get('atac_aggregation', True),
            )
        except Exception as e:
            print(f"Extended trainer failed: {e}")
            print("Falling back to standard TBG trainer...")
            train_tbg(train_config, subprocess_init=subprocess_init)
    else:
        train_tbg(train_config, subprocess_init=subprocess_init)
    shutil.rmtree(tbg_tmpdir, ignore_errors=True)
    print("Training completed")


def run_training_stages(base_params, stages):
    """Run ordered PBG training stages and keep each stage config."""
    configs = {}
    previous_config = base_params.copy()
    previous_checkpoint = None
    previous_output = None

    for stage in stages:
        name = stage["name"]
        resume = stage.get("resume_from_previous", False)
        output = stage.get("output")
        config = previous_config.copy()
        config.update(stage.get("params", {}))

        if resume:
            if previous_checkpoint is None or not os.path.exists(previous_checkpoint):
                raise RuntimeError(
                    f"Cannot start {name}: previous checkpoint was not found."
                )
            output = output or previous_output
            config["checkpoint_path"] = previous_checkpoint
            print(f"Resume checkpoint:\n{os.path.abspath(previous_checkpoint)}")

        if output is None:
            raise ValueError(f"Stage {name} needs an output model name.")

        print("=" * 15, f"Start {name}", "=" * 15)
        pbg_train(pbg_params=config, output=output)
        print("=" * 15, f"Finished {name}", "=" * 15)

        configs[name] = config
        previous_config = config
        previous_checkpoint = config["checkpoint_path"]
        previous_output = output

    return configs
