#!/usr/bin/env python3
"""
convert_biokg_to_csr.py
-----------------------
This script converts the OGB BioKG heterogeneous graph to a homogeneous CSR matrix
for a specified node type. It loads the dataset using OGB's API, extracts edges where both
the source and destination nodes are of the given type, constructs a CSR matrix, and saves
the 'indptr' and 'indices' arrays as numpy files.

Usage:
    python convert_biokg_to_csr.py --node_type gene --output_dir ./data/biokg_csr
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp
from ogb.linkproppred import LinkPropPredDataset

def load_biokg_dataset():
    """
    Loads the OGB BioKG dataset.

    Returns:
        data (dict): The loaded graph data containing keys such as 'edge_index_dict'
                     and 'num_nodes_dict' (for heterogeneous graphs).
    """
    dataset = LinkPropPredDataset(name='ogbl-biokg')
    data = dataset[0]
    return data

def list_available_node_types(data):
    """
    Lists the available node types in the dataset (if the dataset is heterogeneous).

    Parameters:
        data (dict): The BioKG data dictionary.

    Returns:
        list: A list of available node types (or an empty list if not found).
    """
    if 'num_nodes_dict' in data:
        return list(data['num_nodes_dict'].keys())
    else:
        return []

def build_csr_for_node_type(data, target_node_type):
    """
    Builds a CSR matrix for the specified node type by extracting all edges
    where both the source and target nodes are of that type.

    Parameters:
        data (dict): The BioKG data dictionary.
        target_node_type (str): The node type for which to build the CSR matrix.
    
    Returns:
        csr (scipy.sparse.csr_matrix): The resulting CSR matrix.
    
    Raises:
        KeyError: If the expected keys are missing from the dataset.
        ValueError: If the specified node type is not present.
    """
    if 'edge_index_dict' not in data or 'num_nodes_dict' not in data:
        raise KeyError("The dataset does not contain 'edge_index_dict' or 'num_nodes_dict'. "
                       "Please ensure you are using the heterogeneous BioKG dataset.")
    
    if target_node_type not in data['num_nodes_dict']:
        raise ValueError(f"Target node type '{target_node_type}' not found. "
                         f"Available types: {list(data['num_nodes_dict'].keys())}")
    
    num_nodes = data['num_nodes_dict'][target_node_type]
    print(f"Number of nodes for type '{target_node_type}': {num_nodes}")
    
    rows = []
    cols = []
    
    # Iterate over each relation in the heterogeneous graph.
    edge_index_dict = data['edge_index_dict']
    for (src_type, relation, dst_type), edge_index in edge_index_dict.items():
        # Only keep edges where both the source and destination are of the target type.
        if src_type == target_node_type and dst_type == target_node_type:
            # edge_index is expected to be a tensor of shape [2, num_edges]
            # If it is a PyTorch tensor, convert it to a NumPy array.
            if hasattr(edge_index, 'numpy'):
                edge_index_np = edge_index.numpy()
            else:
                edge_index_np = edge_index  # assume it is already a numpy array
            src_nodes = edge_index_np[0]
            dst_nodes = edge_index_np[1]
            rows.extend(src_nodes.tolist())
            cols.extend(dst_nodes.tolist())
            print(f"Relation '{relation}' contributed {len(src_nodes)} edges.")
    
    # Create the CSR matrix with a value of 1 for each edge.
    data_values = np.ones(len(rows), dtype=np.int32)
    csr = sp.csr_matrix((data_values, (rows, cols)), shape=(num_nodes, num_nodes))
    return csr

def save_csr(csr, output_dir):
    """
    Saves the CSR matrix arrays (indptr and indices) to the specified output directory.

    Parameters:
        csr (scipy.sparse.csr_matrix): The CSR matrix.
        output_dir (str): The directory to save the numpy files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    indptr_path = os.path.join(output_dir, "csr_indptr.npy")
    indices_path = os.path.join(output_dir, "csr_indices.npy")
    
    np.save(indptr_path, csr.indptr)
    np.save(indices_path, csr.indices)
    print(f"Saved CSR indptr to {indptr_path}")
    print(f"Saved CSR indices to {indices_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert OGB BioKG to a CSR matrix for a specific node type")
    parser.add_argument('--node_type', type=str, default='gene',
                        help="The target node type to extract edges for (default: 'gene').")
    parser.add_argument('--output_dir', type=str, default='./data/biokg_csr',
                        help="Directory where the CSR numpy files will be saved (default: './data/biokg_csr').")
    args = parser.parse_args()
    
    # Load the BioKG dataset.
    data = load_biokg_dataset()
    
    # List available node types and print them.
    available_types = list_available_node_types(data)
    print("Available node types in dataset:", available_types)
    
    if args.node_type not in available_types:
        print(f"Warning: Specified node type '{args.node_type}' not found in dataset.")
        if available_types:
            chosen_type = available_types[0]
            print(f"Falling back to the first available node type: '{chosen_type}'.")
            args.node_type = chosen_type
        else:
            raise ValueError("No node types found in the dataset.")
    
    # Build the CSR matrix for the specified node type.
    csr_matrix = build_csr_for_node_type(data, args.node_type)
    
    # Save the CSR matrix components.
    save_csr(csr_matrix, args.output_dir)
    
    print("Conversion to CSR matrix completed successfully.")

if __name__ == "__main__":
    main()
