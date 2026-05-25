# FILE: generate_edge_features.py (FIXED VERSION)
# DESCRIPTION: Generates rich edge features (RBF-expanded distances) for each
#              protein based on its 3D structure (PDB file).

import os
import pickle
import numpy as np
import torch
from Bio.PDB import PDBParser
from tqdm import tqdm
import warnings

# --- 1. 配置 ---
PDB_DIR = "./PDB_files/"  # 【重要】确保这里是你存放 PDB 文件的目录
DATASET_PATH = "./GraphPPIS_Dataset/"
FEATURE_PATH = "./GraphPPIS_Feature/"
OUTPUT_DIR_NAME = "edge_features_rbf8"  # 为新特征创建明确的文件夹名

# RBF (Radial Basis Function) 扩展参数
NUM_RBF = 8# 16
RBF_CENTERS = np.linspace(0.0, 20.0, NUM_RBF)
RBF_WIDTH = 1.5


# --- 2. 核心功能函数 ---
def rbf_expand(distances):
    """使用高斯径向基函数将一个距离标量扩展为一个向量"""
    return np.exp(-((distances[..., None] - RBF_CENTERS) ** 2) / RBF_WIDTH ** 2)


def get_ca_coords_from_pdb(pdb_file, chain_id=None):
    """从PDB文件中提取指定链或第一条链的C-alpha原子坐标和序列"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_file)
    except Exception as e:
        print(f"  [Warning] Could not parse PDB file: {pdb_file}. Error: {e}")
        return None, None

    chain = None
    if chain_id:
        try:
            chain = structure[0][chain_id]
        except KeyError:
            print(f"  [Warning] Chain '{chain_id}' not found in {pdb_file}. Trying the first chain...")
            # 如果指定的链找不到，尝试获取第一条链
            if len(structure[0]) > 0:
                chain = list(structure[0].get_chains())[0]
            else:
                return None, None
    else:
        # 如果没有提供 chain_id，默认使用文件中的第一条链
        if len(structure[0]) > 0:
            chain = list(structure[0].get_chains())[0]
        else:
            return None, None

    coords, sequence = [], ""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
    }

    for residue in chain:
        res_name = residue.get_resname()
        if res_name in three_to_one and "CA" in residue:
            coords.append(residue["CA"].get_coord())
            sequence += three_to_one[res_name]

    if not coords:
        return None, None

    return np.array(coords), sequence


def generate_for_dataset(dataset_file, output_dir):
    """读取数据集，为其中每个蛋白质生成边特征"""
    print(f"\nProcessing dataset: {os.path.basename(dataset_file)}")

    with open(dataset_file, "rb") as f:
        data_dict = pickle.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created output directory: {output_dir}")

    processed_count = 0
    skipped_count = 0
    for protein_id, data in tqdm(data_dict.items(),
                                 desc=f"Generating edge features for {os.path.basename(dataset_file)}"):

        # --- 【核心修复】健壮地解析 PDB ID 和 Chain ID ---
        parts = protein_id.split('_')
        pdb_id = parts[0]
        chain_id = parts[1] if len(parts) > 1 else None  # 如果没有下划线，chain_id为None

        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")

        if not os.path.exists(pdb_path):
            skipped_count += 1
            continue

        ca_coords, pdb_seq = get_ca_coords_from_pdb(pdb_path, chain_id)

        if ca_coords is None or pdb_seq is None:
            skipped_count += 1
            continue

        dataset_seq = data[0] if isinstance(data, (list, tuple)) else data.get('sequence', '')
        # 现在序列长度验证更重要，因为链可能是默认选择的
        if len(pdb_seq) != len(dataset_seq):
            # print(f"\n  [Warning] Sequence length mismatch for {protein_id}. PDB: {len(pdb_seq)}, Dataset: {len(dataset_seq)}. Skipping.")
            skipped_count += 1
            continue

        dist_matrix = np.sqrt(np.sum((ca_coords[:, None, :] - ca_coords[None, :, :]) ** 2, axis=-1))
        rbf_dist_features = rbf_expand(dist_matrix).astype(np.float32)

        output_path = os.path.join(output_dir, f"{protein_id}.npy")
        np.save(output_path, rbf_dist_features)
        processed_count += 1

    print(
        f"  Successfully generated {processed_count} edge feature files. Skipped {skipped_count} files (PDB not found or sequence mismatch).")


# --- 3. 执行脚本 ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("--- Starting Edge Feature Generation (RBF-expanded Distances) ---")

    output_base_dir = os.path.join(FEATURE_PATH, OUTPUT_DIR_NAME)

    dataset_files_to_process = ["Train_335.pkl", "Test_60.pkl"]

    for filename in dataset_files_to_process:
        full_path = os.path.join(DATASET_PATH, filename)
        if os.path.exists(full_path):
            generate_for_dataset(full_path, output_base_dir)
        else:
            print(f"\nDataset file not found: {full_path}. Skipping.")

    print("\n--- Edge feature generation complete! ---")
    print(f"--- New edge features are located in: {output_base_dir} ---")