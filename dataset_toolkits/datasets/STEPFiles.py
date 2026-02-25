"""
STEPFiles dataset utilities.

This module handles a simple custom dataset of STEP files in a folder.
Unlike other datasets that pull from HuggingFace CSVs, this discovers
STEP files directly from a source directory.

Expected structure:
    source_dir/
        ├── part1.step
        ├── part2.stp
        └── ...

Output structure:
    output_dir/
        ├── metadata.csv
        ├── raw/{name}.step        # Symlinks to source
        ├── meshes/{name}.obj      # Meshed from STEP
        ├── udf/{name}.npz         # Surface points + UDF
        ├── renders/{name}/        # Multiview renders + transforms.json
        ├── voxels/{name}.ply      # Voxelized mesh
        ├── features/{model}/      # DINOv2 features
        ├── latents/{model}/       # SLAT latents
        └── ss_latents/{model}/    # Sparse structure latents
"""

import os
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    """Add dataset-specific arguments to parser."""
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Directory containing STEP files (only needed for initial setup)')


def get_metadata(source_dir=None, output_dir=None, **kwargs):
    """
    Build metadata from STEP files in source directory.
    
    Uses filename stem as the identifier (instead of SHA256 hash).
    """
    if source_dir is None:
        raise ValueError("--source_dir is required for STEPFiles dataset")
    
    step_files = sorted(
        glob.glob(os.path.join(source_dir, "*.step")) +
        glob.glob(os.path.join(source_dir, "*.stp")) +
        glob.glob(os.path.join(source_dir, "*.STEP")) +
        glob.glob(os.path.join(source_dir, "*.STP"))
    )
    
    if len(step_files) == 0:
        raise ValueError(f"No STEP files found in {source_dir}")
    
    records = []
    for step_file in step_files:
        name = os.path.splitext(os.path.basename(step_file))[0]
        records.append({
            'sha256': name,  # Using filename as identifier
            'name': name,
            'source_path': os.path.abspath(step_file),
            'local_path': f'raw/{name}.step',
            # Default values for filtering (bypass aesthetic score filter)
            'aesthetic_score': 5.5,
            'captions': '["CAD part"]',
        })
    
    return pd.DataFrame(records)


def download(metadata, output_dir, **kwargs):
    """
    "Download" STEP files by creating symlinks to source.
    
    For STEPFiles, this just creates symlinks since files are already local.
    """
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    
    downloaded = {}
    for _, row in metadata.iterrows():
        name = row['sha256']
        source = row['source_path']
        dest = os.path.join(output_dir, 'raw', f'{name}.step')
        
        if not os.path.exists(dest):
            # Create symlink or copy
            if os.path.exists(source):
                os.symlink(source, dest)
                downloaded[name] = f'raw/{name}.step'
            else:
                print(f"Warning: Source file not found: {source}")
        else:
            downloaded[name] = f'raw/{name}.step'
    
    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    """
    Process each instance with the given function.
    
    For STEPFiles, this is simpler than other datasets since files are
    directly accessible (no zip extraction needed).
    """
    metadata_list = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
             tqdm(total=len(metadata_list), desc=desc) as pbar:
            
            def worker(metadatum):
                try:
                    local_path = metadatum.get('local_path')
                    sha256 = metadatum['sha256']
                    
                    if local_path:
                        file_path = os.path.join(output_dir, local_path)
                    else:
                        file_path = metadatum.get('source_path')
                    
                    if file_path and os.path.exists(file_path):
                        record = func(file_path, sha256)
                        if record is not None:
                            records.append(record)
                    else:
                        print(f"File not found for {sha256}: {file_path}")
                    pbar.update()
                except Exception as e:
                    print(f"Error processing {metadatum.get('sha256', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update()
            
            executor.map(worker, metadata_list)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error during processing: {e}")
    
    return pd.DataFrame.from_records(records)


def get_step_file_path(metadata_row, output_dir):
    """Get the actual STEP file path for an instance."""
    local_path = metadata_row.get('local_path')
    if local_path:
        return os.path.join(output_dir, local_path)
    return metadata_row.get('source_path')
