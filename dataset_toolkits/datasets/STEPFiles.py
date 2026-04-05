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
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def add_args(parser: argparse.ArgumentParser):
    """Add dataset-specific arguments to parser."""
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Directory containing STEP/x_t files (only needed for initial setup)')


def get_metadata(source_dir=None, output_dir=None, **kwargs):
    """
    Build metadata from CAD files in source directory.

    Supports .step, .stp, and .x_t files. Uses SHA256 hash of file content as
    the identifier so NPZ files produced by 'nmr --convert' (which are also
    named by SHA256) match automatically.
    """
    if source_dir is None:
        raise ValueError("--source_dir is required for STEPFiles dataset")

    exts = ("*.step", "*.stp", "*.STEP", "*.STP", "*.x_t", "*.X_T")
    cad_files = sorted(
        f for pat in exts for f in glob.glob(os.path.join(source_dir, pat))
    )

    if len(cad_files) == 0:
        raise ValueError(f"No CAD files (.step/.stp/.x_t) found in {source_dir}")

    records = []
    for cad_file in tqdm(cad_files, desc='Hashing source files', leave=False):
        sha = _sha256(cad_file)
        ext = os.path.splitext(cad_file)[1].lower()
        records.append({
            'sha256': sha,
            'name': os.path.splitext(os.path.basename(cad_file))[0],
            'source_path': os.path.abspath(cad_file),
            'local_path': f'raw/{sha}{ext}',
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
        ext = os.path.splitext(source)[1].lower()
        dest = os.path.join(output_dir, 'raw', f'{name}{ext}')
        
        if not os.path.exists(dest):
            if os.path.exists(source):
                os.symlink(source, dest)
                downloaded[name] = f'raw/{name}{ext}'
            else:
                print(f"Warning: Source file not found: {source}")
        else:
            downloaded[name] = f'raw/{name}{ext}'
    
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
