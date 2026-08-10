"""
We have different sources of data that we want to ingest into our lake. This module allows us to upload pcaps into the lake.
pcaps are stored in the raw zone of the lake. 
the source of data is required as an argument to this module. 
the lake can be local or remote (AWS S3).

after running talos init, we have the lake dir created, where the structure is as follows:
lake/
    raw/
    extracted/
    parquets/
    labelled/
    mapped/
the raw zone is where we will upload the pcaps.
"""

import argparse
import logging
from pathlib import Path

# Leverage existing Talos modules for config and storage abstraction
from talos.common.config import Config
from talos.common.lake.lake import LakeClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def upload_pcaps_to_lake(config_path: str, dataset: str, pcap_files: list[str]):
    """Copy pcap files into the lake raw zone.
    
    config_path: path to the central config.yml
    dataset: the logical dataset name (e.g., 'cic-ids-2017')
    pcap_files: iterable of paths (str) to pcap files
    """
    # 1. Load the central configuration
    config = Config.load(config_path)
    
    # 2. Initialize the LakeClient (handles Local vs S3 automatically)
    lake = LakeClient(root=config.root, region=config.region)
    
    for pcap_file in pcap_files:
        src = Path(pcap_file)
        
        if not src.exists():
            logging.warning(f"File not found, skipping: {src}")
            continue
            
        # 3. Resolve the exact URI for the raw zone, scoped by dataset
        dest_uri = lake.uri(zone="raw", dataset=dataset, rel=src.name)
        
        logging.info(f"Uploading {src.name} to {dest_uri}...")
        
        # 4. Put the file (automatically delegates to fsspec for S3 or shutil for local)
        lake.put(local=src, uri=dest_uri)
        
        logging.info(f"Successfully uploaded {src.name}.")


def main():
    parser = argparse.ArgumentParser(description="Upload pcap files to the Talos lake raw zone.")
    
    parser.add_argument("--config", default=None, help="Path to config.yml (defaults to $TALOS_CONFIG or ./config.yml)")
    parser.add_argument("--dataset", required=True, help="Target dataset name (e.g., cic-ids-2017)")
    parser.add_argument("pcap_files", nargs="+", help="Paths to local pcap files to upload")
    
    args = parser.parse_args()

    upload_pcaps_to_lake(
        config_path=args.config,
        dataset=args.dataset,
        pcap_files=args.pcap_files
    )


if __name__ == "__main__":
    main()