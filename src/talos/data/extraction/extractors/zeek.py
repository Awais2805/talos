import logging
import shutil
import subprocess
from pathlib import Path
from talos.data.extraction.base import BaseExtractor, register_extractor

logger = logging.getLogger(__name__)

@register_extractor
class ZeekExtractor(BaseExtractor):
    def __init__(self, image: str = "zeek/zeek:8.2.1", flags: str = "-C", json_logs: bool = True):
        self.image = image
        self.flags = flags.split() if isinstance(flags, str) else flags
        self.json_logs = json_logs
        # Parse version from Docker image tag, defaulting if missing
        self._version = image.split(":")[-1] if ":" in image else "unknown"

    @property
    def name(self) -> str:
        return "zeek"

    @property
    def version(self) -> str:
        return self._version

    def extract(self, pcap_paths: list[str], output_dir: str) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("docker"):
            raise RuntimeError(f"Docker is required for {self.name} extraction but is not installed.")

        for pcap in pcap_paths:
            pcap_path = Path(pcap).resolve()
            if not pcap_path.exists():
                logger.warning(f"PCAP not found, skipping: {pcap_path}")
                continue

            # Create a dedicated sub-folder for this PCAP to prevent log overwriting
            pcap_out = out_dir / pcap_path.stem
            pcap_out.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting features from {pcap_path.name} using {self.name} v{self.version}...")

            # Mount local paths into the Zeek container
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{pcap_path.parent}:/pcap_in:ro",
                "-v", f"{pcap_out}:/zeek_out",
                "-w", "/zeek_out",
                self.image,
                "zeek"
            ]
            
            cmd.extend(self.flags)
            cmd.extend(["-r", f"/pcap_in/{pcap_path.name}"])

            if self.json_logs:
                cmd.append("LogAscii::use_json=T")
                
            # Optional: Load extra scripts to extract the FULL feature set (e.g. MAC addresses, VLANs)
            cmd.append("protocols/conn/mac-logging") 

            try:
                # Capture output to prevent spamming stdout, but log errors if it fails
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Successfully extracted {pcap_path.name} into {pcap_out}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract {pcap_path.name}. stderr: {e.stderr}")