from dbt_loom.config import dbtLoomConfig
from pathlib import Path
import re
import yaml
import os

from dbt_loom.config import (
    FileReferenceConfig,
    LoomConfigurationError,
    ManifestReference,
    ManifestReferenceType,
)


def load_from_s3(config):
    """Load a manifest dictionary from an S3-compatible bucket."""
    from dbt_loom.clients.s3 import S3Client, S3ReferenceConfig

    gcs_client = S3Client(
        bucket_name=config.bucket_name,
        object_name=config.object_name,
    )

    return gcs_client.load_manifest()


def load(manifest_reference):
    """Load a manifest dictionary based on a ManifestReference input."""

    manifest = load_from_s3(manifest_reference.config)

    return manifest


def replace_env_variables(config_str: str) -> str:
    """Replace environment variable placeholders in the configuration string."""
    pattern = r"\$(\w+)|\$\{([^}]+)\}"
    return re.sub(
        pattern,
        lambda match: os.environ.get(
            match.group(1) if match.group(1) is not None else match.group(2), ""
        ),
        config_str,
    )


def read_config(path: Path):
    """Read the dbt-loom configuration file."""
    if not path.exists():
        return None

    with open(path) as file:
        config_content = file.read()

    config_content = replace_env_variables(config_content)

    return dbtLoomConfig(**yaml.load(config_content, yaml.SafeLoader))


path = Path("dbt/dbt_loom.config.yml")

from dbt.cli.main import dbtRunner

dbt_runner = dbtRunner()

res = dbt_runner.invoke(args=["deps"])
