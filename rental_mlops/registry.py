from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import write_model_artifact
from .quality import DEFAULT_GATE


@dataclass(frozen=True)
class RegistryRecord:
    model_name: str
    version: str
    stage: str
    artifact_path: str
    image_tag: str
    created_at: str
    quality: dict[str, Any]
    approval_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_registry_record(
    data_path: str | Path,
    artifact_path: str | Path,
    model_name: str = "rental-price-regressor",
    version: str = "1.0.0",
    stage: str = "staging",
    image_tag: str = "1.0.0",
) -> RegistryRecord:
    metadata = write_model_artifact(artifact_path, data_path, gate=DEFAULT_GATE)
    quality = metadata["quality"]
    return RegistryRecord(
        model_name=model_name,
        version=version,
        stage=stage,
        artifact_path=str(artifact_path),
        image_tag=image_tag,
        created_at=datetime.now(timezone.utc).isoformat(),
        quality=quality,
        approval_status="approved" if quality["passed"] else "rejected",
    )


def write_registry_record(
    output_path: str | Path,
    data_path: str | Path,
    artifact_path: str | Path,
    model_name: str = "rental-price-regressor",
    version: str = "1.0.0",
    stage: str = "staging",
    image_tag: str = "1.0.0",
) -> RegistryRecord:
    record = build_registry_record(
        data_path=data_path,
        artifact_path=artifact_path,
        model_name=model_name,
        version=version,
        stage=stage,
        image_tag=image_tag,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
