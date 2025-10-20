from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class AttentionFile:
    """Metadata describing a single NPZ attention file."""

    path: Path
    start_id: int
    end_id: int

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def sample_ids(self) -> range:
        return range(self.start_id, self.end_id)


@dataclass
class SampleRecord:
    """Container for all information required to visualise a sample."""

    file: AttentionFile
    batch_index: int
    source: str
    prediction: str
    attentions: np.ndarray  # (layer, head, seq, seq)

    @property
    def layer_count(self) -> int:
        return int(self.attentions.shape[0])

    @property
    def head_count(self) -> int:
        return int(self.attentions.shape[1])

    @property
    def sequence_length(self) -> int:
        return int(self.attentions.shape[-1])


class AttentionDataset:
    """Utility class that provides convenient access to NPZ attention dumps."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.data_dir}' does not exist")

        self._files: List[AttentionFile] = self._scan_files()
        if not self._files:
            raise FileNotFoundError(
                f"No npz files were found under '{self.data_dir}'."
            )

    def files(self) -> Sequence[AttentionFile]:
        """Return the available files sorted by their starting id."""

        return list(self._files)

    def _scan_files(self) -> List[AttentionFile]:
        candidates: Iterable[Path] = sorted(self.data_dir.glob("*.npz"))
        files: List[AttentionFile] = []
        for candidate in candidates:
            with np.load(candidate, allow_pickle=True) as data:
                start_raw = data.get("start_id")
                end_raw = data.get("end_id")
                if start_raw is None or end_raw is None:
                    continue
                start_id = int(np.asarray(start_raw).item())
                end_id = int(np.asarray(end_raw).item())
            files.append(AttentionFile(candidate, start_id, end_id))

        files.sort(key=lambda item: (item.start_id, item.end_id, item.name))
        return files

    @lru_cache(maxsize=4)
    def _load_file(self, path: Path) -> dict:
        """Load and cache the full content of an NPZ file."""

        with np.load(path, allow_pickle=True) as data:
            attentions = data["attentions"]
            sources = data["sources"].tolist()
            predictions = data["predictions"].tolist()
            start_id = int(np.asarray(data["start_id"]).item())
            end_id = int(np.asarray(data["end_id"]).item())

        return {
            "attentions": attentions,
            "sources": sources,
            "predictions": predictions,
            "start_id": start_id,
            "end_id": end_id,
        }

    def get_sample(self, sample_id: int) -> SampleRecord:
        """Load a particular sample using its global identifier."""

        for file_meta in self._files:
            if sample_id in file_meta.sample_ids:
                payload = self._load_file(file_meta.path)
                batch_index = sample_id - payload["start_id"]
                if batch_index < 0 or batch_index >= len(payload["sources"]):
                    raise IndexError(
                        f"Sample id {sample_id} not found within file '{file_meta.name}'."
                    )

                attentions: np.ndarray = payload["attentions"][:, batch_index, :, :, :]
                return SampleRecord(
                    file=file_meta,
                    batch_index=batch_index,
                    source=str(payload["sources"][batch_index]),
                    prediction=str(payload["predictions"][batch_index]),
                    attentions=attentions,
                )

        raise KeyError(f"Sample id {sample_id} was not present in any NPZ file under '{self.data_dir}'.")

