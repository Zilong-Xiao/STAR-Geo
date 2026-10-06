from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


TOPOLOGY_LABELS: Tuple[str, ...] = (
    "Straight",
    "T-junction",
    "Y-junction",
    "Cross",
    "Complex",
)
SCALE_LABELS: Tuple[str, ...] = (
    "Alley",
    "Narrow",
    "Standard",
    "Broad",
    "Massive",
)
FUNCTION_LABELS: Tuple[str, ...] = (
    "Local",
    "Secondary",
    "Primary",
    "Highway",
)

_VIEW_ALIASES = {
    "ground": "ground",
    "query": "ground",
    "street": "ground",
    "streetview": "ground",
    "street-view": "ground",
    "panorama": "ground",
    "pano": "ground",
    "aerial": "aerial",
    "satellite": "aerial",
    "reference": "aerial",
    "sat": "aerial",
}


def _label_key(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _build_label_aliases(labels: Sequence[str], extras: Mapping[str, str]) -> Dict[str, str]:
    aliases = {_label_key(label): label for label in labels}
    aliases.update({_label_key(src): dst for src, dst in extras.items()})
    return aliases


_TOPOLOGY_ALIASES = _build_label_aliases(
    TOPOLOGY_LABELS,
    {
        "linear": "Straight",
        "no junction": "Straight",
        "tjunction": "T-junction",
        "t intersection": "T-junction",
        "yjunction": "Y-junction",
        "y intersection": "Y-junction",
        "crossjunction": "Cross",
        "cross intersection": "Cross",
        "four way": "Cross",
        "four-way": "Cross",
        "roundabout": "Complex",
    },
)
_SCALE_ALIASES = _build_label_aliases(SCALE_LABELS, {})
_FUNCTION_ALIASES = _build_label_aliases(FUNCTION_LABELS, {})


def _canonical_label(value: Any, factor: str) -> str:
    aliases = {
        "topology": _TOPOLOGY_ALIASES,
        "scale": _SCALE_ALIASES,
        "function": _FUNCTION_ALIASES,
    }[factor]
    key = _label_key(value)
    if key not in aliases:
        allowed = {
            "topology": TOPOLOGY_LABELS,
            "scale": SCALE_LABELS,
            "function": FUNCTION_LABELS,
        }[factor]
        raise ValueError(f"Unknown {factor} label {value!r}. Expected one of {list(allowed)}.")
    return aliases[key]


def _canonical_view(value: Any) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if key not in _VIEW_ALIASES:
        raise ValueError(f"Unknown TSF view {value!r}; expected ground/query or aerial/satellite.")
    return _VIEW_ALIASES[key]


def _normalize_filename(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


@dataclass(frozen=True)
class TSFRecord:
    city: str
    filename: str
    view: str
    topology: str
    scale: str
    function: str

    @property
    def ids(self) -> Tuple[int, int, int]:
        return (
            TOPOLOGY_LABELS.index(self.topology),
            SCALE_LABELS.index(self.scale),
            FUNCTION_LABELS.index(self.function),
        )


class TSFAnnotationStore:
    """Loads independent ground and aerial TSF annotations.

    The preferred release format is JSONL with one record per image::

        {"city":"Chicago", "filename":"...jpg", "view":"ground",
         "topology":"Straight", "scale":"Standard", "function":"Local"}

    A JSON object containing a ``records`` list, nested ``ground``/``aerial``
    mappings, or city-level mappings is also accepted. Legacy city-to-list files
    without an explicit view are interpreted as aerial-only records and will
    fail validation if ground annotations are required.
    """

    def __init__(self, annotation_path: str | Path):
        self.annotation_path = Path(annotation_path)
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"TSF annotation file not found: {self.annotation_path}")

        self._records: Dict[Tuple[str, str, str], TSFRecord] = {}
        self._basename_records: Dict[Tuple[str, str, str], List[TSFRecord]] = defaultdict(list)
        self._global_records: Dict[Tuple[str, str], List[TSFRecord]] = defaultdict(list)
        self._global_basename_records: Dict[Tuple[str, str], List[TSFRecord]] = defaultdict(list)
        for record in self._read_records(self.annotation_path):
            self._insert(record)

        if not self._records:
            raise ValueError(f"No valid TSF records were found in {self.annotation_path}")

    @property
    def topology_labels(self) -> Tuple[str, ...]:
        return TOPOLOGY_LABELS

    @property
    def scale_labels(self) -> Tuple[str, ...]:
        return SCALE_LABELS

    @property
    def function_labels(self) -> Tuple[str, ...]:
        return FUNCTION_LABELS

    def _insert(self, record: TSFRecord) -> None:
        city = record.city.strip()
        filename = _normalize_filename(record.filename)
        normalized = TSFRecord(
            city=city,
            filename=filename,
            view=_canonical_view(record.view),
            topology=_canonical_label(record.topology, "topology"),
            scale=_canonical_label(record.scale, "scale"),
            function=_canonical_label(record.function, "function"),
        )
        key = (normalized.view, city.lower(), filename.lower())
        existing = self._records.get(key)
        if existing is not None and existing != normalized:
            raise ValueError(f"Conflicting TSF annotations for {key}: {existing} vs {normalized}")
        self._records[key] = normalized
        basename_key = (normalized.view, city.lower(), Path(filename).name.lower())
        self._basename_records[basename_key].append(normalized)
        self._global_records[(normalized.view, filename.lower())].append(normalized)
        self._global_basename_records[(normalized.view, Path(filename).name.lower())].append(normalized)

    @classmethod
    def _read_records(cls, path: Path) -> Iterator[TSFRecord]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                    yield cls._record_from_mapping(item, inherited={})
            return

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from cls._walk_payload(payload, inherited={})

    @classmethod
    def _walk_payload(cls, node: Any, inherited: Mapping[str, Any]) -> Iterator[TSFRecord]:
        if isinstance(node, list):
            if cls._looks_like_row(node):
                yield cls._record_from_row(node, inherited)
                return
            for item in node:
                yield from cls._walk_payload(item, inherited)
            return

        if not isinstance(node, Mapping):
            raise ValueError(f"Unsupported TSF annotation node: {type(node).__name__}")

        lower_keys = {str(key).lower(): key for key in node.keys()}
        if cls._looks_like_record(node):
            yield cls._record_from_mapping(node, inherited)
            return

        if "records" in lower_keys:
            yield from cls._walk_payload(node[lower_keys["records"]], inherited)
            return

        for key, value in node.items():
            key_text = str(key)
            key_lower = key_text.lower()
            next_inherited = dict(inherited)
            if key_lower in _VIEW_ALIASES:
                next_inherited["view"] = _VIEW_ALIASES[key_lower]
            elif key_lower in {"cities", "data", "annotations", "items"}:
                pass
            elif "city" not in next_inherited:
                next_inherited["city"] = key_text
            else:
                next_inherited["group"] = key_text
            yield from cls._walk_payload(value, next_inherited)

    @staticmethod
    def _looks_like_record(node: Mapping[str, Any]) -> bool:
        keys = {_label_key(key) for key in node.keys()}
        has_filename = bool(keys & {"filename", "file", "image", "imageid", "path", "sat", "ground"})
        return has_filename and {"topology", "scale", "function"}.issubset(keys)

    @staticmethod
    def _looks_like_row(node: Sequence[Any]) -> bool:
        return len(node) in {4, 5, 6} and not any(isinstance(item, (list, dict)) for item in node)

    @classmethod
    def _record_from_mapping(cls, item: Mapping[str, Any], inherited: Mapping[str, Any]) -> TSFRecord:
        normalized = {_label_key(key): value for key, value in item.items()}

        city = normalized.get("city", inherited.get("city", ""))
        view = normalized.get("view", normalized.get("imagetype", inherited.get("view")))

        filename = None
        for key in ("filename", "file", "image", "imageid", "path", "sat", "ground"):
            if key in normalized:
                filename = normalized[key]
                if view is None and key == "sat":
                    view = "aerial"
                elif view is None and key == "ground":
                    view = "ground"
                break

        if filename is None:
            raise ValueError(f"TSF record is missing filename: {item}")
        if view is None:
            # Legacy city -> [sat_filename, T, S, F] files contain aerial labels only.
            view = "aerial"

        return TSFRecord(
            city=str(city),
            filename=str(filename),
            view=str(view),
            topology=str(normalized["topology"]),
            scale=str(normalized["scale"]),
            function=str(normalized["function"]),
        )

    @classmethod
    def _record_from_row(cls, row: Sequence[Any], inherited: Mapping[str, Any]) -> TSFRecord:
        city = inherited.get("city")
        view = inherited.get("view")
        if len(row) == 4:
            filename, topology, scale, function = row
        elif len(row) == 5:
            # Prefer [filename, view, T, S, F]; otherwise [city, filename, T, S, F].
            if str(row[1]).strip().lower() in _VIEW_ALIASES:
                filename, view, topology, scale, function = row
            else:
                city, filename, topology, scale, function = row
        else:
            city, filename, view, topology, scale, function = row

        if city is None:
            city = ""
        if view is None:
            view = "aerial"
        return TSFRecord(
            city=str(city),
            filename=str(filename),
            view=str(view),
            topology=str(topology),
            scale=str(scale),
            function=str(function),
        )

    def get(self, city: str, filename: str, view: str) -> TSFRecord:
        canonical_view = _canonical_view(view)
        normalized_filename = _normalize_filename(filename)
        city_key = city.strip().lower()

        for candidate_city in (city_key, ""):
            direct_key = (canonical_view, candidate_city, normalized_filename.lower())
            record = self._records.get(direct_key)
            if record is not None:
                return record

            basename_key = (canonical_view, candidate_city, Path(normalized_filename).name.lower())
            candidates = self._basename_records.get(basename_key, [])
            unique = {(candidate.filename, candidate.ids): candidate for candidate in candidates}
            if len(unique) == 1:
                return next(iter(unique.values()))
            if len(unique) > 1:
                raise KeyError(
                    f"Ambiguous TSF annotation for {canonical_view}/{city}/{filename}; "
                    f"matched {[candidate.filename for candidate in unique.values()]}"
                )

        # Records in the paper's generic JSONL format may omit city. Use a
        # global fallback only when the filename identifies one annotation.
        global_candidates = self._global_records.get((canonical_view, normalized_filename.lower()), [])
        if not global_candidates:
            global_candidates = self._global_basename_records.get(
                (canonical_view, Path(normalized_filename).name.lower()), []
            )
        unique = {(candidate.city, candidate.filename, candidate.ids): candidate for candidate in global_candidates}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            raise KeyError(
                f"Ambiguous city-free TSF annotation for {canonical_view}/{filename}; "
                "include the city or a city-qualified relative filename in the annotation record"
            )
        raise KeyError(
            f"Missing independent {canonical_view} TSF annotation for city={city!r}, "
            f"filename={filename!r} in {self.annotation_path}"
        )

    def iter_records(self, view: Optional[str] = None, cities: Optional[Iterable[str]] = None) -> Iterator[TSFRecord]:
        canonical_view = _canonical_view(view) if view is not None else None
        city_set = {str(city).strip().lower() for city in cities} if cities is not None else None
        for record in self._records.values():
            if canonical_view is not None and record.view != canonical_view:
                continue
            if city_set is not None and record.city.strip().lower() not in city_set:
                continue
            yield record

    def validate_pairs(self, pairs: Iterable[Tuple[str, str, str]]) -> None:
        missing: List[str] = []
        for city, filename, view in pairs:
            try:
                self.get(city, filename, view)
            except KeyError as exc:
                missing.append(str(exc))
                if len(missing) >= 20:
                    break
        if missing:
            details = "\n".join(f"  - {message}" for message in missing)
            raise ValueError(
                "The annotation file does not provide the independent ground/aerial TSF inputs "
                "required by the paper. Examples:\n" + details
            )


class ImageTransform:
    """Resize, convert to CHW tensor, and normalize with the backbone statistics."""

    def __init__(self, size: int, mean: Sequence[float], std: Sequence[float], ground_cutting: int = 0):
        self.size = int(size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.ground_cutting = int(ground_cutting)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.ground_cutting > 0:
            cut = self.ground_cutting
            if image.shape[0] <= 2 * cut:
                raise ValueError(f"ground_cutting={cut} is too large for image height {image.shape[0]}")
            image = image[cut:-cut]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR_EXACT)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        tensor = torch.from_numpy(image).float().div_(255.0)
        return (tensor - self.mean) / self.std


def _resolve_image_path(root: Path, view: str, city: str, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    normalized = Path(_normalize_filename(filename))
    if normalized.parts and normalized.parts[0].lower() in {"ground", "satellite"}:
        return root / normalized
    directory = "ground" if _canonical_view(view) == "ground" else "satellite"
    return root / directory / city / normalized


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"VIGOR split file not found: {path}")
    frame = pd.read_csv(path, header=None, sep=r"\s+")
    required_columns = [0, 1, 4, 7, 10]
    if max(required_columns) >= frame.shape[1]:
        raise ValueError(f"Unexpected VIGOR split format in {path}: only {frame.shape[1]} columns")
    return frame.loc[:, required_columns].rename(
        columns={0: "ground", 1: "sat", 4: "sat_np1", 7: "sat_np2", 10: "sat_np3"}
    )


def _cities_for(setting: str, split: str) -> List[str]:
    setting = setting.lower()
    split = split.lower()
    if setting == "same":
        return ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
    if setting != "cross":
        raise ValueError("setting must be 'same' or 'cross'")
    return ["NewYork", "Seattle"] if split == "train" else ["Chicago", "SanFrancisco"]


def _split_path(root: Path, city: str, setting: str, split: str) -> Path:
    if setting == "same":
        return root / "splits" / city / f"same_area_balanced_{split}.txt"
    return root / "splits" / city / "pano_label_balanced.txt"


def _coerce_annotation_store(value: str | Path | TSFAnnotationStore) -> TSFAnnotationStore:
    return value if isinstance(value, TSFAnnotationStore) else TSFAnnotationStore(value)


class VIGORIndex:
    def __init__(self, data_root: str | Path, setting: str, split: str, annotations: TSFAnnotationStore):
        self.root = Path(data_root)
        self.setting = setting.lower()
        self.split = split.lower()
        self.cities = _cities_for(self.setting, self.split)
        self.annotations = annotations

        satellite_rows: List[Dict[str, Any]] = []
        satellite_key_to_index: Dict[Tuple[str, str], int] = {}
        satellite_basename_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        # Enumerate the actual aerial gallery on disk, then attach independently
        # generated aerial TSF records. This also supports JSONL records that omit
        # the city field, provided filenames are globally unambiguous.
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        for city in self.cities:
            city_directory = self.root / "satellite" / city
            if not city_directory.is_dir():
                raise FileNotFoundError(f"VIGOR satellite directory not found: {city_directory}")
            image_paths = sorted(
                path for path in city_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in image_extensions
            )
            if not image_paths:
                raise ValueError(f"No aerial images found in {city_directory}")
            for image_path in image_paths:
                filename = image_path.relative_to(city_directory).as_posix()
                annotation = annotations.get(city, filename, "aerial")
                index = len(satellite_rows)
                satellite_rows.append(
                    {
                        "city": city,
                        "filename": filename,
                        "path": image_path,
                        "tsf_ids": annotation.ids,
                    }
                )
                satellite_key_to_index[(city.lower(), filename.lower())] = index
                satellite_basename_to_indices[(city.lower(), Path(filename).name.lower())].append(index)

        if not satellite_rows:
            raise ValueError(f"No aerial images were found for cities {self.cities}")

        def satellite_index(city: str, filename: str) -> int:
            normalized = _normalize_filename(filename)
            direct = satellite_key_to_index.get((city.lower(), normalized.lower()))
            if direct is not None:
                return direct
            candidates = satellite_basename_to_indices.get((city.lower(), Path(normalized).name.lower()), [])
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise KeyError(f"Ambiguous satellite filename {city}/{filename} in TSF annotations")
            raise KeyError(f"Satellite {city}/{filename} from a VIGOR split has no aerial TSF record")

        ground_rows: List[Dict[str, Any]] = []
        for city in self.cities:
            split_frame = _read_split(_split_path(self.root, city, self.setting, self.split))
            for _, row in split_frame.iterrows():
                sat_names = [str(row[name]) for name in ("sat", "sat_np1", "sat_np2", "sat_np3")]
                sat_indices = tuple(satellite_index(city, name) for name in sat_names)
                ground_name = str(row["ground"])
                ground_tsf = annotations.get(city, ground_name, "ground")
                ground_rows.append(
                    {
                        "city": city,
                        "filename": ground_name,
                        "path": _resolve_image_path(self.root, "ground", city, ground_name),
                        "sat_indices": sat_indices,
                        "tsf_ids": ground_tsf.ids,
                    }
                )

        self.satellites = satellite_rows
        self.grounds = ground_rows


class VIGORTrainDataset(Dataset):
    """Matched ground-aerial training pairs with independent TSF tuples."""

    def __init__(
        self,
        data_root: str | Path,
        annotation_path: str | Path | TSFAnnotationStore,
        setting: str,
        query_transform: ImageTransform,
        reference_transform: ImageTransform,
        flip_probability: float = 0.5,
        rotation_probability: float = 0.75,
    ):
        self.annotations = _coerce_annotation_store(annotation_path)
        self.index = VIGORIndex(data_root, setting, "train", self.annotations)
        self.query_transform = query_transform
        self.reference_transform = reference_transform
        self.flip_probability = float(flip_probability)
        self.rotation_probability = float(rotation_probability)

        self.samples: List[Tuple[int, int]] = [
            (ground_index, int(ground["sat_indices"][0]))
            for ground_index, ground in enumerate(self.index.grounds)
        ]
        self.reference_ids: List[int] = [sat_index for _, sat_index in self.samples]

    @property
    def topology_labels(self) -> Tuple[str, ...]:
        return TOPOLOGY_LABELS

    @property
    def scale_labels(self) -> Tuple[str, ...]:
        return SCALE_LABELS

    @property
    def function_labels(self) -> Tuple[str, ...]:
        return FUNCTION_LABELS

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int):
        ground_index, satellite_index = self.samples[item]
        ground = self.index.grounds[ground_index]
        satellite = self.index.satellites[satellite_index]

        query = _read_rgb(ground["path"])
        reference = _read_rgb(satellite["path"])

        if random.random() < self.flip_probability:
            query = cv2.flip(query, 1)
            reference = cv2.flip(reference, 1)

        query_tensor = self.query_transform(query)
        reference_tensor = self.reference_transform(reference)

        if random.random() < self.rotation_probability:
            quarter_turns = random.choice((1, 2, 3))
            reference_tensor = torch.rot90(reference_tensor, k=quarter_turns, dims=(1, 2))
            shift = -(query_tensor.shape[2] // 4) * quarter_turns
            query_tensor = torch.roll(query_tensor, shifts=shift, dims=2)

        return (
            query_tensor,
            reference_tensor,
            torch.tensor(ground["tsf_ids"], dtype=torch.long),
            torch.tensor(satellite["tsf_ids"], dtype=torch.long),
            torch.tensor(satellite_index, dtype=torch.long),
        )


class VIGOREvalDataset(Dataset):
    """Query or gallery split used by the official VIGOR retrieval protocol."""

    def __init__(
        self,
        data_root: str | Path,
        annotation_path: str | Path | TSFAnnotationStore,
        setting: str,
        split: str,
        view: str,
        transform: ImageTransform,
        index: Optional[VIGORIndex] = None,
    ):
        self.annotations = _coerce_annotation_store(annotation_path)
        self.index = index or VIGORIndex(data_root, setting, split, self.annotations)
        self.view = _canonical_view(view)
        self.transform = transform

        if self.view == "aerial":
            if split == "train":
                used = sorted({int(item["sat_indices"][0]) for item in self.index.grounds})
                self.items = [self.index.satellites[index] for index in used]
                self.labels = used
            else:
                self.items = list(self.index.satellites)
                self.labels = list(range(len(self.index.satellites)))
        else:
            self.items = list(self.index.grounds)
            self.labels = [item["sat_indices"] for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        record = self.items[item]
        image = self.transform(_read_rgb(record["path"]))
        tsf_ids = torch.tensor(record["tsf_ids"], dtype=torch.long)
        label = torch.tensor(self.labels[item], dtype=torch.long)
        return image, tsf_ids, label


class UniqueReferenceBatchSampler(Sampler[List[int]]):
    """Forms batches with at most one ground image for each strict aerial target.

    The symmetric in-batch contrastive objective assumes one positive per row and
    column. VIGOR can contain multiple ground panoramas for an aerial tile, so a
    plain random sampler can create false negatives. This sampler uses every
    training pair while preventing duplicate reference IDs within a batch.
    """

    def __init__(
        self,
        reference_ids: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 1,
    ):
        if batch_size <= 1:
            raise ValueError("batch_size must be greater than one for contrastive training")
        self.reference_ids = list(map(int, reference_ids))
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: MutableMapping[int, List[int]] = defaultdict(list)
        for sample_index, reference_id in enumerate(self.reference_ids):
            buckets[reference_id].append(sample_index)
        for bucket in buckets.values():
            rng.shuffle(bucket)

        pending: List[int] = []
        pending_references = set()
        while buckets:
            reference_order = list(buckets.keys())
            rng.shuffle(reference_order)
            added = False
            for reference_id in reference_order:
                if reference_id in pending_references:
                    continue
                pending.append(buckets[reference_id].pop())
                pending_references.add(reference_id)
                added = True
                if not buckets[reference_id]:
                    del buckets[reference_id]
                if len(pending) == self.batch_size:
                    yield pending
                    pending = []
                    pending_references = set()

            if not added and pending:
                # Every remaining bucket duplicates an item already in the partial
                # batch. Close the batch before continuing to preserve uniqueness.
                if not self.drop_last:
                    yield pending
                pending = []
                pending_references = set()

        if pending and not self.drop_last:
            yield pending

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.reference_ids) // self.batch_size
        return math.ceil(len(self.reference_ids) / self.batch_size)
