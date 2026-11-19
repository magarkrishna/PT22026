"""Visual similarity search for the hybrid Food-101 classifier using FAISS.
Run with::
    python faiss_search.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageOps

from model_loader import REPO_ID, load_model

LOGGER = logging.getLogger("food_classifier.faiss_search")

DATASET_ID = "ethz/food101"
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_IMAGES = 500
DEFAULT_SEED = 42
DEFAULT_TOP_K = 3

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "eval_outputs"
INDEX_PATH = OUTPUT_DIR / "food_index.faiss"
METADATA_PATH = OUTPUT_DIR / "food_index_metadata.json"
IMAGE_DIR = OUTPUT_DIR / "food_index_images"

# The fusion module's global output is the richest representation in the
# model.  The CNN branch's pooled vector is used as a fallback if the fusion
# output is unavailable (for example with an older checkpoint).
PRIMARY_FEATURE_KEY = "fused_global"
FEATURE_KEYS = ("fused_global", "cnn_global", "vit_global")

# Reference images are stored at a small size; the gallery only needs
# thumbnails, and this keeps eval_outputs/ reasonably small.
THUMBNAIL_MAX_SIDE = 320
THUMBNAIL_QUALITY = 85


class IndexNotBuiltError(RuntimeError):
    """Raised when the FAISS index or its metadata has not been built yet."""


# Embedding helpers
def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a model tensor to a 2-D ``float32`` NumPy array."""
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 1:
        array = array[None, :]
    return np.ascontiguousarray(array, dtype=np.float32)


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row so L2 distance ranks by cosine similarity."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def embedding_from_output(
    output: Any, normalize: bool = True
) -> np.ndarray | None:
    """Extract an embedding matrix from a model output.

    Uses the fusion module's global vector when available, falling back to
    the CNN branch's pooled vector and then the ViT branch's CLS vector.
    Returns ``None`` if the model output contains no feature tensors.
    """
    if not isinstance(output, dict):
        return None

    for key in FEATURE_KEYS:
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            embeddings = _to_numpy(value)
            return normalize_rows(embeddings) if normalize else embeddings
    return None


def _prepare_image(image: Any) -> Image.Image:
    """Return an EXIF-corrected RGB ``PIL.Image`` from a dataset sample."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return ImageOps.exif_transpose(image).convert("RGB")


def extract_embedding(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Return the embedding of a single image as a ``(1, dim)`` array."""
    if device is None:
        device = getattr(model, "device", torch.device("cpu"))

    rgb_image = _prepare_image(image)
    tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        try:
            output = model(tensor, return_features=True)
        except TypeError:  # model without the return_features argument
            output = model(tensor)

    embeddings = embedding_from_output(output, normalize=normalize)
    if embeddings is None:
        raise RuntimeError(
            "The model did not return feature embeddings. Expected one of "
            f"{', '.join(FEATURE_KEYS)} in its output."
        )
    return embeddings


def extract_dataset_embeddings(
    model: torch.nn.Module,
    transform,
    dataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, List[int]]:
    """Embed dataset samples in batches.

    Returns ``(embeddings, kept_indices)``.  Samples that cannot be read are
    skipped and logged, mirroring the behaviour of :mod:`evaluate`.
    """
    all_embeddings: List[np.ndarray] = []
    kept_indices: List[int] = []
    total = len(indices)
    processed = 0
    next_report = 100

    for start in range(0, total, batch_size):
        chunk = list(indices[start : start + batch_size])
        tensors: List[torch.Tensor] = []
        chunk_indices: List[int] = []

        for dataset_index in chunk:
            try:
                item = dataset[int(dataset_index)]
                image = _prepare_image(item["image"])
                tensors.append(transform(image))
                chunk_indices.append(int(dataset_index))
            except Exception as exc:  # noqa: BLE001 - skip corrupt samples
                LOGGER.warning(
                    "Skipping dataset index %s: %s", dataset_index, exc
                )

        if tensors:
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                try:
                    output = model(batch, return_features=True)
                except TypeError:
                    output = model(batch)

            embeddings = embedding_from_output(output)
            if embeddings is None:
                raise RuntimeError(
                    "The model did not return feature embeddings. Expected "
                    f"one of {', '.join(FEATURE_KEYS)} in its output."
                )
            if embeddings.shape[0] != len(tensors):
                raise RuntimeError(
                    f"Got {embeddings.shape[0]} embeddings for "
                    f"{len(tensors)} images."
                )
            all_embeddings.append(embeddings)
            kept_indices.extend(chunk_indices)

        processed += len(chunk)
        if processed >= next_report or processed == total:
            print(f"Embedded {processed}/{total} images", flush=True)
            while next_report <= processed:
                next_report += 100

    if not all_embeddings:
        raise RuntimeError(
            "No embeddings could be extracted from the dataset. The dataset "
            "may be unreadable or all samples were skipped."
        )

    return np.ascontiguousarray(
        np.concatenate(all_embeddings, axis=0), dtype=np.float32
    ), kept_indices

# Dataset / metadata helpers
def get_dataset_label_names(dataset) -> List[str] | None:
    """Return the dataset's ordered class names, if available."""
    try:
        names = list(dataset.features["label"].names)
        return names or None
    except Exception:  # noqa: BLE001 - not a ClassLabel feature
        return None


def sample_balanced_indices(
    labels: Sequence[int],
    max_images: int,
    seed: int = DEFAULT_SEED,
) -> List[int]:
    """Pick up to ``max_images`` rows spread evenly across all classes.

    Food-101 is ordered by class, so ``range(500)`` would cover only two
    categories.  A class-balanced sample keeps the index useful for
    class-restricted similarity search without embedding the whole split.
    """
    total = len(labels)
    if max_images <= 0 or max_images >= total:
        return list(range(total))

    by_class: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)

    per_class = max(1, max_images // len(by_class))
    selected: List[int] = []
    for class_id in sorted(by_class):
        selected.extend(by_class[class_id][:per_class])

    remainder = max_images - len(selected)
    if remainder > 0:
        extras: List[int] = []
        for class_id in sorted(by_class):
            extras.extend(by_class[class_id][per_class:])
        rng.shuffle(extras)
        selected.extend(extras[:remainder])

    return sorted(selected[:max_images])


def _relative_to_script(path: Path) -> str:
    """Store paths relative to this file so the project stays portable."""
    try:
        return path.resolve().relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_image_path(path: str | Path) -> Path:
    """Resolve a stored (possibly relative) reference-image path."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = SCRIPT_DIR / resolved
    return resolved.resolve()


def save_reference_images(
    dataset,
    dataset_indices: Sequence[int],
    image_dir: Path = IMAGE_DIR,
) -> List[str]:
    """Save small JPEG copies of the indexed images and return their paths."""
    image_dir.mkdir(parents=True, exist_ok=True)
    stored_paths: List[str] = []

    for order, dataset_index in enumerate(dataset_indices):
        target = image_dir / f"{order:06d}.jpg"
        try:
            # Always overwrite: a rebuild can map the same row number to a
            # different image when the sampling changes.
            item = dataset[int(dataset_index)]
            image = _prepare_image(item["image"]).copy()
            image.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE))
            image.save(
                target, format="JPEG", quality=THUMBNAIL_QUALITY
            )
            stored_paths.append(_relative_to_script(target))
        except Exception as exc:  # noqa: BLE001 - keep the index usable
            LOGGER.warning(
                "Could not save reference image for dataset index %s: %s",
                dataset_index,
                exc,
            )
            stored_paths.append("")

    return stored_paths


def build_metadata(
    dataset,
    kept_indices: Sequence[int],
    image_paths: Sequence[str],
    idx_to_label: Dict[int, str],
    split: str,
    dimension: int,
    num_embeddings: int,
    sample_mode: str = "stratified",
    max_images: int | None = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Build the JSON metadata that maps FAISS rows to paths and labels.

    Each record carries a ``label`` field with the image's Food-101 class
    name.  :class:`ClassIndex` uses it to restrict searches to one category.
    """
    label_names = get_dataset_label_names(dataset)
    dataset_labels = dataset["label"]
    name_to_model_index = {
        name: index for index, name in idx_to_label.items()
    }

    if label_names is None:
        LOGGER.warning(
            "Dataset label names are unavailable; using the model's class "
            "mapping for reference labels."
        )

    images: List[Dict[str, Any]] = []
    for row, dataset_index in enumerate(kept_indices):
        try:
            dataset_label = int(dataset_labels[int(dataset_index)])
        except Exception:  # noqa: BLE001 - defensive fallback
            dataset_label = -1

        if label_names and 0 <= dataset_label < len(label_names):
            label_name = label_names[dataset_label]
            model_label = name_to_model_index.get(label_name)
        else:
            label_name = idx_to_label.get(dataset_label, f"class_{dataset_label}")
            model_label = dataset_label if dataset_label >= 0 else None

        images.append(
            {
                "row": row,
                "path": image_paths[row] if row < len(image_paths) else "",
                # Food-101 class label; used for class-restricted search.
                "label": label_name,
                "label_index": model_label,
                "dataset_index": int(dataset_index),
            }
        )

    class_counts: Dict[str, int] = {}
    for record in images:
        label = str(record.get("label") or "")
        if label:
            class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "dataset": DATASET_ID,
        "split": split,
        "sample_mode": sample_mode,
        "max_images": int(max_images) if max_images else None,
        "seed": int(seed),
        "model_repo": REPO_ID,
        "feature": PRIMARY_FEATURE_KEY,
        "dimension": int(dimension),
        "metric": "L2 on L2-normalised embeddings (equivalent to cosine)",
        "normalized": True,
        "count": int(num_embeddings),
        "num_classes": len(class_counts),
        "class_counts": class_counts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images": images,
    }

# Index build / load / search
def build_index(
    split: str = DEFAULT_SPLIT,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    max_images: int = DEFAULT_MAX_IMAGES,
    seed: int = DEFAULT_SEED,
    sample_mode: str = "stratified",
    index_path: Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
    image_dir: Path = IMAGE_DIR,
) -> Tuple[Any, Dict[str, Any]]:
    """Build and persist the FAISS index plus its JSON metadata."""
    LOGGER.info("Loading model from %s ...", REPO_ID)
    model, transform, idx_to_label = load_model(device=device)
    resolved_device = getattr(model, "device", torch.device("cpu"))
    LOGGER.info("Model loaded on %s.", resolved_device)

    LOGGER.info("Loading %s split '%s' ...", DATASET_ID, split)
    try:
        dataset = load_dataset(DATASET_ID, split=split)
    except Exception as exc:  # noqa: BLE001 - friendly console message
        raise RuntimeError(
            f"Could not load {DATASET_ID} split '{split}'. Check your internet "
            "connection and the `datasets` installation."
        ) from exc

    if sample_mode == "head":
        indices = list(range(min(max_images, len(dataset))))
        LOGGER.warning(
            "Using the first %d rows; Food-101 is ordered by class, so the "
            "index will cover very few categories.",
            len(indices),
        )
    else:
        indices = sample_balanced_indices(
            dataset["label"], max_images, seed
        )

    LOGGER.info(
        "Embedding %d image(s) from split '%s' (%s sampling) ...",
        len(indices),
        split,
        sample_mode,
    )

    embeddings, kept_indices = extract_dataset_embeddings(
        model=model,
        transform=transform,
        dataset=dataset,
        indices=indices,
        device=resolved_device,
        batch_size=max(1, batch_size),
    )

    if len(kept_indices) != embeddings.shape[0]:
        raise RuntimeError(
            "Embedding count does not match the number of indexed samples."
        )

    LOGGER.info(
        "Saving %d reference images to %s ...", len(kept_indices), image_dir
    )
    image_paths = save_reference_images(dataset, kept_indices, image_dir)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    LOGGER.info(
        "Built FAISS index: %d vectors, %d dimensions.",
        index.ntotal,
        index.d,
    )

    metadata = build_metadata(
        dataset=dataset,
        kept_indices=kept_indices,
        image_paths=image_paths,
        idx_to_label=idx_to_label,
        split=split,
        dimension=embeddings.shape[1],
        num_embeddings=index.ntotal,
        sample_mode=sample_mode,
        max_images=max_images,
        seed=seed,
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved %s", index_path)
    LOGGER.info("Saved %s", metadata_path)

    return index, metadata


def index_exists(
    index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH
) -> bool:
    """Return True if both the index file and its metadata exist."""
    return Path(index_path).is_file() and Path(metadata_path).is_file()


def load_index(
    index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Load the FAISS index and the image metadata.

    Raises :class:`IndexNotBuiltError` with a helpful message when the index
    has not been generated yet, so the UI can explain how to build it.
    """
    if not index_exists(index_path, metadata_path):
        raise IndexNotBuiltError(
            "Visual similarity index not found. Build it once with "
            "`python faiss_search.py` (writes eval_outputs/food_index.faiss)."
        )

    index = faiss.read_index(str(index_path))
    try:
        data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexNotBuiltError(
            f"Could not read FAISS metadata from {metadata_path}: {exc}"
        ) from exc

    images = data.get("images", []) if isinstance(data, dict) else data
    if not isinstance(images, list) or not images:
        raise IndexNotBuiltError(
            f"FAISS metadata in {metadata_path} contains no image records."
        )
    if index.ntotal != len(images):
        raise IndexNotBuiltError(
            f"FAISS index has {index.ntotal} vectors but the metadata lists "
            f"{len(images)} images. Rebuild with `python faiss_search.py`."
        )

    return index, images


def _class_key(label: Any) -> str:
    """Normalise a class label so lookups are case/space insensitive."""
    return str(label or "").strip().lower().replace(" ", "_")


def _result_record(
    record: Dict[str, Any], distance: float
) -> Dict[str, Any]:
    """Attach distance, cosine-equivalent similarity, and an absolute path."""
    result = dict(record)
    squared_l2 = float(distance)
    result["distance"] = squared_l2
    # For unit vectors, ||a - b||^2 = 2 - 2*cos(a, b).
    result["similarity"] = float(max(-1.0, min(1.0, 1.0 - squared_l2 / 2.0)))
    result["path"] = str(resolve_image_path(result.get("path", "")))
    return result


class ClassIndex:
    """Class-restricted search over an existing flat FAISS index.

    ``IndexFlatL2`` has no metadata filter, so this wrapper keeps a
    ``{class label: [row ids]}`` dictionary and lazily builds one small
    ``IndexFlatL2`` per class from the parent index's vectors.  A query for
    ``"pizza"`` therefore only compares against indexed pizza images, never
    against visually similar textures from another category.
    """

    def __init__(self, index, images: Sequence[Dict[str, Any]]) -> None:
        self.index = index
        self.images = list(images)
        self._rows: Dict[str, List[int]] = {}
        self._class_indexes: Dict[str, Any] = {}
        self._vectors: np.ndarray | None = None
        self._lock = threading.Lock()

        for row, record in enumerate(self.images):
            label = _class_key(record.get("label"))
            if label:
                self._rows.setdefault(label, []).append(row)

    def classes(self) -> List[str]:
        """Return the class labels present in the index."""
        return sorted(self._rows)

    def has_class(self, label: str) -> bool:
        """Return True if the index contains at least one image of ``label``."""
        return _class_key(label) in self._rows

    def count(self, label: str) -> int:
        """Return the number of indexed images for ``label``."""
        return len(self._rows.get(_class_key(label), []))

    def _parent_vectors(self) -> np.ndarray:
        if self._vectors is None:
            try:
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception:  # noqa: BLE001 - some wrappers lack reconstruct_n
                vectors = np.stack(
                    [
                        self.index.reconstruct(i)
                        for i in range(int(self.index.ntotal))
                    ]
                )
            self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        return self._vectors

    def _class_index(self, key: str):
        """Return the cached sub-index for one normalised class label."""
        class_index = self._class_indexes.get(key)
        if class_index is not None:
            return class_index

        rows = self._rows[key]
        vectors = self._parent_vectors()[rows]
        class_index = faiss.IndexFlatL2(int(self.index.d))
        class_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._class_indexes[key] = class_index
        return class_index

    def search(
        self,
        embedding: np.ndarray,
        label: str,
        k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Search only among reference images whose label matches ``label``.

        Returns an empty list when the class has no indexed images.
        """
        key = _class_key(label)
        rows = self._rows.get(key)
        if not rows:
            return []

        query = normalize_rows(np.asarray(embedding, dtype=np.float32))
        if query.shape[1] != int(self.index.d):
            raise RuntimeError(
                f"Query embedding has {query.shape[1]} dimensions but the "
                f"FAISS index expects {self.index.d}. Rebuild the index with "
                "the same model."
            )

        with self._lock:
            class_index = self._class_index(key)

        k = max(1, min(int(k), len(rows)))
        distances, neighbours = class_index.search(query, k)

        results: List[Dict[str, Any]] = []
        for distance, neighbour in zip(distances[0], neighbours[0]):
            neighbour = int(neighbour)
            if neighbour < 0 or neighbour >= len(rows):
                continue
            results.append(
                _result_record(self.images[rows[neighbour]], distance)
            )
        return results


def search_similar(
    index,
    embedding: np.ndarray,
    images: Sequence[Dict[str, Any]],
    k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Return the ``k`` nearest reference images for one embedding.

    ``embedding`` may be ``(dim,)`` or ``(1, dim)``; it is L2-normalised to
    match the indexed vectors.  Each result is a copy of the metadata record
    with ``distance``, ``similarity``, and an absolute ``path`` added.

    This function searches the whole index.  Use :class:`ClassIndex` to
    restrict results to the predicted class.
    """
    query = normalize_rows(np.asarray(embedding, dtype=np.float32))
    if query.shape[1] != index.d:
        raise RuntimeError(
            f"Query embedding has {query.shape[1]} dimensions but the FAISS "
            f"index expects {index.d}. Rebuild the index with the same model."
        )

    k = max(1, min(int(k), int(index.ntotal)))
    distances, neighbours = index.search(query, k)

    results: List[Dict[str, Any]] = []
    for distance, neighbour in zip(distances[0], neighbours[0]):
        neighbour = int(neighbour)
        if neighbour < 0 or neighbour >= len(images):
            continue
        results.append(_result_record(images[neighbour], distance))

    return results
# CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the FAISS visual-similarity index for Food-101."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to index (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=(
            "Number of reference images to index "
            f"(default: {DEFAULT_MAX_IMAGES})."
        ),
    )
    parser.add_argument(
        "--sample-mode",
        choices=("stratified", "head"),
        default="stratified",
        help=(
            "How to subsample: 'stratified' (default, class-balanced) or "
            "'head' (first N rows, which covers only a few classes)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Sampling seed."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Embedding batch size."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. cpu, cuda, mps (default: auto).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        index, metadata = build_index(
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            max_images=args.max_images,
            seed=args.seed,
            sample_mode=args.sample_mode,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - friendly console error
        LOGGER.exception("Failed to build the FAISS index")
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print(f"Indexed images : {metadata['count']}")
    print(f"Classes covered: {metadata['num_classes']}")
    print(f"Dimensions     : {metadata['dimension']}")
    print(f"Feature        : {metadata['feature']}")
    print(f"Index file     : {INDEX_PATH}")
    print(f"Metadata file  : {METADATA_PATH}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
