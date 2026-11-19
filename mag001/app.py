"""Gradio application for the hybrid Food-101 image classifier.
Run with::
    python app.py
"""

from __future__ import annotations

import html
import logging
import os
import threading
from typing import Any, Dict, List, Tuple

import gradio as gr
import torch
from PIL import Image, ImageOps

from food_info import FOOD_INFO
from model_loader import load_model

try:
    import faiss_search
except Exception as exc:  # noqa: BLE001 - keep the app usable without FAISS
    faiss_search = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR: str | None = str(exc)
else:
    _FAISS_IMPORT_ERROR = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("food_classifier.app")

# Lazy, thread-safe model loading so the UI still starts if the download
# fails; the error is then shown to the user on the first prediction.
_MODEL: Any = None
_TRANSFORM: Any = None
_IDX_TO_LABEL: Dict[int, str] | None = None
_LOAD_ERROR: str | None = None
_LOAD_LOCK = threading.Lock()

# The FAISS index is loaded lazily and cached after the first search.  It is
# optional: when it is missing the app still classifies images normally.
_FAISS_INDEX: Any = None
_FAISS_IMAGES: List[Dict[str, Any]] | None = None
_FAISS_CLASS_INDEX: Any = None
_FAISS_ERROR: str | None = None
_FAISS_LOCK = threading.Lock()

_READY_PREDICTED = (
    "### 🍽️ Predicted Food\n\n"
    "Upload or capture an image, then press **Analyze Food**."
)
_EMPTY_NUTRITION = (
    "### 🔥 Calories\n\n—\n\n"
    "### 🧂 Ingredients\n\n—\n\n"
    "### 📖 Description\n\n—"
)
_EMPTY_SIMILAR_STATUS = "_Run **Analyze Food** to see visually similar dishes._"

# Minimal styling only: keep the Label compact without overriding theme colours.
CUSTOM_CSS = """
.compact-label,
.compact-label .label-container {
    min-height: 0 !important;
}

.compact-label {
    padding: 0.25rem !important;
}

.compact-gallery {
    min-height: 0 !important;
}
"""


def _get_model():
    """Load the model once and return ``(model, transform, idx_to_label)``."""
    global _MODEL, _TRANSFORM, _IDX_TO_LABEL, _LOAD_ERROR

    if _MODEL is not None:
        return _MODEL, _TRANSFORM, _IDX_TO_LABEL

    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL, _TRANSFORM, _IDX_TO_LABEL

        try:
            LOGGER.info("Loading Food-101 hybrid model (first request) ...")
            requested_device = os.getenv("FOOD_CLASSIFIER_DEVICE") or None
            _MODEL, _TRANSFORM, _IDX_TO_LABEL = load_model(device=requested_device)
            _LOAD_ERROR = None
        except Exception as exc:  # noqa: BLE001 - show a friendly UI error
            _LOAD_ERROR = str(exc)
            LOGGER.exception("Model loading failed")
            raise RuntimeError(
                "The classification model could not be loaded. "
                "Please check your internet connection and try again."
            ) from exc

    return _MODEL, _TRANSFORM, _IDX_TO_LABEL


def _get_faiss_index() -> Tuple[Any, List[Dict[str, Any]], Any]:
    """Load the FAISS index, metadata, and class-restricted wrapper once.

    Raises an exception with a user-readable message when the index has not
    been built yet (``python faiss_search.py``) or FAISS is not installed.
    """
    global _FAISS_INDEX, _FAISS_IMAGES, _FAISS_CLASS_INDEX, _FAISS_ERROR

    if _FAISS_CLASS_INDEX is not None:
        return _FAISS_INDEX, _FAISS_IMAGES, _FAISS_CLASS_INDEX

    if faiss_search is None:
        raise RuntimeError(
            "Visual similarity search is unavailable because FAISS could not "
            f"be imported ({_FAISS_IMPORT_ERROR or 'unknown error'}). "
            "Install it with `pip install faiss-cpu`."
        )

    with _FAISS_LOCK:
        if _FAISS_CLASS_INDEX is not None:
            return _FAISS_INDEX, _FAISS_IMAGES, _FAISS_CLASS_INDEX

        try:
            LOGGER.info("Loading FAISS similarity index (first search) ...")
            index, records = faiss_search.load_index()
            _FAISS_INDEX = index
            _FAISS_IMAGES = records
            # Per-class sub-indexes are built lazily on the first lookup.
            _FAISS_CLASS_INDEX = faiss_search.ClassIndex(index, records)
            _FAISS_ERROR = None
        except Exception as exc:  # noqa: BLE001 - shown in the UI instead
            _FAISS_ERROR = str(exc)
            LOGGER.warning("FAISS index unavailable: %s", exc)
            raise

    return _FAISS_INDEX, _FAISS_IMAGES, _FAISS_CLASS_INDEX


def _pretty(name: str) -> str:
    """Turn ``fish_and_chips`` into ``Fish And Chips``."""
    return name.replace("_", " ").title()


def _message_outputs(
    message: str,
    similar_status: str | None = None,
) -> Tuple[Dict[str, float], str, str, List[Tuple[str, str]], str]:
    """Return an empty prediction plus a readable message in each panel."""
    return (
        {},
        f"### 🍽️ Predicted Food\n\n{message}",
        _EMPTY_NUTRITION,
        [],
        similar_status or _EMPTY_SIMILAR_STATUS,
    )


def _calorie_badge(calories: int | float) -> str:
    """Return a colour-coded calorie badge for the nutrition panel."""
    if calories < 200:
        return f"🟢 Low ({calories} kcal)"
    if calories <= 400:
        return f"🟡 Moderate ({calories} kcal)"
    return f"🔴 High ({calories} kcal)"


def _format_nutrition(info: Dict | None) -> str:
    """Build one combined Markdown block for the nutrition column."""
    if info is None:
        unavailable = "> **Nutrition information not available** for this category."
        return (
            "### 🔥 Calories\n\n"
            f"{unavailable}\n\n"
            "### 🧂 Ingredients\n\n"
            f"{unavailable}\n\n"
            "### 📖 Description\n\n"
            f"{unavailable}"
        )

    calories = info.get("calories_per_100g")
    calories_text = (
        "Nutrition information not available."
        if calories is None
        else _calorie_badge(calories)
    )

    ingredients = info.get("ingredients") or []
    ingredients_text = (
        ", ".join(html.escape(str(item)) for item in ingredients)
        if ingredients
        else "Not available"
    )

    description = info.get("description") or "No description available."

    return (
        "### 🔥 Calories\n\n"
        f"{calories_text}\n\n"
        "### 🧂 Ingredients\n\n"
        f"{ingredients_text}\n\n"
        "### 📖 Description\n\n"
        f"{html.escape(str(description))}"
    )


def _format_predicted(predicted: str, confidence: float) -> str:
    """Build the prominent predicted-food panel."""
    warning = ""
    if confidence < 0.30:
        warning = (
            "\n\n> ⚠️ **Low confidence.** The image may not contain a "
            "Food-101 category."
        )
    return (
        "### 🍽️ Predicted Food\n\n"
        f"## {html.escape(_pretty(predicted))}\n\n"
        f"**Confidence:** {confidence * 100:.2f}%"
        f"{warning}"
    )


def _find_similar_images(
    model: Any,
    image: Image.Image,
    transform: Any,
    device: torch.device,
    output: Any,
    predicted_label: str,
    k: int = 3,
) -> Tuple[List[Tuple[str, str]], str]:
    """Search the FAISS index among images of the predicted class only.

    Returns ``(gallery_items, status_message)``.  Failures are reported in the
    status message instead of raising, so a missing index never breaks
    classification.  When the predicted class has no indexed references the
    gallery is an empty list.
    """
    if faiss_search is None:
        return [], (
            "_Visual similarity search is unavailable because FAISS could not "
            f"be imported._ `{html.escape(_FAISS_IMPORT_ERROR or '')}`"
        )

    try:
        _, _, class_index = _get_faiss_index()
    except Exception as exc:  # noqa: BLE001 - index missing or unreadable
        return [], (
            "_Visual similarity index not built yet._ "
            f"`{html.escape(str(exc))}`"
        )

    if not class_index.has_class(predicted_label):
        return [], (
            f"_No reference images for **{html.escape(_pretty(predicted_label))}** "
            "in the similarity index._"
        )

    try:
        # Reuse the forward pass already computed for classification when it
        # contains feature tensors; otherwise run a dedicated embedding pass.
        embedding = faiss_search.embedding_from_output(output)
        if embedding is None:
            embedding = faiss_search.extract_embedding(
                model, image, transform, device
            )
        results = class_index.search(
            embedding, predicted_label, k=k
        )
    except Exception as exc:  # noqa: BLE001 - never break the UI
        LOGGER.exception("FAISS similarity search failed")
        return [], (
            "_Visual similarity search failed:_ "
            f"`{html.escape(type(exc).__name__)}: {html.escape(str(exc))}`"
        )

    gallery: List[Tuple[str, str]] = []
    for record in results:
        path = str(record.get("path") or "")
        if not path or not os.path.isfile(path):
            continue
        label = _pretty(str(record.get("label") or "")) or "Similar dish"
        similarity = record.get("similarity")
        if isinstance(similarity, (int, float)):
            caption = f"{label} · {float(similarity) * 100:.0f}% match"
        else:
            caption = label
        gallery.append((path, caption))

    if not gallery:
        return [], (
            f"_No similar images found for "
            f"**{html.escape(_pretty(predicted_label))}**._"
        )

    return gallery, (
        f"_Showing the top {len(gallery)} nearest "
        f"**{html.escape(_pretty(predicted_label))}** images from the FAISS "
        "index._"
    )


def predict(
    image: Image.Image | None,
) -> Tuple[Dict[str, float], str, str, List[Tuple[str, str]], str]:
    """Classify an image and return the UI output values.

    Returns
    top3
        Mapping of pretty food names to probabilities for ``gr.Label``.
    predicted
        Markdown for the prominent predicted-food panel.
    nutrition
        Combined Markdown block with calories, ingredients, and description.
    similar_gallery
        ``(image_path, caption)`` pairs for the same-class FAISS matches.
    similar_status
        Markdown status line for the similarity panel.

    This function is deliberately defensive: every failure mode returns a
    readable message instead of raising and breaking the Gradio event.
    """
    if image is None:
        return _message_outputs(
            "Please upload an image or take a webcam photo first."
        )

    if not isinstance(image, Image.Image):
        return _message_outputs(
            "The input could not be interpreted as an image. Please try a "
            "PNG, JPEG, or WebP file."
        )

    try:
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:  # noqa: BLE001 - corrupt/truncated image
        LOGGER.exception("Could not convert image to RGB")
        return _message_outputs(
            "The image file appears to be corrupted or unreadable."
        )

    try:
        model, transform, idx_to_label = _get_model()
    except Exception as exc:  # noqa: BLE001
        detail = (
            f"\n\n**Details:** `{html.escape(_LOAD_ERROR)}`"
            if _LOAD_ERROR
            else ""
        )
        return _message_outputs(
            f"**Model unavailable:** {html.escape(str(exc))}{detail}"
        )

    try:
        device = getattr(model, "device", torch.device("cpu"))
        tensor = transform(rgb_image).unsqueeze(0).to(device)

        # Ask for intermediate features in the same forward pass so the FAISS
        # search can reuse the embedding instead of running the model twice.
        with torch.inference_mode():
            try:
                output = model(tensor, return_features=True)
            except TypeError:  # model without the return_features argument
                output = model(tensor)

        # The model normally returns a dict, but accept a raw tensor or a
        # tuple as well so an unexpected API change does not crash the app.
        if isinstance(output, dict):
            logits = output.get("logits")
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if not isinstance(logits, torch.Tensor):
            raise ValueError("Model did not return a tensor of logits.")

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.ndim != 2:
            raise ValueError(
                f"Unexpected model output shape: {tuple(logits.shape)}. "
                "Expected (batch, classes)."
            )

        if logits.shape[0] != 1:
            raise ValueError(
                f"Unexpected batch dimension: {logits.shape[0]} (expected 1)."
            )

        if logits.shape[1] != len(idx_to_label):
            raise ValueError(
                f"Model returned {logits.shape[1]} classes but the class "
                f"mapping contains {len(idx_to_label)}."
            )

        probabilities = torch.softmax(logits.float(), dim=1)[0]
        k = min(3, probabilities.numel())
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top: List[Tuple[str, float]] = []
        for score, index in zip(top_probs.tolist(), top_indices.tolist()):
            top.append(
                (idx_to_label.get(int(index), f"class_{index}"), float(score))
            )

        if not top:
            raise ValueError("The model returned no predictions.")

        predicted = top[0][0]
        info = FOOD_INFO.get(predicted)
        top3 = {_pretty(name): float(score) for name, score in top}

        similar_gallery, similar_status = _find_similar_images(
            model=model,
            image=rgb_image,
            transform=transform,
            device=device,
            output=output,
            predicted_label=predicted,
        )

        return (
            top3,
            _format_predicted(predicted, top[0][1]),
            _format_nutrition(info),
            similar_gallery,
            similar_status,
        )

    except Exception as exc:  # noqa: BLE001 - never break the UI
        LOGGER.exception("Inference failed")
        return _message_outputs(
            "**Classification failed:** "
            f"`{html.escape(type(exc).__name__)}: {html.escape(str(exc))}`\n\n"
            "Please try another image. If the problem persists, delete the "
            "Hugging Face cache and restart the app."
        )


def _clear_outputs():
    """Reset the image and all result panels."""
    return (
        None,
        {},
        _READY_PREDICTED,
        _EMPTY_NUTRITION,
        [],
        _EMPTY_SIMILAR_STATUS,
    )


def build_interface() -> gr.Blocks:
    """Create the Gradio Blocks interface."""
    with gr.Blocks(
        title="AI Food Analyzer",
        theme=gr.themes.Default(), #theme=gr.themes.Soft() theme=gr.themes.Monochrome() theme=gr.themes.Glass()
        # theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")
        #theme=gr.themes.Default(font=["Arial", "sans-serif"]),
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown(
            "# AI Food Analyzer\n"
            "Upload a photo to instantly identify the food, estimate calories, and see the ingredients."
        )
        # Main layout 3 columns side by side
        with gr.Row():
            #c1 image and webcam
            with gr.Column(scale=1, min_width=300):
                image_input = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="📷 Upload or capture a food image",
                    height=360,
                )
                #analyze and clear buttons
                with gr.Row():
                    classify_button = gr.Button(
                        "🔍 Analyze Food", variant="primary", scale=2
                    )
                    clear_button = gr.Button(
                        "🗑️ Clear", variant="secondary", scale=1
                    )

                gr.Markdown(
                    "<sub>First prediction downloads the model checkpoint "
                    "(about 1.2 GB) and may take a minute.</sub>"
                )
                #demo examples
                gr.Examples(
                    examples=[["pizza.jpg"], ["dumplings.jpg"]],
                    inputs=image_input,
                    label="Try an example:",
                )
            #c2 predictions
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🎯 Top-3 Predictions")
                #gradio label components to show confidence bars
                top3_output = gr.Label(
                    num_top_classes=3,
                    show_label=False,
                    elem_classes=["compact-label"],
                )
                predicted_output = gr.Markdown(_READY_PREDICTED)
            #c3 nutrition and similar images
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🥗 Nutrition")
                nutrition_output = gr.Markdown(_EMPTY_NUTRITION)

                gr.Markdown("### 🖼️ Visually Similar Dishes")
                similar_gallery = gr.Gallery(
                    label="Top-3 same-class FAISS matches",
                    show_label=False,
                    columns=3,
                    rows=1,
                    height=200,
                    object_fit="cover",
                    allow_preview=True,
                    elem_classes=["compact-gallery"],
                )
                similar_status = gr.Markdown(_EMPTY_SIMILAR_STATUS)
            #event listeners
        classify_button.click(
            fn=predict,
            inputs=image_input,
            outputs=[
                top3_output,
                predicted_output,
                nutrition_output,
                similar_gallery,
                similar_status,
            ],
        )
        clear_button.click(
            fn=_clear_outputs,
            inputs=None,
            outputs=[
                image_input,
                top3_output,
                predicted_output,
                nutrition_output,
                similar_gallery,
                similar_status,
            ],
        )
        #footer
        gr.Markdown(
            """
            ---
            **Model:** [`codealchemist01/food-image-classifier-hybrid`](https://huggingface.co/codealchemist01/food-image-classifier-hybrid)
            · ResNet50 + DeiT-Base with adaptive cross-attention fusion
            · Trained on Food-101 (101 classes)
            · Nutrition values are approximate and not medical advice.
            """
        )

    return demo


demo = build_interface()


if __name__ == "__main__":
    demo.launch(quiet=True)