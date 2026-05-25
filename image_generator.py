"""
image_generator.py - Diagram Generator
  1. Cloudflare AI (FLUX) generates a clean UNLABELLED illustration.
  2. Hardcoded anatomy maps provide accurate label positions for known diagrams.
     Llama is used as fallback for unknown topics.
  3. Pillow draws crisp labels with connector lines from label → anatomy point.
"""

import os
import re
import json
import base64
import logging
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("EduBot.ImageGen")


# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED ANATOMY MAPS
# (x, y) are 0.0–1.0 fractions of image width/height — the actual dot position
# on the anatomy.  Labels are then placed at the edge of the image and a line
# is drawn FROM the label box edge TOWARD the dot.
# ─────────────────────────────────────────────────────────────────────────────

ANATOMY_MAPS: dict[str, list[dict]] = {

    # ── HUMAN HEART ──────────────────────────────────────────────────────────
    "heart": [
        {"label": "Aorta",                   "x": 0.50, "y": 0.10},
        {"label": "Pulmonary Artery",         "x": 0.38, "y": 0.14},
        {"label": "Superior Vena Cava",       "x": 0.64, "y": 0.16},
        {"label": "Pulmonary Veins",          "x": 0.72, "y": 0.30},
        {"label": "Left Atrium",              "x": 0.62, "y": 0.32},
        {"label": "Right Atrium",             "x": 0.38, "y": 0.32},
        {"label": "Aortic Valve",             "x": 0.52, "y": 0.40},
        {"label": "Mitral Valve",             "x": 0.60, "y": 0.46},
        {"label": "Tricuspid Valve",          "x": 0.42, "y": 0.46},
        {"label": "Left Ventricle",           "x": 0.62, "y": 0.60},
        {"label": "Right Ventricle",          "x": 0.40, "y": 0.60},
        {"label": "Inferior Vena Cava",       "x": 0.46, "y": 0.82},
    ],

    # ── HUMAN BRAIN ──────────────────────────────────────────────────────────
    "brain": [
        {"label": "Frontal Lobe",             "x": 0.28, "y": 0.22},
        {"label": "Parietal Lobe",            "x": 0.54, "y": 0.18},
        {"label": "Occipital Lobe",           "x": 0.76, "y": 0.28},
        {"label": "Temporal Lobe",            "x": 0.34, "y": 0.54},
        {"label": "Cerebellum",               "x": 0.70, "y": 0.65},
        {"label": "Brain Stem",               "x": 0.54, "y": 0.78},
        {"label": "Corpus Callosum",          "x": 0.50, "y": 0.38},
        {"label": "Thalamus",                 "x": 0.52, "y": 0.48},
        {"label": "Hypothalamus",             "x": 0.46, "y": 0.56},
        {"label": "Pituitary Gland",          "x": 0.44, "y": 0.64},
        {"label": "Medulla Oblongata",        "x": 0.56, "y": 0.84},
        {"label": "Cerebral Cortex",          "x": 0.36, "y": 0.14},
    ],

    # ── HUMAN EYE ─────────────────────────────────────────────────────────────
    "eye": [
        {"label": "Cornea",                   "x": 0.22, "y": 0.50},
        {"label": "Iris",                     "x": 0.32, "y": 0.50},
        {"label": "Pupil",                    "x": 0.32, "y": 0.50},
        {"label": "Lens",                     "x": 0.40, "y": 0.50},
        {"label": "Retina",                   "x": 0.74, "y": 0.50},
        {"label": "Optic Nerve",              "x": 0.84, "y": 0.54},
        {"label": "Vitreous Humour",          "x": 0.58, "y": 0.50},
        {"label": "Sclera",                   "x": 0.60, "y": 0.26},
        {"label": "Choroid",                  "x": 0.68, "y": 0.30},
        {"label": "Fovea",                    "x": 0.72, "y": 0.50},
        {"label": "Aqueous Humour",           "x": 0.30, "y": 0.38},
        {"label": "Ciliary Body",             "x": 0.46, "y": 0.32},
    ],

    # ── PLANT CELL ────────────────────────────────────────────────────────────
    "plant cell": [
        {"label": "Cell Wall",                "x": 0.50, "y": 0.10},
        {"label": "Cell Membrane",            "x": 0.50, "y": 0.14},
        {"label": "Nucleus",                  "x": 0.46, "y": 0.46},
        {"label": "Chloroplast",              "x": 0.30, "y": 0.36},
        {"label": "Mitochondria",             "x": 0.66, "y": 0.38},
        {"label": "Vacuole",                  "x": 0.52, "y": 0.56},
        {"label": "Endoplasmic Reticulum",    "x": 0.36, "y": 0.58},
        {"label": "Golgi Apparatus",          "x": 0.64, "y": 0.58},
        {"label": "Ribosome",                 "x": 0.44, "y": 0.30},
        {"label": "Cytoplasm",                "x": 0.72, "y": 0.52},
        {"label": "Nucleolus",                "x": 0.46, "y": 0.44},
        {"label": "Plasmodesma",              "x": 0.50, "y": 0.88},
    ],

    # ── ANIMAL CELL ───────────────────────────────────────────────────────────
    "animal cell": [
        {"label": "Cell Membrane",            "x": 0.50, "y": 0.12},
        {"label": "Nucleus",                  "x": 0.48, "y": 0.46},
        {"label": "Nucleolus",                "x": 0.48, "y": 0.44},
        {"label": "Mitochondria",             "x": 0.68, "y": 0.38},
        {"label": "Ribosome",                 "x": 0.36, "y": 0.32},
        {"label": "Endoplasmic Reticulum",    "x": 0.34, "y": 0.54},
        {"label": "Golgi Apparatus",          "x": 0.64, "y": 0.56},
        {"label": "Lysosome",                 "x": 0.70, "y": 0.60},
        {"label": "Centriole",                "x": 0.54, "y": 0.34},
        {"label": "Cytoplasm",                "x": 0.56, "y": 0.68},
        {"label": "Cytoskeleton",             "x": 0.28, "y": 0.44},
        {"label": "Vacuole",                  "x": 0.44, "y": 0.62},
    ],

    # ── HUMAN EAR ─────────────────────────────────────────────────────────────
    "ear": [
        {"label": "Pinna (Auricle)",          "x": 0.16, "y": 0.42},
        {"label": "Ear Canal",                "x": 0.28, "y": 0.50},
        {"label": "Eardrum (Tympanic Membrane)", "x": 0.38, "y": 0.50},
        {"label": "Malleus",                  "x": 0.44, "y": 0.44},
        {"label": "Incus",                    "x": 0.50, "y": 0.42},
        {"label": "Stapes",                   "x": 0.56, "y": 0.44},
        {"label": "Oval Window",              "x": 0.60, "y": 0.50},
        {"label": "Cochlea",                  "x": 0.66, "y": 0.56},
        {"label": "Auditory Nerve",           "x": 0.76, "y": 0.60},
        {"label": "Eustachian Tube",          "x": 0.56, "y": 0.70},
        {"label": "Semicircular Canals",      "x": 0.64, "y": 0.36},
        {"label": "Vestibule",                "x": 0.60, "y": 0.46},
    ],

    # ── FLOWER ────────────────────────────────────────────────────────────────
    "flower": [
        {"label": "Petal",                    "x": 0.50, "y": 0.12},
        {"label": "Stamen",                   "x": 0.54, "y": 0.36},
        {"label": "Pistil",                   "x": 0.48, "y": 0.38},
        {"label": "Stigma",                   "x": 0.48, "y": 0.30},
        {"label": "Style",                    "x": 0.48, "y": 0.42},
        {"label": "Ovary",                    "x": 0.48, "y": 0.52},
        {"label": "Anther",                   "x": 0.58, "y": 0.28},
        {"label": "Filament",                 "x": 0.58, "y": 0.38},
        {"label": "Sepal",                    "x": 0.42, "y": 0.58},
        {"label": "Receptacle",               "x": 0.50, "y": 0.68},
        {"label": "Pedicel (Stem)",           "x": 0.50, "y": 0.80},
        {"label": "Nectary",                  "x": 0.40, "y": 0.50},
    ],

    # ── HUMAN KIDNEY ─────────────────────────────────────────────────────────
    "kidney": [
        {"label": "Renal Cortex",             "x": 0.50, "y": 0.18},
        {"label": "Renal Medulla",            "x": 0.50, "y": 0.38},
        {"label": "Renal Pelvis",             "x": 0.54, "y": 0.54},
        {"label": "Renal Artery",             "x": 0.62, "y": 0.42},
        {"label": "Renal Vein",               "x": 0.62, "y": 0.52},
        {"label": "Ureter",                   "x": 0.58, "y": 0.72},
        {"label": "Nephron",                  "x": 0.38, "y": 0.32},
        {"label": "Glomerulus",               "x": 0.34, "y": 0.44},
        {"label": "Bowman's Capsule",         "x": 0.32, "y": 0.52},
        {"label": "Loop of Henle",            "x": 0.40, "y": 0.60},
        {"label": "Collecting Duct",          "x": 0.50, "y": 0.64},
        {"label": "Renal Capsule",            "x": 0.50, "y": 0.08},
    ],
}


def _match_anatomy(topic: str) -> Optional[list[dict]]:
    """Return hardcoded labels if the topic matches a known anatomy map."""
    t = topic.lower()
    for key, labels in ANATOMY_MAPS.items():
        if key in t:
            return labels
    return None


def _cf_url(model: str) -> str:
    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{os.environ['CF_ACCOUNT_ID']}/ai/run/{model}"
    )

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['CF_API_TOKEN']}",
        "Content-Type": "application/json",
    }


async def _get_labels_llm(client: httpx.AsyncClient, topic: str) -> list[dict]:
    """Fallback: ask Llama for labels when topic isn't in hardcoded maps."""
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a science expert. Reply ONLY with a JSON array, no markdown. "
                    "Each item: {\"label\": \"correct scientific name\", \"x\": 0.0-1.0, \"y\": 0.0-1.0}. "
                    "x=0 left, x=1 right, y=0 top, y=1 bottom. "
                    "Place dots ON the actual anatomical part inside the diagram. "
                    "Spread them across the whole image — not clustered in one corner."
                ),
            },
            {
                "role": "user",
                "content": f"Give 12 correctly named labelled parts for a diagram of: {topic}",
            },
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    resp = await client.post(
        _cf_url("@cf/meta/llama-3.1-8b-instruct"),
        headers=_headers(),
        json=payload,
        timeout=30.0,
    )

    if resp.status_code != 200:
        logger.error(f"Llama label error {resp.status_code}: {resp.text}")
        return []

    raw = resp.json().get("result", {}).get("response", "")
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        logger.error(f"Label JSON parse failed: {raw[:300]}")
    return []


async def _generate_base_image(client: httpx.AsyncClient, topic: str) -> Optional[bytes]:
    """Generate an UNLABELLED illustration — explicitly no text."""
    clean_topic = re.sub(
        r'\b(diagram|label|labelled|labeled|labeling|annotated|with labels|chart)\b',
        '', topic, flags=re.IGNORECASE
    ).strip()

    prompt = (
        f"Plain illustration of {clean_topic}, no text. "
        "White background. "
        "Render ZERO letters, ZERO words, ZERO numbers, ZERO annotations anywhere. "
        "Just the drawing itself, nothing written."
    )
    resp = await client.post(
        _cf_url("@cf/black-forest-labs/flux-1-schnell"),
        headers=_headers(),
        json={"prompt": prompt},
        timeout=60.0,
    )

    if resp.status_code != 200:
        logger.error(f"Image gen error {resp.status_code}: {resp.text}")
        return None

    b64 = resp.json().get("result", {}).get("image")
    return base64.b64decode(b64) if b64 else None


# ─────────────────────────────────────────────────────────────────────────────
# LABEL OVERLAY — dot on anatomy, label at edge, line between them
# ─────────────────────────────────────────────────────────────────────────────

# How far labels are pushed toward the nearest image edge (fraction of image size)
_EDGE_MARGIN = 0.05   # labels sit this far from the edge
_MIN_LEADER  = 30     # minimum connector line length (px) before we skip drawing


def _edge_position(ax: float, ay: float, W: int, H: int, tw: int, th: int,
                   margin: int) -> tuple[int, int]:
    """
    Push the label to the nearest edge of the image so connectors clearly
    point inward.  Returns (tx, ty) — top-left corner of the text bounding box.
    """
    # Determine which edge the dot is closest to
    d_left   = ax
    d_right  = W - ax
    d_top    = ay
    d_bottom = H - ay

    min_d = min(d_left, d_right, d_top, d_bottom)

    if min_d == d_left:          # push label to LEFT edge
        tx = margin
        ty = int(ay - th / 2)
    elif min_d == d_right:       # push label to RIGHT edge
        tx = W - tw - margin
        ty = int(ay - th / 2)
    elif min_d == d_top:         # push label to TOP edge
        tx = int(ax - tw / 2)
        ty = margin
    else:                        # push label to BOTTOM edge
        tx = int(ax - tw / 2)
        ty = H - th - margin

    # Clamp so text is never clipped
    tx = max(margin, min(tx, W - tw - margin))
    ty = max(margin, min(ty, H - th - margin))
    return tx, ty


def _line_from_label_to_dot(tx: int, ty: int, tw: int, th: int,
                             ax: int, ay: int) -> tuple[int, int]:
    """
    Return the point on the label box's border that is closest to the anatomy
    dot — so the connector line starts at the box edge, not its centre.
    """
    # Centre of the label box
    lx = tx + tw // 2
    ly = ty + th // 2

    # Direction from label centre → anatomy dot
    dx = ax - lx
    dy = ay - ly

    if dx == 0 and dy == 0:
        return lx, ly

    # Intersect with the 4 sides of the bounding box
    candidates = []
    half_w, half_h = tw / 2 + 4, th / 2 + 3   # +padding added in draw.rectangle

    if dx != 0:
        t_right = half_w / abs(dx)
        t_left  = half_w / abs(dx)
        for t in (t_right, t_left):
            cx_ = lx + dx * t
            cy_ = ly + dy * t
            if abs(cx_ - lx) <= half_w + 1 and abs(cy_ - ly) <= half_h + 1:
                candidates.append((cx_, cy_, t))

    if dy != 0:
        t_top    = half_h / abs(dy)
        t_bottom = half_h / abs(dy)
        for t in (t_top, t_bottom):
            cx_ = lx + dx * t
            cy_ = ly + dy * t
            if abs(cx_ - lx) <= half_w + 1 and abs(cy_ - ly) <= half_h + 1:
                candidates.append((cx_, cy_, t))

    if not candidates:
        return lx, ly

    # Pick the candidate closest to the dot (largest t still ≤ 1 is overkill;
    # just take smallest positive t — the first edge we hit leaving the box)
    best = min(candidates, key=lambda c: c[2])
    return int(best[0]), int(best[1])


def _overlay_labels(image_bytes: bytes, labels: list[dict]) -> bytes:
    """Pillow draws real text labels with edge-anchored connector lines."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(14, H // 42),
        )
    except IOError:
        font = ImageFont.load_default()

    margin = max(6, int(W * _EDGE_MARGIN))
    dot_r  = max(5, W // 130)
    line_w = max(1, W // 380)

    for item in labels:
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        # Anatomy dot position (the point we're labelling)
        ax = int(float(item.get("x", 0.5)) * W)
        ay = int(float(item.get("y", 0.5)) * H)

        # Measure text
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Place label near the closest image edge
        tx, ty = _edge_position(ax, ay, W, H, tw, th, margin)

        # White pill behind text
        draw.rectangle(
            [tx - 4, ty - 3, tx + tw + 4, ty + th + 3],
            fill=(255, 255, 255),
            outline=(180, 180, 180),
        )

        # Connector line: starts at label box edge, ends just before the dot
        lx, ly = _line_from_label_to_dot(tx, ty, tw, th, ax, ay)
        line_len = ((ax - lx) ** 2 + (ay - ly) ** 2) ** 0.5
        if line_len >= _MIN_LEADER:
            draw.line([(lx, ly), (ax, ay)], fill=(90, 90, 90), width=line_w)

        # Red dot ON the anatomy
        draw.ellipse(
            [ax - dot_r, ay - dot_r, ax + dot_r, ay + dot_r],
            fill=(210, 30, 30),
            outline=(140, 10, 10),
        )

        # Label text
        draw.text((tx, ty), label, fill=(15, 15, 15), font=font)

    out = BytesIO()
    img.save(out, format="JPEG", quality=93)
    return out.getvalue()


# ─────────────────────────────────────────────────────────────────────────────

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    for var in ("CF_ACCOUNT_ID", "CF_API_TOKEN"):
        if not os.environ.get(var):
            logger.error(f"{var} environment variable is missing.")
            return None

    async with httpx.AsyncClient() as client:
        # Run image generation and label lookup concurrently
        import asyncio
        base_image_task = asyncio.create_task(_generate_base_image(client, prompt))

        # Use hardcoded map first; fall back to Llama for unknown topics
        labels = _match_anatomy(prompt)
        if labels is None:
            logger.info(f"No hardcoded map for '{prompt}' — asking Llama.")
            labels = await _get_labels_llm(client, prompt)
        else:
            logger.info(f"Using hardcoded anatomy map for '{prompt}'.")

        base_image = await base_image_task

    if not base_image:
        return None

    if not labels:
        logger.warning("No labels — returning unlabelled image.")
        return base_image, prompt

    try:
        return _overlay_labels(base_image, labels), prompt
    except Exception as e:
        logger.error(f"Label overlay failed: {e}")
        return base_image, prompt
