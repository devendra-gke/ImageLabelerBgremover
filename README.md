# 🧠 Smart Image Processor (AI Background Remover)

A Django-based AI-powered image processing tool that:
- Removes background using AI (BiRefNet)
- Adds text from Excel mapping
- Saves images into:
  - Itemwise folders (Category)
  - StyleBio folders (Style)
- Supports bulk image upload

---

## ⚙️ Requirements

- Python **3.11** (Mandatory)
- pip
- Virtual Environment (recommended)

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/devendra-gke/ImageLabelerBgremover.git
cd ImageLabelerBgremover```

### 2. Install and run

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python manage.py runserver
```

Open http://127.0.0.1:8000/ . The first image downloads the segmentation
model (~1 GB for `bria-rmbg`); later runs reuse it.

---

## 💎 v2.0 — Jewellery detection & accurate background removal

Every photo now goes through these steps (`processor/services/jewellery_detection.py`):

1. **Locate** – a small, fast model (`u2netp`) finds the jewellery on a
   reduced copy of the photo (chain links / stones / clasps are grouped into
   one item; dust is ignored). The boxes are shown in the result inspector.
2. **Cut out** – the accurate model (`bria-rmbg`) runs **once**, on the
   full-resolution crop around the jewellery. Small products in big photos
   (earrings, rings, studs) get several times more detail: thin chains,
   prongs and stone outlines stay sharp instead of blurry/haloed.
3. **Clean-up** (`hole_cleanup.py`, crop only) – using a model of the backdrop colour:
   * backdrop inside hollows is removed (ring/bangle centres, chain gaps, open pendants)
   * soft drop shadows and haze are removed
   * faded chain links / prongs are restored
   * dust specks are removed

   Diamond facets and sparkle are protected (only flat regions at backdrop
   brightness are cleared; brighter-than-backdrop facets never are).
   * **diamonds are kept fully opaque** – models often leave stones half
     transparent (they reflect the backdrop), which makes them look grey on
     dark pages and washed-out on white. Stones are recognised by their facet
     texture and nearby sparkle and set to full opacity; small gaps between
     settings are recognised as smooth, backdrop-coloured and framed by
     coloured metal, and cleared. Settings: `solidify_stones`,
     `backdrop_texture_max`, `stone_sparkle_min`, `gap_border_min_saturation`.
   * **edges are refined at full resolution** – the model's matte is computed
     at ~1024 px; a guided filter re-fits it to the real photo edges, the
     camera's sharpening halo is ignored, and backdrop colour is removed from
     soft edge pixels (no light/grey outline on dark pages). Settings:
     `refine_edges`, `edge_radius_ratio`, `edge_contrast`, `decontaminate_edges`.
   * the backdrop is estimated **locally** (lighting varies across a photo), so
     openwork hollows framed by white gold / pavé are cleared too, and small
     backdrop pockets inside gold chains are opened (`open_metal_pockets`).
   * very large crops (a necklace filling a 30 MP photo) are cleaned on a
     reduced copy and in strips, so they stay fast and within memory
     (`full_res_cleanup_max_px`).
4. **Validate** – warnings when the product touches the photo edge, looks
   blurry, or nothing is detected (such photos are skipped).

Catalogue images are now **auto-cropped and centred** with the preset's
`padding_percent` (`image_processing.auto_crop`), so every item is framed the same.

Click any processed result to open the **inspector**: before/after slider,
detection boxes, warnings, transparent PNG download and **Open in 3D**.

### Settings (`jewellery_image_settings.json` / `diamond_image_settings.json`)

| Key | Section | Meaning |
|---|---|---|
| `model` | `background_removal` | `bria-rmbg` (default), `birefnet-general`, `birefnet-massive`, `isnet-general-use` (light, for low-RAM PCs) |
| `detection.model` | `background_removal` | fast locator model (default `u2netp`) |
| `detection.refine` | `background_removal` | cut out on the crop around the jewellery (fast + accurate; keep on) |
| `detection.detector` | `background_removal` | `saliency` (default) or `prompt` (Grounding DINO – needs `pip install torch transformers`; for worn jewellery, hands, display busts) |
| `remove_inner_holes`, `hole_*` | `background_removal` | hollow clean-up |
| `remove_background_shadows` | `background_removal` | soft-shadow removal |
| `recover_thin_details` | `background_removal` | restore faded chain links / prongs |
| `remove_specks` | `background_removal` | drop dust specks |
| `auto_crop`, `padding_percent`, `max_upscale` | `image_processing` | catalogue framing |

The environment variable `BG_REMOVAL_MODEL` overrides the preset model (e.g.
`set BG_REMOVAL_MODEL=isnet-general-use` on a PC with 8 GB RAM).

---

## ⚡ Speed

* The accurate model runs once per photo (on the jewellery crop); a tiny
  model (~0.5 s) does the locating.
* Clean-up and colour protection run only on the crop, not the whole photo,
  so 24–50 MP camera photos cost about the same as small ones.
* The browser sends photos one per request, two at a time, with a live
  progress bar, time-left estimate and a **Stop** button. Results appear as
  each photo finishes; one bad photo never fails the batch.
* Responses carry small preview thumbnails; full-size files stay on disk.
* Models load in the background when the server starts (`[warm-up] AI models ready`).

Measured on a 2-core test machine with a 28 MP photo: **33 s → 9 s** per
image (light test model). With `bria-rmbg` the model share is larger, so the
gain is mostly the removed second model pass plus ~20 s of full-photo work.

**Use the GPU (biggest win, 5–20× on the model step):**

| PC | Install instead of `onnxruntime` |
|---|---|
| Any Windows GPU (NVIDIA / AMD / Intel) | `pip uninstall onnxruntime && pip install onnxruntime-directml` |
| NVIDIA + CUDA | `pip uninstall onnxruntime && pip install onnxruntime-gpu` |
| Intel CPU without GPU | `pip uninstall onnxruntime && pip install onnxruntime-openvino` |

The app picks the GPU automatically (override with `ONNX_PROVIDERS`).

Other knobs: `BG_REMOVAL_MODEL=birefnet-general-lite` (faster accurate model),
`ONNX_THREADS=8` (CPU threads), `WARMUP_MODELS=0` (skip start-up warm-up).

---

## ⬡ 3D view

Open the **3D view** tab, drop a jewellery photo (it is cut out automatically)
or a transparent PNG, or click **Open in 3D** on a processed result.

* **Depth source** – `Shape` (rounded relief from the outline, instant, offline),
  `AI` (Depth Anything V2, ~100 MB, downloaded once from GitHub on first use to
  `~/.image_labeler/models`), `Hybrid` (both; default once the AI model is present).
* **Shape** – depth, surface detail (engraving / facets), smoothing, mesh quality,
  mirrored back side for full 360°.
* **Look** – Studio / Polished metal / Soft matte / Photo-true materials, light,
  exposure, background colour or transparent, floor shadow, wireframe.
* **Motion** – swing (default, ±35°), full 360° spin, or free drag; loop length.
* **Export** – PNG image (2000 px), turntable video (MP4 in Chrome/Edge, WebM
  otherwise; one loop), 3D model `.glb` (Windows 3D Viewer, Blender, websites via
  `<model-viewer>`), and a ZIP of 24/36/72 frames for 360° product spinners.

A single photo has no information about the sides or back, so the model is a
relief: it looks best from the front and up to ~40° each side.

All 3D code runs in the browser (three.js is bundled in
`processor/static/vendor/three`, no internet needed).

### Tests

```bash
python manage.py test processor
```

## v2.5 - white outline and leftover backdrop
* Backdrop that sits right beside a wire or around the outline is no longer given back as a pale outline (`solidify_stones` now leaves anything that borders a flat patch of backdrop).
* New `remove_thin_gaps` pass clears narrow backdrop strips between two strands of coloured metal (hooks, double wires, scroll gaps).
* The stone-reflection restore no longer brings back flat patches that match the exterior backdrop colour.

## v2.6 - leftover backdrop haze
- New final pass `clear_backdrop_haze`: semi-transparent pixels whose colour matches the local backdrop (neutral, not brighter) are made fully transparent, so no white/grey film or blur remains inside medallions and openwork.
- Open backdrop areas inside medallions are detected more reliably and are no longer re-filled by the stone/reflection restore steps.
- Tunables in DEFAULTS: `haze_color_tolerance`, `haze_chroma`, `haze_max_brighter`, `haze_max_alpha`, `clear_backdrop_haze`.

## v2.7 - solid backdrop patches left opaque
- New final pass `clear_flat_backdrop`: solid pale/grey blobs inside medallions and openwork that the model left opaque are cleared. A blob is cleared only if (a) it survives a morphological open (so faceted diamonds do not qualify), (b) it is mostly bordered by warm metal, and (c) its colour matches the backdrop. This keeps large diamond facets safe.
- Tunables: `flat_backdrop_dist`, `flat_backdrop_chroma`, `flat_backdrop_open`, `flat_backdrop_min_area`, `flat_backdrop_metal_frac`, `flat_backdrop_core_dist`, `clear_flat_backdrop`.
- Note: for white-gold/silver pieces the metal-border test will not match, so the pass stays off rather than risk stones.

## v2.7.1 - new UI
- Warm editorial theme (cream paper, burnt-orange accent, gold highlights; Fraunces / IBM Plex Mono / Manrope fonts).
- Effects: cursor-following glow, floating gem outlines, scroll-reveal cards, hard-shadow lift on cards and buttons, shimmer on the title, shine sweep on the main button, striped progress bar, drag-over highlight on drop zones, animated modal. Honours `prefers-reduced-motion`.
- Template only: no element IDs or JS hooks changed, so processing and the 3D viewer behave as before.

## v2.7.2 - Gurukrupa pink theme
- Palette switched to pink: blush paper background, Gurukrupa pink accent (`#d6336c`), deep plum ink, rose-gold highlight. Change `--accent`, `--paper`, `--ink` in the `:root` block of `image_upload.html` to retheme.

## v2.7.3 - Mission 3X logo, lighter pink
- Gurukrupa | Mission 3X logo added to the top bar and the header (`processor/static/img/gk-3x-logo.png`).
- Palette lightened: pale blush background, softer pink accent (`--accent: #e8629a`).

## v2.8 - smoother edges, tiny edges and holes
New final pass `finish_edges` (runs after every other clean-up, in strips so 30 MP crops stay light):
- **Smooth contour**: the matte edge is blurred and re-sharpened around 50 % so stair-steps and wobble disappear but the edge stays crisp. Thin wires keep their width.
- **Defringe**: partly-transparent edge pixels that are exactly the backdrop colour (neutral, not brighter) are dropped, so no leftover coloured outline.
- **Tiny holes**: small enclosed openings (links, bead holes, filigree dots) are cleared when they are flat backdrop colour, framed by metal, and the model itself was unsure there.
- **Diamond shine protected**: stone clusters, sparkle neighbourhoods and anything brighter than the backdrop are never touched.
- Tunables in DEFAULTS: `finish_edges`, `smooth_gain`, `smooth_sigma_ratio`, `defringe_*`, `tiny_hole_*`, `finish_zone_*`.

## v2.9 - openings between spokes and stone rows
- New `clear_backdrop_wedges` pass (native resolution): small openings between spokes, wires and stone rows that are exactly the backdrop colour (tiny colour distance, flat, neutral, no sparkle inside, touching metal or stones) are cleared. Diamond facets and shine are not touched.
- Tunables: `wedge_*` in DEFAULTS; `clear_wedges: False` turns it off.
