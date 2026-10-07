"""
Hunyuan3D 2 MV — multi-view image-to-mesh for Modly.

Reference : https://huggingface.co/tencent/Hunyuan3D-2mv

Modly hands generators a single image, so extra views travel inside it:

  layout = "grid"    2x2 view sheet   ┌───────┬───────┐
                                       │ front │ left  │
                                       ├───────┼───────┤
                                       │ back  │ right │
                                       └───────┴───────┘
  layout = "row"     turnaround strip, left to right: front, left, back[, right],
                     split into the separate figures (arms may overlap, not touch)
  layout = "single"  the image is the front view only
  layout = "auto"    picks one of the above from where the figures are

Blank cells (uniform colour or fully transparent) are skipped; front is required.
Views follow Tencent's convention: "left" is the object turned 90° clockwise seen
from above (i.e. the object's own left side), "right" is 270°.

Headless callers (CLI / MCP) may also pass absolute file paths in params as
`left_image_path`, `back_image_path`, `right_image_path`; these override the
matching sheet cell.
"""
import hashlib
import io
import random
import sys
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageChops, ImageStat

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID       = "tencent/Hunyuan3D-2mv"
_SUBFOLDER        = "hunyuan3d-dit-v2-mv"
# hy3dgen is fetched from a pinned commit and checked against its archive hash
# before any of it is imported.
_HY3DGEN_COMMIT   = "f8db63096c8282cb27354314d896feba5ba6ff8a"
_HY3DGEN_SHA256   = "1be140879646034a98941e494072ce0ab26918fdc2fb59210443d56d41bc5622"
_GITHUB_ZIP       = f"https://github.com/Tencent/Hunyuan3D-2/archive/{_HY3DGEN_COMMIT}.zip"

_VIEWS            = ("front", "left", "back", "right")
_EXTRA_VIEW_PARAM = "{view}_image_path"
_IMAGE_SUFFIXES   = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_LAYOUTS          = ("auto", "grid", "row", "single")


class Hunyuan3DMVGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-mv"
    DISPLAY_NAME = "Hunyuan3D 2 MV"
    VRAM_GB      = 6

    _rembg_session = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check if self.download_check else _SUBFOLDER
        model_dir = self.model_dir / subfolder
        return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen()

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        if sys.platform == "darwin":
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            dtype = torch.float32  # MPS has limited fp16 op coverage
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32

        subfolder = self.download_check if self.download_check else _SUBFOLDER
        print(f"[Hunyuan3DMVGenerator] Loading pipeline from {self.model_dir} (subfolder={subfolder})…")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
            dtype=dtype,
        )
        self._model  = pipeline
        self._device = device
        print(f"[Hunyuan3DMVGenerator] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        self._rembg_session = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        layout           = str(params.get("layout", "auto"))
        num_steps        = int(params.get("num_inference_steps", 30))
        vert_count       = int(params.get("vertex_count", 0))
        octree_res       = int(params.get("octree_resolution", 380))
        guidance_scale   = float(params.get("guidance_scale", 5.5))
        seed             = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        self._report(progress_cb, 3, "Reading views…")
        raw_views = self._collect_views(image_bytes, layout, params)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 5, "Removing background…")
        views = {}
        for view, img in raw_views.items():
            cut = self._preprocess(img)
            if cut is not None:
                views[view] = cut
            self._check_cancelled(cancel_event)
        if "front" not in views:
            raise ValueError("No object found in the front view after background removal.")

        used = ", ".join(views)
        print(f"[Hunyuan3DMVGenerator] Views: {used}  seed={seed}")
        step = f"Generating 3D shape ({used})…"
        self._report(progress_cb, 12, step)
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, 82, step, stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=views,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=20000 if getattr(self, "_device", "") == "cuda" else 4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if params.get("remove_floaters", True) not in (False, "false", 0, "0"):
            self._report(progress_cb, 84, "Removing floating specks…")
            mesh = self._remove_floaters(mesh)

        if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
            self._report(progress_cb, 85, "Optimizing mesh…")
            mesh = self._decimate(mesh, vert_count)

        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    def _collect_views(self, image_bytes: bytes, layout: str, params: dict) -> dict:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        if layout not in _LAYOUTS:
            raise ValueError(f"Unknown layout '{layout}'. Use one of: {', '.join(_LAYOUTS)}.")

        figures = None
        if layout in ("auto", "row"):
            figures = self._split_figures(image)
            if layout == "auto":
                mask     = self._foreground_mask(image)
                row_runs = self._runs(mask, axis=1)
                if len(row_runs) >= 2:
                    layout = "grid"
                elif len(figures) >= 2:
                    layout = "row"
                else:
                    layout = "single"
                print(f"[Hunyuan3DMVGenerator] Auto layout: {layout} "
                      f"({len(figures)} figure(s), {len(row_runs)} row group(s))")

        if layout == "single":
            views = {"front": image}
        elif layout == "row":
            if not 1 <= len(figures) <= len(_VIEWS):
                raise ValueError(
                    f"Found {len(figures)} separate figures in the strip; expected 1-4 "
                    "(front, left, back, right) that don't touch each other."
                )
            views = dict(zip(_VIEWS, figures))
        elif layout == "grid":
            w, h   = image.size
            hw, hh = w // 2, h // 2
            cells  = {
                "front": (0,  0,  hw, hh),
                "left":  (hw, 0,  w,  hh),
                "back":  (0,  hh, hw, h),
                "right": (hw, hh, w,  h),
            }
            views = {view: image.crop(box) for view, box in cells.items()}

        for view in _VIEWS[1:]:
            value = params.get(_EXTRA_VIEW_PARAM.format(view=view))
            if not value:
                continue
            p = Path(str(value)).expanduser().resolve()
            if p.suffix.lower() not in _IMAGE_SUFFIXES:
                raise ValueError(f"{view} view must be an image file ({', '.join(sorted(_IMAGE_SUFFIXES))}): {p}")
            if not p.is_file():
                raise FileNotFoundError(f"{view} view image not found: {p}")
            img = Image.open(p)
            img.load()
            views[view] = img

        views = {v: img for v, img in views.items() if not self._is_blank(img)}
        if "front" not in views:
            raise ValueError(
                "Front view is empty. With the 2x2 sheet layout the front view goes in the top-left cell."
            )
        return {v: views[v] for v in _VIEWS if v in views}

    @classmethod
    def _split_figures(cls, image: Image.Image) -> list:
        """
        Figures of a turnaround strip, left to right, each cropped on its own.

        Figures are the connected shapes at least 60% as tall as the tallest one;
        smaller detached bits (a loose strand of hair, a gem) join the figure they sit
        under. Splitting by shape rather than by empty columns lets T-pose arms reach
        past the next figure, and anything of a neighbour inside a crop is blanked out.
        """
        import numpy as np
        from scipy import ndimage

        fg = np.asarray(cls._foreground_mask(image, max_side=None)) > 0
        labels, count = ndimage.label(fg)
        if count == 0:
            return []
        boxes   = ndimage.find_objects(labels)
        areas   = ndimage.sum_labels(fg, labels, index=range(1, count + 1))
        heights = [b[0].stop - b[0].start for b in boxes]
        tallest = max(heights)
        main = sorted(
            (i for i in range(count) if heights[i] >= tallest * 0.6),
            key=lambda i: (boxes[i][1].start + boxes[i][1].stop) / 2,
        )
        centres = [(boxes[i][1].start + boxes[i][1].stop) / 2 for i in main]

        owner = np.zeros(count + 1, dtype=np.int32)  # label -> figure number + 1, 0 = dropped
        for i in range(count):
            if i in main:
                owner[i + 1] = main.index(i) + 1
            elif areas[i] >= 16:
                cx = (boxes[i][1].start + boxes[i][1].stop) / 2
                owner[i + 1] = int(np.argmin([abs(cx - c) for c in centres])) + 1
        figure_of = owner[labels]

        rgba = np.array(image.convert("RGBA"))
        opaque = rgba[..., 3].min() >= 250
        if opaque:
            h, w = fg.shape
            corners = rgba[[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1], :3]
            bg = np.median(corners, axis=0).astype(np.uint8)

        figures = []
        for n in range(1, len(main) + 1):
            ys, xs = np.nonzero(figure_of == n)
            pad = max(4, int(0.03 * (ys.max() - ys.min())))
            y0, y1 = max(0, ys.min() - pad), min(fg.shape[0], ys.max() + 1 + pad)
            x0, x1 = max(0, xs.min() - pad), min(fg.shape[1], xs.max() + 1 + pad)
            crop  = rgba[y0:y1, x0:x1].copy()
            other = (figure_of[y0:y1, x0:x1] != n) & fg[y0:y1, x0:x1]
            if opaque:
                crop[other, :3] = bg  # keep it opaque so background removal still runs
            else:
                crop[other, 3] = 0
            figures.append(Image.fromarray(crop, "RGBA"))
        return figures

    @staticmethod
    def _foreground_mask(image: Image.Image, max_side: Optional[int] = 512) -> Image.Image:
        """Binary mask of the figures, downscaled to max_side for speed when given."""
        img = image.convert("RGBA")
        if max_side:
            img.thumbnail((max_side, max_side))
        alpha = img.getchannel("A")
        if alpha.getextrema()[0] < 250:
            return alpha.point(lambda a: 255 if a > 16 else 0)
        # Opaque image: background is whatever colour the corners share.
        rgb = img.convert("RGB")
        w, h = rgb.size
        corners = [rgb.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
        bg = tuple(sorted(c[i] for c in corners)[1] for i in range(3))
        diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg))
        return ImageChops.lighter(ImageChops.lighter(*diff.split()[:2]), diff.split()[2]).point(
            lambda d: 255 if d > 32 else 0
        )

    @staticmethod
    def _runs(mask: Image.Image, axis: int) -> list:
        """
        Groups of occupied columns (axis=0) or rows (axis=1) in a mask, as
        (start, end) in mask pixels. Specks are ignored.
        """
        w, h = mask.size
        length, across = (w, h) if axis == 0 else (h, w)
        px = mask.load()
        counts = [
            sum(1 for j in range(across) if px[(i, j) if axis == 0 else (j, i)])
            for i in range(length)
        ]
        runs, start = [], None
        for i, n in enumerate(counts + [0]):
            occupied = n > across * 0.005
            if occupied and start is None:
                start = i
            elif not occupied and start is not None:
                if i - start >= length * 0.02:
                    runs.append((start, i))
                start = None
        return runs

    @staticmethod
    def _is_blank(img: Image.Image) -> bool:
        if img.width < 8 or img.height < 8:
            return True
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            alpha = img.convert("RGBA").getchannel("A")
            if alpha.getextrema()[1] < 16:
                return True
        stddev = ImageStat.Stat(img.convert("L")).stddev[0]
        return stddev < 2.0

    def _preprocess(self, img: Image.Image) -> Optional[Image.Image]:
        """Background-removed RGBA, or None when nothing is left of the object."""
        rgba = img.convert("RGBA")
        lo, hi = rgba.getchannel("A").getextrema()
        if lo < 250 and hi > 16:
            # Already cut out (transparent background) — keep the artist's mask.
            out = rgba
        else:
            out = self._remove_background(img)
        if out.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox() is None:
            return None
        return out

    def _remove_background(self, img: Image.Image) -> Image.Image:
        import rembg
        if self._rembg_session is None:
            try:
                self._rembg_session = rembg.new_session("u2net")
            except Exception:
                # cuDNN/CUDA incompatibility — fall back to CPU
                self._rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
        try:
            return rembg.remove(img, session=self._rembg_session).convert("RGBA")
        except Exception:
            self._rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=self._rembg_session).convert("RGBA")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remove_floaters(mesh, min_share: float = 0.005, max_share: float = 0.1, overhang: float = 0.1):
        """
        Drops disconnected pieces that are either specks (under min_share of all faces)
        or stray rods and sheets. The model sometimes leaves thin pieces running straight
        along the depth axis; a smaller piece (under max_share) counts as one when it
        sticks out of the main body's bounding box by more than `overhang` of its size,
        or when it is a thin rod or sheet whose longest side is the depth axis.
        """
        import numpy as np
        import trimesh
        try:
            parts = mesh.split(only_watertight=False)
        except Exception as exc:
            print(f"[Hunyuan3DMVGenerator] Floater removal skipped: {exc}")
            return mesh
        if len(parts) <= 1:
            return mesh
        main   = max(parts, key=lambda p: len(p.faces))
        margin = main.extents * overhang
        lo, hi = main.bounds[0] - margin, main.bounds[1] + margin

        def stray(p) -> bool:
            if len(p.faces) < len(mesh.faces) * min_share:
                return True
            if len(p.faces) >= len(mesh.faces) * max_share:
                return False
            sticks_out = np.any(p.bounds[0] < lo) or np.any(p.bounds[1] > hi)
            ext = p.extents
            depth_rod = int(np.argmax(ext)) == 2 and np.sort(ext)[1] < ext.max() * 0.1
            return bool(sticks_out or depth_rod)

        keep = [p for p in parts if p is main or not stray(p)]
        print(f"[Hunyuan3DMVGenerator] Removed {len(parts) - len(keep)} floating piece(s).")
        return trimesh.util.concatenate(keep)

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[Hunyuan3DMVGenerator] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3DMVGenerator] Downloading {_HF_REPO_ID} (standard variant, safetensors only)…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-mv-fast/**",
                "hunyuan3d-dit-v2-mv-turbo/**",
                "*.ckpt",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )
        print("[Hunyuan3DMVGenerator] Download complete.")

    def _ensure_hy3dgen(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dgen still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_hy3dgen(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3DMVGenerator] Downloading hy3dgen source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != _HY3DGEN_SHA256:
            raise RuntimeError(
                f"hy3dgen archive checksum mismatch (expected {_HY3DGEN_SHA256}, got {digest}); refusing to install it."
            )
        print("[Hunyuan3DMVGenerator] Extracting hy3dgen…")

        strip  = f"Hunyuan3D-2-{_HY3DGEN_COMMIT}/"
        prefix = f"{strip}hy3dgen/"
        root   = dest.resolve()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = (dest / rel).resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError(f"Refusing to extract outside {dest}: {member}")
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Hunyuan3DMVGenerator] hy3dgen extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "layout",
                "label":   "Input Layout",
                "type":    "select",
                "default": "auto",
                "options": [
                    {"value": "auto",   "label": "Auto detect"},
                    {"value": "row",    "label": "Turnaround strip (front, left, back, right)"},
                    {"value": "grid",   "label": "2x2 view sheet"},
                    {"value": "single", "label": "Single front image"},
                ],
                "tooltip": "Auto detects the layout. 2x2 sheet: front | left / back | right. Strip: front, left, back[, right] side by side with space between. Blank cells are skipped.",
            },
            {
                "id":      "num_inference_steps",
                "label":   "Quality",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast"},
                    {"value": 30, "label": "Balanced"},
                    {"value": 50, "label": "High"},
                ],
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 380, "label": "Medium"},
                    {"value": 512, "label": "High"},
                ],
                "tooltip": "Octree resolution for mesh reconstruction. Higher = more detail but slower and more VRAM.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "float",
                "default": 5.5,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input views.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     2147483647,
                "tooltip": "Seed for reproducibility. Set to -1 for a random seed.",
            },
        ]
