# Hunyuan3D 2 MV — Modly extension

Multi-view image-to-mesh for [Modly](https://github.com/lightningpixel/modly), using
Tencent's [`Hunyuan3D-2mv`](https://huggingface.co/tencent/Hunyuan3D-2mv). Giving the
model the back and sides of an object, instead of letting it guess them from one photo,
gives noticeably more faithful shapes. Output is geometry only (no textures).

## Giving it several views

Modly passes one image to a generator, so the views travel inside it. Two layouts are
understood, and **Input Layout = Auto detect** (the default) tells them apart:

**Turnaround strip**: figures side by side, left to right: front, left, back and
optionally right, with some empty space between them. This is the usual character
turnaround sheet; the image is split at the gaps, so T-pose arms are fine as long as
they don't touch the next figure.

```
 front    left    back   (right)
```

**2x2 sheet**:

```
┌───────┬───────┐
│ front │ left  │
├───────┼───────┤
│ back  │ right │
└───────┴───────┘
```

- **front** is required; leave any other view out (or its cell blank) to skip it.
- **left** is the object's own left side: its front points to the **left edge of the
  image** (walk to *your* right around the object). **right** is the opposite.
- Keep the same distance, height and lighting in every view, with the object centred.
- Backgrounds are removed automatically; images that already have a transparent
  background keep their own cut-out. On an opaque strip, the background should be one
  plain colour so the gaps between figures can be found.
- A single ordinary photo is detected as a single front view.

Build a sheet from separate photos with the bundled helper (needs Python + Pillow):

```
python tools/make_sheet.py --front front.jpg --left left.jpg --back back.jpg -o sheet.png
```

Then upload `sheet.png` in Modly. Auto detect handles it; pick a layout explicitly only
if detection guesses wrong.

From the `modly` CLI / MCP you can skip the sheet and pass extra views as absolute paths:

```
modly workflow-run from-image --image /abs/front.jpg --model hunyuan3d-mv/generate \
  --params-json '{"layout":"single","back_image_path":"/abs/back.jpg","left_image_path":"/abs/left.jpg"}'
```

Small disconnected specks (under 0.5% of the mesh) are removed automatically; pass
`"remove_floaters": false` in the params to keep everything.

## Installing on another machine

The extension is just this folder; it works on macOS (Apple Silicon/MPS) and
Windows/Linux (NVIDIA CUDA). `setup.py` picks the right PyTorch for the machine.

1. Copy this folder **without** its `venv/` into Modly's extensions directory, or use
   Modly's **Install from GitHub** with
   `https://github.com/open-southeners/modly-hunyuan3d-mv-extension` (then skip step 2).
   A copied folder must be named `hunyuan3d-mv`. Find the directory in Modly's settings, or
   with `modly config paths`.
2. With Modly running, build the extension's Python environment (takes several minutes;
   downloads PyTorch):

   ```
   curl -X POST http://127.0.0.1:8765/extensions/setup/hunyuan3d-mv
   curl -X POST http://127.0.0.1:8765/extensions/reload
   ```

   On Windows PowerShell, type `curl.exe` instead of `curl`.
3. In Modly's Models page, download **Hunyuan3D 2 MV** (about 4.9 GB, the standard
   variant only). If you skip this, the weights download on the first generation instead.

To update later, replace `manifest.json`, `generator.py`, `setup.py` and `tools/` and
reload; re-run setup only if `setup.py` changed.

## Requirements

- About 6 GB of VRAM on NVIDIA (fp16). On Apple Silicon it runs in fp32 on MPS; budget roughly
  12 GB+ of unified memory (an estimate, not measured).
- About 10 GB of disk: 4.9 GB weights plus the Python environment.

## Credits and licensing

- `setup.py` and the model loading / download code in `generator.py` are adapted from
  Lightning Pixel's official
  [Hunyuan3D 2 Mini extension](https://github.com/lightningpixel/modly-hunyuan3d-mini-extension).
  That repository has no licence, so those portions are not relicensed here.
- Everything else in this repository is MIT licensed; see [LICENSE](LICENSE) for the exact scope.
- The model weights and `hy3dgen` code, downloaded at runtime, are Tencent's under the
  [Tencent Hunyuan Community License](https://huggingface.co/tencent/Hunyuan3D-2mv/blob/main/LICENSE),
  which does not apply in the EU, the UK or South Korea.
