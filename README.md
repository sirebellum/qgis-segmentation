# QGIS Map Segmentation
A [Quant Civil](https://www.quantcivil.ai) product

![image](https://github.com/sirebellum/qgis-segmentation/assets/25124443/898b5b91-830f-47b1-9300-ca173fe093de)

Welcome to the plugin description!

Ever needed to intelligently divide your map into different sections? Have you been drawing polygons for days?
Not anymore! Introducing the machine learning powered qgis plugin that will change your workflow.

## Dependencies

Segmenter now ships with a lightweight dependency bootstrapper for the numpy-only runtime. When the plugin loads it checks for `numpy` and, if it is missing, installs it into `<plugin>/vendor` using the Python interpreter that ships with QGIS. This keeps the QGIS installation untouched while ensuring the model can run.

Set `SEGMENTER_SKIP_AUTO_INSTALL=1` if you prefer to manage dependencies yourself. If QGIS embeds Python in a non-standard way, set `SEGMENTER_PYTHON` to the absolute path of the interpreter that should run `pip`.

If you need to perform a manual install (for example on an offline machine), open the QGIS Python console and run:

```
import sys
import subprocess
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "numpy>=1.23,<2.0",
    "--target",
    r"/path/to/segmenter/vendor",
])
```

Replace the `--target` path with the Segmenter plugin directory in your profile. Contact help@quantcivil.ai if you run into issues.

## Instructions
Below are the steps for a basic segmentation:
1. Choose a raster. Make sure the raster is in RGB format with 3 channels. The tool will process everything within the raster, so if you find your device running out of RAM, try using a smaller section of the raster or reducing the resolution. If you are using a web generated map (Google satelite), please convert the map to a raster first with the "Convert map to raster" tool available in QGIS.
2. Enter the number of segments. Choose a value that reflects the variety of land cover in view (e.g., 6–8 for roads/buildings/trees/water).
3. Segment! A raster layer will be produced overlaying the input with the different segments using the monolithic next-gen numpy model.

## Example Images
Below are some examples from the tool. First, a sample map, rendered at 0.1 map units per pixel:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/c3cdf14d-3717-4e39-ad98-71c7e457cc15">

Then, a segmentation map produced by the next-gen numpy model at low resolution with 6 segments:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/50863c0c-64b0-4603-b2e7-85ddaba60f21">

## Documentation
- Plugin architecture and runtime notes: [docs/plugin/ARCHITECTURE.md](docs/plugin/ARCHITECTURE.md), [docs/plugin/MODEL_NOTES.md](docs/plugin/MODEL_NOTES.md)
- Dataset preparation and manifests: [docs/dataset/DATASETS.md](docs/dataset/DATASETS.md)
- Training pipeline: [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) and [docs/training/MODEL_HISTORY.md](docs/training/MODEL_HISTORY.md)
- Change log and agent workflow: [docs/AGENTIC_HISTORY.md](docs/AGENTIC_HISTORY.md) and [docs/AGENTIC_REQUIRED.md](docs/AGENTIC_REQUIRED.md)
