# QGIS Map Segmentation
A [Quant Civil](https://www.quantcivil.ai) product

![image](https://github.com/sirebellum/qgis-segmentation/assets/25124443/898b5b91-830f-47b1-9300-ca173fe093de)

Welcome!

Ever needed to intelligently divide your map into different sections? Have you been drawing polygons for days?
Not anymore! Introducing the algorithmic qgis plugin that will change your land coverage workflow.

## Dependencies

Land Coverage Calculator now ships with a lightweight dependency bootstrapper. When the plugin loads it checks for `torch` and `numpy` and, if they are missing, installs them into `<plugin>/vendor` using the Python interpreter that ships with QGIS. This keeps the QGIS installation untouched while ensuring the land coverage algorithm can run.

The default PyTorch spec is chosen automatically based on the Python runtime that QGIS embeds (Python 3.13 gets a `torch>=2.5.1,<3.0` constraint, Python 3.12 uses `torch>=2.3.1,<3.0`, and older versions stay on `torch==2.2.2`). Override this behaviour anytime with `SEGMENTER_TORCH_SPEC="torch==<version>"` if you need to pin a particular wheel.

On macOS we also set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before loading PyTorch to avoid the `libomp` duplication crash that can occur when the plugin spins up OpenMP workloads inside the QGIS process.

If QGIS embeds Python in a non-standard way, set `SEGMENTER_PYTHON` to the absolute path of the interpreter that should run `pip`.

While the algorithm is running, the log panel in the dialog now streams live status updates (tiling progress, clustering stages, rendering) so you can monitor long operations.

If you need to perform a manual install (for example on an offline machine), open the QGIS Python console and run:

```
import sys
import subprocess
subprocess.check_call([
	sys.executable,
	"-m",
	"pip",
	"install",
	"torch>=2.5.1,<3.0",
	"numpy>=1.23,<2.0",
	"--target",
	r"/path/to/segmenter/vendor",
])
```

Replace the `--target` path with the plugin directory in your profile. Contact joshua.herrera227@gmail.com if you run into issues.

## Input Types

The segmenter requires a **3-band RGB GeoTIFF raster** as input. The plugin supports two input workflows:

### Raster Input (Direct)
Select a local 3-band GeoTIFF raster layer directly. The tool will process the entire raster using K-Means segmentation.

### Map/Web Service Input (Assisted Conversion)
If you select a map service layer (WMS, WMTS, XYZ tiles, ArcGIS services) or a vector layer, the plugin automatically opens the **Convert map to raster** processing dialog with:
- **Extent**: current map canvas extent
- **Layer**: the selected layer
- **Resolution**: 1 map unit per pixel

Adjust the settings as needed and run the conversion. The output GeoTIFF will be added to your project—select it as input for segmentation.

## Instructions
Below are the steps for a basic segmentation:
1. Choose a layer. If you select a 3-band GeoTIFF raster, segmentation can proceed directly. If you select a web map or vector layer, a conversion dialog will appear to help you create a compatible raster.
2. Choose a resolution. Higher resolutions will result in a more detailed segmentation map. Lower resolutions will be less noisy.
3. Choose number of coverage bins. This will determine how many segments are generated. You should choose the number based on how "complex" the raster is. That is, if there 4 different kinds of land cover on your map (trees, roads, buildings, water, etc.), you should set the number of segments to something slightly higher than 4, between 6 and 8.
4. Segment! A raster layer will be produced overlaying the input with the different segments.

## Example Images
Below are some examples from the tool. Here is a sample map, rendered at 1.0 map units per pixel:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/c3cdf14d-3717-4e39-ad98-71c7e457cc15">

The land coverage map at high resolution with 4 segments:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/2bf88670-db9f-48dc-919d-4813f788a1cd">

The land coverage map at high resolution with 8 segments:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/2bf88670-db9f-48dc-919d-4813f788a1cd">
