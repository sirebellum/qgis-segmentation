# QGIS Map Segmentation
A [Quant Civil](https://www.quantcivil.ai) product

![image](https://github.com/sirebellum/qgis-segmentation/assets/25124443/898b5b91-830f-47b1-9300-ca173fe093de)

Welcome!

Ever needed to intelligently divide your map into different sections? Have you been drawing polygons for days?
Not anymore! Introducing the machine learning powered qgis plugin that will change your workflow.

## Dependencies

Segmenter now ships with a lightweight dependency bootstrapper. When the plugin loads it checks for `torch`, `scikit-learn`, and `numpy` and, if they are missing, installs them into `<plugin>/vendor` using the Python interpreter that ships with QGIS. This keeps the QGIS installation untouched while ensuring the models can run.

On macOS the installer defaults to the CPU build of PyTorch. Windows and Linux systems default to the CUDA 12.1 build so that GPU acceleration remains available; set `SEGMENTER_TORCH_INDEX_URL` or `SEGMENTER_TORCH_SPEC` in the environment before starting QGIS to pin a different wheel. Set `SEGMENTER_SKIP_AUTO_INSTALL=1` if you prefer to manage dependencies yourself.

If QGIS embeds Python in a non-standard way, set `SEGMENTER_PYTHON` to the absolute path of the interpreter that should run `pip`.

While a model is running, the log panel in the dialog now streams live status updates (tiling progress, clustering stages, rendering) so you can monitor long operations.

Large rasters are processed with a YOLO-style overlapping sliding window. The plugin inspects the available memory on CUDA devices (using 0.9% of the free memory), MPS (0.75%), or CPU (1%) to pick a safe chunk size, then blends the overlapping predictions back together so you get a seamless output without exhausting GPU memory. Within each chunk the CNN tiles are now batched and prefetched to the GPU using the same memory budget, so utilization stays high without increasing peak RAM use.

On first launch, Segmenter profiles your local GPU to pick the best safety factor and prefetch depth for the batching logic. Results are cached in `perf_profile.json` inside the plugin so later runs reuse them instantly. Set `SEGMENTER_SKIP_PROFILING=1` if you prefer the default heuristics.

If you need to perform a manual install (for example on an offline machine), open the QGIS Python console and run:

```
import sys
import subprocess
subprocess.check_call([
	sys.executable,
	"-m",
	"pip",
	"install",
	"torch==2.2.2",
	"scikit-learn>=1.1,<2.0",
	"numpy>=1.23,<2.0",
	"--target",
	r"/path/to/segmenter/vendor",
])
```

Replace the `--target` path with the Segmenter plugin directory in your profile. Contact help@quantcivil.ai if you run into issues.

## Instructions
Below are the steps for a basic segmentation:
1. Choose a raster. Make sure the raster is in RGB format with 3 channels. The tool will process everything within the raster, so if you find your device running out of RAM, try using a smaller section of the raster or reducing the resolution. If you are using a web generated map (Google satelite), please convert the map to a raster first with the "Convert map to raster" tool available in QGIS.
2. Choose a model. CNN will provide better results but requires more processing power, see images below.
3. Choose a resolution. Higher resolutions will result in a more detailed segmentation map. Lower resolutions will be less noisy.
4. Choose number of segments. This will determine how many segments are generated. You should choose the number based on how "complex" the raster is. That is, if there 4 different kinds of land cover on your map (trees, roads, buildings, water, etc.), you should set the number of segments to something slightly higher than 4, between 6 and 8.
5. Segment! A raster layer will be produced overlaying the input with the different segments.

## Example Images
Below are some examples from the tool. First, a sample map, rendered at 0.1 map units per pixel:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/c3cdf14d-3717-4e39-ad98-71c7e457cc15">

Then, the K-Means segmentation map at low resolution with 6 segments:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/2bf88670-db9f-48dc-919d-4813f788a1cd">

Finally, a CNN segmentation map at low resolution with 6 segments:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/50863c0c-64b0-4603-b2e7-85ddaba60f21">

In general, CNN provides cleaner results while K-Means is quick and easy.
