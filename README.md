# QGIS Map Segmentation
A [Quant Civil](https://www.quantcivil.ai) product

![image](https://github.com/sirebellum/qgis-segmentation/assets/25124443/898b5b91-830f-47b1-9300-ca173fe093de)

Welcome!

Ever needed to intelligently divide your map into different sections? Have you been drawing polygons for days?
Not anymore! Introducing the machine learning powered qgis plugin that will change your workflow.

## License

Because machine learning takes time and compute power, this tool provides pre-trained models to accelerate your workflow. There is a mode that works without a license (K-Means), but a license is recommended for the most accurate results. To inquire about a license, please contact sales@quantcivil.ai.

## Dependencies

To install the dependencies required by the tool, open your python console in QGIS and enter the following
lines:

```
import pip
pip.main(["install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118"])
pip.main(["install", "scikit-learn", "opencv-python"])
```

If the above install does not work, or if you have other questions, please contact help@quantcivil.ai.

## Instructions
Below are the steps for a basic segmentation:
1. Choose a raster. Make sure the raster is in RGB format with 3 channels. The tool will process everything within the raster, so if you find your device running out of RAM, try using a smaller section of the raster or reducing the resolution. If you are using a web generated map (Google satelite), please convert the map to a raster first with the "Convert map to raster" tool available in QGIS.
2. Choose a model. There are two models available for the tool, K-Means and CNN. K-Means can be used for free while the CNN model requires a license. CNN will provide better results, see images below.
3. Choose a resolution. Higher resolutions will result in a more detailed segmentation map. Lower resolutions will be less noisy.
4. Choose number of segments. This will determine how many segments are generated. You should choose the number based on how "complex" the raster is. That is, if there 4 different kinds of land cover on your map (trees, roads, buildings, water, etc.), you should set the number of segments to something slightly higher than 4, between 6 and 8.
5. Segment! A raster layer will be produced overlaying the input with the different segments.
