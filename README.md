# QGIS Segmentation
A Quant Civil product

![image](https://github.com/sirebellum/qgis-segmentation/assets/25124443/898b5b91-830f-47b1-9300-ca173fe093de)

Welcome! This tool is currently under construction and functionality is subject to change.

Ever needed to intelligently divide your map into different sections? Have you been drawing polygons for days?
Not anymore! Introducing the machine learning powered qgis plugin that will change your workflow.

This tool is designed to facilitate segmenting a map into different areas. It is purely a visual tool.
This means that if two areas of the map "look" similar enough, it will classify them as the same.
If you see this occuring in practice, try training on the area of interest. Training may allow
the model to pick up on subtle differences that the default model does not.

This tool currently produces a raster layer. Vector layers tend to be easier to work with, so feel free
to convert to a vector layer via raster > convert > polyganize in QGIS. From the vector, you can then
bucket each layer in the attributes.

## License

Because machine learning takes time and compute power, this tool allows the user to offload training
and prediction to remote servers. There are also pre-trained models available for quick out-of-the-box
segmentation. However, because of the aforementioned compute power requirements, these features require
a license. To inquire about a license, please contact sales@quantcivil.ai.

## Dependencies

If you are using this tool for free without a license, an nvidia gpu is highly recommended (required).
To install the dependencies required by the tool, open your python console in QGIS and enter the following
lines:

```
import pip
pip.main(["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"])
pip.main(["install", "scikit-learn", "opencv-python", "qimage2ndarray"])
```

If the above install does not work, or if you have other questions, please contact help@quantcivil,ai.

Below are some screenshots of the tool's handywork. Enjoy!

<img width="1000" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/051d0f14-ea68-4d30-8578-cedf2f7487a1">
<img width="1000" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/80971a81-71f6-4561-a9a7-c42187efc9ab">
<img width="1000" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/0b86f99e-08ef-4163-93e5-83b55333a3a5">
