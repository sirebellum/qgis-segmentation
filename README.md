# QGIS Map Segmentation
A [Quant Civil](https://www.quantcivil.ai) product

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

If you are using this tool for free without a license, an nvidia gpu is highly recommended.
To install the dependencies required by the tool, open your python console in QGIS and enter the following
lines:

```
import pip
pip.main(["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"])
pip.main(["install", "scikit-learn", "opencv-python", "qimage2ndarray"])
```

If the above install does not work, or if you have other questions, please contact help@quantcivil.ai.

## Instructions
There are two "modes" in the tool, training and segmenting. Training is for tuning a model to satelite imagery that you provide, while segmenting uses that trained model to produce a map of the different types of land cover.
Below are the steps for a basic segmentation:
1. Load a map into qgis, the tool will process everything within the canvas extent (everything you can see).
2. Choose an area slightly larger than the area that you want to segment. This is for training.
3. Choose a training time and resolution. Training time dictates how "good" the model is allowed to get. Resolution dictates how big the pixels are in the final raster map.
4. (Optional) Choose a pretrained model based on your chosen resolution.
5. (Optional) Hit the offload button to send the job to a server with a dedicated gpu and lots of ram. It'll get the job done in no time, and it'll even report its status if you hit refresh!
6. Hit train and watch the magic happen. Depending on your chosen settings and the size of the area you are processing, it can take anywhere from 5 minutes to several days to complete. As a general rule of thumb, training will take a lot longer than segmenting. You will be limited by your available RAM here. The tool will not let you exceed your available ram. NOTE: if you are running without a gpu, your training times will be upwards of weeks. It is NOT RECOMMENDED to use this tool without a gpu. Like mentioned earlier, servers are available to offload your jobs if you find your computer can not process maps quickly enough.
7. If you are training locally, the image in the toolbox will show the status of the model. If the image looks like a piece of the map, it's doing a good job. If it looks like a solid color or some weird pattern, something is wrong! Please contact us if this occurs. If you are training in offload mode, the status bar will display an accurate status of the process on the remote server. Once it is done, pressing the refresh button will download the result.
8. Once trained, you can segment any area of the map using the segment button. It will produce a raster of the visible canvas extent. Use the Num Segments box to indicate to the tool how many different types of landcover you expect in the image. The tool isn't perfect, so it may take some tinkering with this value to get something usable. A value somewhere in the range 3-9 is typically most useful.
9. (Optional) Save your model so you can use it later.

Below are some screenshots of the tool's handywork. The settings used were high+ resolution with 6 segments. The model was trained for two days on 200 square miles of data.

<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/d4e6555b-b3aa-4138-98e2-f9f25a97dca4">
<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/4ba5ce87-bc01-4323-aa1a-11445e9b42df">


Here is the same area processed by a model of the same settings, but the model has only been trained for 5 minutes on this area specifically. Pretty good!

<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/df85ddcc-89c2-4425-860e-96e4716f6e1d">
<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/4ba5ce87-bc01-4323-aa1a-11445e9b42df">


However, when we apply the model trained for 5 minutes to a new map area...

<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/c8e64df2-fb37-4147-8759-dc416bd77d69">
<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/de9a869c-0d92-42b6-90b8-3da2c4635349">


It doesn't generalize well outside of the area it was trained in. If we fine-tune for another 5 minutes on this area though...

<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/669f5305-1922-43d2-9b91-4c7cf645ebc6">
<img width="400" alt="image" src="https://github.com/sirebellum/qgis-segmentation/assets/25124443/de9a869c-0d92-42b6-90b8-3da2c4635349">


Fine-tuning the model for your area of the map will always render the best results. The models available for download are only a best-guess at segmentation.
