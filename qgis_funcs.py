from osgeo import gdal
from tempfile import gettempdir
from qgis.core import QgsRasterLayer, QgsProject
import os

# Render raster from array
def render_raster(array, bounding_box, layer_name, epsg):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        os.path.join(gettempdir(), layer_name + ".tif"),
        array.shape[2],
        array.shape[1],
        array.shape[0],
        gdal.GDT_Byte,
    )

    out_srs = gdal.osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)

    dataset.SetProjection(out_srs.ExportToWkt())

    dataset.SetGeoTransform(
        (
            bounding_box.xMinimum(),  # 0
            bounding_box.width() / array.shape[2],  # 1
            0,  # 2
            bounding_box.yMaximum(),  # 3
            0,  # 4
            -bounding_box.height() / array.shape[1],
        )
    )

    for c in range(array.shape[0]):
        dataset.GetRasterBand(c + 1).WriteArray(array[c, :, :])
    dataset = None

    raster_layer = QgsRasterLayer(
        os.path.join(gettempdir(), layer_name + ".tif"), layer_name
    )
    raster_layer.renderer().setOpacity(1.0)

    QgsProject.instance().addMapLayer(raster_layer, True)
