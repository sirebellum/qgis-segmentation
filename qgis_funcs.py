from osgeo import gdal
from tempfile import gettempdir
from qgis.core import QgsRasterLayer, QgsProject
import os

try:
    from .raster_utils import ensure_channel_first
except ImportError:  # pragma: no cover
    from raster_utils import ensure_channel_first


def render_raster(array, bounding_box, layer_name, epsg):
    """Render raster from array."""
    driver = gdal.GetDriverByName("GTiff")
    array = ensure_channel_first(array)
    channels, height, width = array.shape

    dataset = driver.Create(
        os.path.join(gettempdir(), layer_name + ".tif"),
        width,
        height,
        channels,
        gdal.GDT_Byte,
    )

    out_srs = gdal.osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)

    dataset.SetProjection(out_srs.ExportToWkt())

    dataset.SetGeoTransform(
        (
            bounding_box.xMinimum(),  # 0
            bounding_box.width() / width,  # 1
            0,  # 2
            bounding_box.yMaximum(),  # 3
            0,  # 4
            -bounding_box.height() / height,
        )
    )

    for c in range(channels):
        dataset.GetRasterBand(c + 1).WriteArray(array[c, :, :])
    dataset = None

    raster_layer = QgsRasterLayer(
        os.path.join(gettempdir(), layer_name + ".tif"), layer_name
    )
    raster_layer.renderer().setOpacity(1.0)

    QgsProject.instance().addMapLayer(raster_layer, True)
