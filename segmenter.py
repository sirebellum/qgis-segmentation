"""
/***************************************************************************
 Segmenter
                                 A QGIS plugin
 This plugin segments the map into discrete buckets
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-05-26
        git sha              : $Format:%H$
        copyright            : (C) 2023 by Quant Civil
        email                : joshua.herrera@quantcivil.ai
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import (
    QSettings,
    QTranslator,
    QCoreApplication,
    QThread,
)
from qgis.PyQt.QtGui import QIcon, QPixmap
from qgis.PyQt.QtWidgets import QAction

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .segmenter_dialog import SegmenterDialog
import os.path
from tempfile import gettempdir

from qgis.core import QgsApplication, QgsRasterLayer, QgsProject

from io import BytesIO
import requests
from osgeo import gdal

from .keygen import activate_license
import torch
from sklearn.cluster import KMeans
import numpy as np
import cv2

TILE_SIZE = 512
NUM_SEGMENTS = 32


class Segmenter:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value("locale/userLocale")[0:2]
        locale_path = os.path.join(
            self.plugin_dir, "i18n", "Segmenter_{}.qm".format(locale)
        )

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr("&Map Segmenter")

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

        QSettings().setValue("/qgis/parallel_rendering", True)
        threadcount = QThread.idealThreadCount()
        QgsApplication.setMaxThreads(threadcount)

        self.keygen_account_id = "ae9bc51d-ae1c-482e-9223-c4243cd7e434"

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate("Segmenter", message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None,
    ):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ":/plugins/segmenter/icon.png"
        self.add_action(
            icon_path,
            text=self.tr("Segment the map"),
            callback=self.run,
            parent=self.iface.mainWindow(),
        )

        # will be set False in run()
        self.first_start = True

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr("&Map Segmenter"), action)
            self.iface.removeToolBarIcon(action)

    # Predict coverage map using kmeans
    def predict_kmeans(self, array, num_segments=16, resolution=16):

        # Instantiate kmeans model
        kmeans = KMeans(n_clusters=num_segments)

        # Pad to resolution
        channel_pad = (0, 0)
        height_pad = (0, array.shape[1] % resolution)
        width_pad = (0, array.shape[2] % resolution)
        array_padded = np.pad(array, (channel_pad, height_pad, width_pad), mode="constant")

        # Reshape into 2d
        array_2d = array_padded.reshape(
            array_padded.shape[0],
            array_padded.shape[1] // resolution,
            resolution,
            array_padded.shape[2] // resolution,
            resolution,
        )
        array_2d = array_2d.transpose(1, 3, 0, 2, 4)
        array_2d = array_2d.reshape(
            array_2d.shape[0] * array_2d.shape[1],
            array_2d.shape[2] * resolution * resolution,
        )

        # Fit kmeans model to random subset
        size = 10000 if array_2d.shape[0] > 10000 else array_2d.shape[0]
        idx = np.random.randint(0, array_2d.shape[0], size=size)
        kmeans = kmeans.fit(array_2d[idx])

        # Get clusters
        clusters = kmeans.predict(array_2d)

        # Reshape clusters to match map
        clusters = clusters.reshape(
            1,
            1,
            array.shape[1] // resolution,
            array.shape[2] // resolution,
        )

        # Get rid of padding
        clusters = clusters[:, :, :array.shape[1] // resolution, :array.shape[2] // resolution]

        # Upsample to original size
        clusters = torch.tensor(clusters).to(self.device)
        clusters = torch.nn.Upsample(
            size=(array.shape[-2], array.shape[-1]), mode="nearest"
        )(clusters.byte())
        clusters = clusters[0]

        return clusters.cpu().numpy()

    # Predict coverage map using cnn
    def predict_cnn(self, array, num_segments, resolution):

        # Print message about gpu
        if self.device == torch.device("cpu"):
            self.dlg.inputBox.setPlainText("WARNING: GPU not available. Using CPU instead.")

        assert array.shape[0] == 3, f"Invalid array shape! \n{array.shape}"

        # Download and load model
        model_bytes = self.keygen_model(
            resolution,
            self.key
        )
        cnn_model = torch.jit.load(model_bytes).to(self.device)
        cnn_model.eval()

        # Pad to tile_size
        channel_pad = (0, 0)
        height_pad = (0, TILE_SIZE - array.shape[1] % TILE_SIZE)
        width_pad = (0, TILE_SIZE - array.shape[2] % TILE_SIZE)
        array_padded = np.pad(
            array,
            (channel_pad, height_pad, width_pad),
            mode="constant",
        )

        # Reshape into tiles
        tiles = array_padded.reshape(
            3,
            array_padded.shape[1] // TILE_SIZE,
            TILE_SIZE,
            array_padded.shape[2] // TILE_SIZE,
            TILE_SIZE,
        )
        tiles = tiles.transpose(1, 3, 0, 2, 4)
        tiles = tiles.reshape(
            tiles.shape[0] * tiles.shape[1],
            3,
            TILE_SIZE,
            TILE_SIZE,
        )

        # Convert to float range [0, 1]
        tiles = tiles.astype("float32") / 255

        # Convert to torch
        tiles = torch.from_numpy(tiles).to(self.device)

        # Predict vectors
        batch_size = 1
        coverage_map = []
        for i in range(0, tiles.shape[0], batch_size):
            with torch.no_grad():
                vectors = cnn_model.forward(tiles[i:i+batch_size])
            coverage_map.append(vectors)
        coverage_map = torch.concatenate(coverage_map, dim=0)
        coverage_map = coverage_map.cpu().numpy()

        # Convert from tiles to one big map
        coverage_map = coverage_map.reshape(
            array_padded.shape[1] // TILE_SIZE,
            array_padded.shape[2] // TILE_SIZE,
            coverage_map.shape[1],
            coverage_map.shape[2],
            coverage_map.shape[3],
        )
        coverage_map = coverage_map.transpose(2, 0, 3 ,1, 4)
        coverage_map = coverage_map.reshape(
            coverage_map.shape[0],
            coverage_map.shape[1] * coverage_map.shape[2],
            coverage_map.shape[3] * coverage_map.shape[4],
        )

        # Perform kmeans to get num_segments clusters
        coverage_map = self.predict_kmeans(
            coverage_map,
            num_segments=num_segments,
            resolution=1,
        )

        # Upsample
        coverage_map = torch.tensor(coverage_map).to(self.device)
        coverage_map = torch.unsqueeze(coverage_map, dim=0)
        coverage_map = torch.nn.Upsample(size=(array_padded.shape[-2], array_padded.shape[-1]), mode="nearest")(coverage_map)

        # Get rid of padding
        coverage_map = coverage_map[0, :, :array.shape[1], :array.shape[2]]

        return coverage_map.cpu().numpy()

    # Render raster from array
    def render_raster(self, array, bounding_box, layer_name):
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            os.path.join(gettempdir(), layer_name+".tif"),
            array.shape[2],
            array.shape[1],
            array.shape[0],
            gdal.GDT_Byte,
        )

        out_srs = gdal.osr.SpatialReference()
        out_srs.ImportFromEPSG(self.canvas.layer(0).crs().postgisSrid())

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
            dataset.GetRasterBand(c + 1).WriteArray(
                array[c, :, :]
            )
        dataset = None

        raster_layer = QgsRasterLayer(
            os.path.join(gettempdir(), layer_name+".tif"), layer_name
        )
        raster_layer.renderer().setOpacity(0.5)

        QgsProject.instance().addMapLayer(raster_layer, True)

    # Predict coverage map
    def predict(self):
        # Load user specified raster
        layer_name = self.dlg.inputLayer.currentText()
        layer = QgsProject.instance().mapLayersByName(layer_name)[0]
        assert layer.isValid(), f"Invalid raster layer! \n{layer_name}"
        raster = gdal.Open(layer.source())
        layer_array = raster.ReadAsArray()

        # Get user specified num segments
        num_segments = int(self.dlg.inputSegments.text())

        resolution_map = {
            "very high": 2,
            "high": 4,
            "medium": 8,
            "low": 16,
        }

        # Get user specified resolution
        resolution = resolution_map[self.dlg.inputRes.currentText()]

        # Perform prediction based on model selected
        if self.model == "kmeans":
            coverage_map = self.predict_kmeans(
                layer_array,
                num_segments=num_segments,
                resolution=resolution,
            )
        elif self.model == "cnn":
            coverage_map = self.predict_cnn(
                layer_array,
                num_segments=num_segments,
                resolution=self.dlg.inputRes.currentText(),
            )
        
        # Render coverage map
        self.render_raster(
            coverage_map,
            layer.extent(),
            f"{layer_name}_{self.model}_{num_segments}_{resolution}",
        )

        # Update message
        self.dlg.inputBox.setPlainText("Segmenting map...")

    # Download model from keygen
    def keygen_model(self, model_name, key):
        # Get license activation token
        url = "https://api.keygen.sh/v1/accounts/{}/me".format(self.keygen_account_id)
        headers = {
            "Authorization": "License {}".format(key),
            "Accept": "application/vnd.api+json",
        }
        response = requests.get(url, headers=headers).json()
        token = response["data"]["attributes"]["metadata"]["token"]
        if "data" not in response.keys():
            raise ValueError(f"Invalid key! \n{response}")

        # Get redirect url for artifact
        url = "https://api.keygen.sh/v1/accounts/{}/artifacts/{}".format(
            self.keygen_account_id, model_name
        )
        headers = {
            "Authorization": "Bearer {}".format(token),
            "Accept": "application/vnd.api+json",
        }
        response = requests.get(url, headers=headers, allow_redirects=False).json()
        if "data" not in response.keys():
            raise ValueError(f"Invalid response from model server! \n{response}")
        redirect = response["data"]["links"]["redirect"]

        file_data = requests.get(redirect).content

        return BytesIO(file_data)

    # Process user input box
    def submit(self):
        key = self.dlg.inputBox.toPlainText()
        license_key = os.path.join(self.plugin_dir, "qgis_key")

        # Process special input
        if key == "delete_key":
            os.remove(license_key)
            self.dlg.inputBox.setPlainText("Key deleted!")
            return

        # Make sure entered key looks like a key
        if len(key.split("-")) != 4:
            return
        if len(key.split("-")[-1]) < 4:
            return

        # Check existing key
        if os.path.exists(license_key):
            with open(license_key, "r") as f:
                file_key = f.readline()
            if file_key != key:
                os.remove(license_key)

        # activate license
        activated, msg = activate_license(key, self.keygen_account_id)
        if activated:
            with open(license_key, "x") as f:
                f.write(key)
            self.dlg.inputBox.setPlainText(f"Valid: {msg}")
            self.key = key
        else:
            self.dlg.inputBox.setPlainText(f"Invalid: {msg}")
            self.key = "nokey"

    #  Display models in dropdown
    def render_models(self):
        model_list = ["K-Means", "CNN"]
        self.dlg.inputLoadModel.clear()
        for model in model_list:
            self.dlg.inputLoadModel.addItem(model)

    # Display layers in dropdown
    def render_layers(self):
        layer_list = [layer.name() for layer in self.canvas.layers()]
        self.dlg.inputLayer.clear()
        for layer in layer_list:
            self.dlg.inputLayer.addItem(layer)

    # Display resolutions in dropdown
    def render_resolutions(self):
        res_list = ["very high", "high", "medium", "low"]
        self.dlg.inputRes.clear()
        for res in res_list:
            self.dlg.inputRes.addItem(str(res))

    # Set model based on selected dropdown
    def set_model(self):
        model = self.dlg.inputLoadModel.currentText()
        if model == "K-Means":
            self.model = "kmeans"
        elif model == "CNN":
            self.model = "cnn"

    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = SegmenterDialog()
            self.canvas = self.iface.mapCanvas()

            # use gpu if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            # Populate drop down menus
            self.render_models()
            self.render_layers()
            self.render_resolutions()

            # Set gpu message
            gpu_msg = "GPU available."
            if self.device == torch.device("cpu"):
                gpu_msg = "GPU not available. Using CPU instead."

            # Set up license
            self.license_path = os.path.join(self.plugin_dir, "qgis_key")
            self.key = "nokey"
            if not os.path.exists(self.license_path):
                self.dlg.inputBox.setPlainText(f"Please input key with dashes\n{gpu_msg}")
            else:
                # Check license
                with open(self.license_path, "r") as f:
                    self.key = f.readline()
                act, msg = activate_license(self.key, self.keygen_account_id)
                if not act:
                    os.remove(self.license_path)
                    self.key = "nokey"
                self.dlg.inputBox.setPlainText(f"{msg}\n{gpu_msg}")

            # Attach inputs
            self.dlg.inputBox.textChanged.connect(self.submit)
            self.dlg.buttonPredict.clicked.connect(self.predict)
            self.dlg.inputLoadModel.currentIndexChanged.connect(self.set_model)

            # Render logo
            img_path = os.path.join(self.plugin_dir, "logo.png")
            pix = QPixmap(img_path)
            self.dlg.imageLarge.setPixmap(pix)

        # show the dialog
        self.render_layers()
        self.dlg.show()
