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
    QEventLoop,
    QSize,
    QThread,
    QObject,
    pyqtSignal,
)
from qgis.PyQt.QtGui import QIcon, QPixmap
from qgis.PyQt.QtWidgets import QAction

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .segmenter_dialog import SegmenterDialog
import os.path
from math import floor

from qgis.core import (
    QgsProject,
    QgsRectangle,
    QgsPointXY,
    QgsRasterLayer,
    QgsApplication,
    QgsRasterBlock,
    Qgis,
    QgsMapRendererParallelJob,
)

from pathlib import Path
from io import BytesIO
import requests
from hashlib import sha256
from osgeo import gdal as osgdal
import tempfile
import socket
import pickle
import json

import qimage2ndarray
from sklearn import cluster
import cv2
import numpy as np
from glob import glob

from .model import AE
from .keygen import activate_license
import torch

class Trainer(QObject):
    finished = pyqtSignal()

    def __init__(
        self, model, epochs, data, tile_size, pixel_size, y_tiles, x_tiles, device, key
    ):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.data = data
        self.tile_size = tile_size
        self.pixel_size = pixel_size
        self.y_tiles = y_tiles
        self.x_tiles = x_tiles
        self.device = device
        self.key = key
        self.ip = "0.0.0.0"

    def run(self):
        # Format package
        package = {}

        # Create image for transmission
        image_data = np.empty(
            (self.tile_size * self.y_tiles, self.tile_size * self.x_tiles, 3),
            dtype="uint8",
        )
        for yy in range(self.y_tiles):
            for xx in range(self.x_tiles):
                image_data[
                    yy * self.tile_size : yy * self.tile_size + self.tile_size,
                    xx * self.tile_size : xx * self.tile_size + self.tile_size,
                ] = self.data[xx + yy * self.x_tiles]
        data_buffer = cv2.imencode(".jpg", image_data)[1]
        package["data"] = data_buffer.tobytes().decode("latin-1")

        # Rest of the stuff
        model_buffer = BytesIO()
        torch.save(self.model.state_dict(), model_buffer)
        package["model_state"] = model_buffer.getvalue().decode("latin-1")
        package["epochs"] = self.epochs
        package["tile_size"] = self.tile_size
        package["x_tiles"] = self.x_tiles
        package["y_tiles"] = self.y_tiles
        package["pixel_size"] = self.pixel_size
        package["key"] = self.key

        url = "http://qgis.quantcivil.ai:5000/train"
        response = requests.post(url, json=package, timeout=1e6)
        if response.status_code == 200:
            self.ip = response.text
        else:
            self.ip = -1

        self.finished.emit()


class Predictor(QObject):
    finished = pyqtSignal()

    def __init__(
        self,
        model,
        data,
        tile_size,
        pixel_size,
        y_tiles,
        x_tiles,
        device,
        clusters,
        key,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.tile_size = tile_size
        self.pixel_size = pixel_size
        self.y_tiles = y_tiles
        self.x_tiles = x_tiles
        self.device = device
        self.clusters = clusters
        self.key = key
        self.ip = "0.0.0.0"

    def run(self):
        # Format package
        package = {}

        # Create image for transmission
        image_data = np.empty(
            (self.tile_size * self.y_tiles, self.tile_size * self.x_tiles, 3),
            dtype="uint8",
        )
        for yy in range(self.y_tiles):
            for xx in range(self.x_tiles):
                image_data[
                    yy * self.tile_size : yy * self.tile_size + self.tile_size,
                    xx * self.tile_size : xx * self.tile_size + self.tile_size,
                ] = self.data[yy, xx]
        data_buffer = cv2.imencode(".jpg", image_data)[1]
        package["data"] = data_buffer.tobytes().decode("latin-1")

        # Rest of the stuff
        model_buffer = BytesIO()
        torch.save(self.model.state_dict(), model_buffer)
        package["model_state"] = model_buffer.getvalue().decode("latin-1")
        package["tile_size"] = self.tile_size
        package["x_tiles"] = self.x_tiles
        package["y_tiles"] = self.y_tiles
        package["pixel_size"] = self.pixel_size
        package["clusters"] = self.clusters
        package["key"] = self.key

        url = "http://qgis.quantcivil.ai:5000/predict"
        response = requests.post(url, json=package, timeout=1e6)
        if response.status_code == 200:
            self.ip = response.text
        else:
            self.ip = -1

        self.finished.emit()


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

    def get_input(self, bounding_box):
        settings = self.iface.mapCanvas().mapSettings()
        settings.setExtent(bounding_box)
        settings.setOutputSize(QSize(self.tile_size, self.tile_size))
        renderer = QgsMapRendererParallelJob(settings)

        event_loop = QEventLoop()
        renderer.finished.connect(event_loop.quit)
        renderer.start()
        event_loop.exec_()

        img = renderer.renderedImage()
        img = qimage2ndarray.rgb_view(img)
        img = cv2.resize(img, (self.tile_size, self.tile_size))
        return img

    def finished_remote_train(self):

        if self.trainer.ip == -1:
            self.dlg.inputKey.setPlainText(f"Bees are busy, try again in a moment.")
        else:
            self.dlg.inputKey.setPlainText(f"Finished uploading to {self.trainer.ip}!")
        self.tthread = None

    def update_progress(self):

        if self.worker is None:
            self.dlg.inputKey.setPlainText("Nothing offloaded, try offloading a job.")
            return

        if self.worker.ip == "0.0.0.0":
            self.dlg.inputKey.setPlainText("Hold your horses 🐴\nWe're still uploading.")
            return
        elif self.worker.ip == -1:
            self.dlg.inputKey.setPlainText("Bees are busy, try again in a moment.")
            self.worker = None
            return

        url = "http://qgis.quantcivil.ai:5000/watcher/" + self.worker.ip
        progress = int(requests.get(url).text)
        self.dlg.progressBar.setValue(progress)

        # Fun little progress updates
        if progress == 0:
            self.dlg.inputKey.setPlainText("Annihilating the launchpad 🚀")
        elif progress < 10:
            self.dlg.inputKey.setPlainText("Fighting gravity 🍎")
        elif progress < 20:
            self.dlg.inputKey.setPlainText("Avoiding a deathspiral 💀")
        elif progress < 30:
            self.dlg.inputKey.setPlainText("Entering low earth orbit 🌎")
        elif progress < 40:
            self.dlg.inputKey.setPlainText("Readying the bits 👩‍💻")
        elif progress < 50:
            self.dlg.inputKey.setPlainText("Performing experiments 🔬")
        elif progress < 60:
            self.dlg.inputKey.setPlainText("Fighting thermal fallout ☢")
        elif progress < 70:
            self.dlg.inputKey.setPlainText("Crunching numbers 🔢")
        elif progress < 80:
            self.dlg.inputKey.setPlainText("Packing up 📦")
        elif progress < 90:
            self.dlg.inputKey.setPlainText("Beaming down 🖖")
        elif progress < 100:
            self.dlg.inputKey.setPlainText("Wrapping up 🎁")
        else:
            self.dlg.inputKey.setPlainText("The eagle has landed 🦅")

        if progress == 100:
            url = "http://qgis.quantcivil.ai:5000/payload/" + self.worker.ip
            payload = requests.get(url)
            if payload.status_code == 400:
                self.dlg.inputKey.setPlainText("Error, try again.")
                self.worker.ip = -1
                return
            elif payload.status_code == 201:  # model
                state_dict = torch.load(
                    BytesIO(payload.text.encode("latin-1")),
                    weights_only=True,
                    map_location=self.device,
                )
                self.set_model(state_dict)
                self.dlg.inputKey.setPlainText("Mission success ✅\nModel loaded")
            elif payload.status_code == 200:  # raster
                data = np.asarray(
                    bytearray(payload.text.encode("latin-1")), dtype="uint8"
                )
                data = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                self.write_raster_layer(data, self.bounding_box)
                self.dlg.inputKey.setPlainText("Mission success ✅\nRaster loaded")

            self.worker.ip == -1

    def train(self):
        time_enum = {
            "very short": 5,
            "short": 25,
            "medium": 50,
            "long": 100,
            "very long": 500,
        }
        message = ""
        if self.device == torch.device("cpu"):
            message = "WARNING: no gpu!"

        # Wait for downloading model
        if self.dlg.inputUseServer.isChecked():
            if self.tthread is not None:
                self.dlg.inputKey.setPlainText("Still training remotely...")
                return
            elif self.pthread is not None:
                self.dlg.inputKey.setPlainText("Waiting on remote prediction...")
                return

        # Get number of tiles to train on
        y_tiles = int(self.canvas.extent().height() / self.processing_scale)
        x_tiles = int(self.canvas.extent().width() / self.processing_scale)

        # Set up model stuff
        epochs = time_enum[self.dlg.inputTrainingTime.currentText()]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.L1Loss()

        # Render and tile map
        self.dlg.inputKey.setPlainText("Rendering map...\nPlease don't move the canvas.")
        map_tiles = np.empty(
            (y_tiles * x_tiles, self.tile_size, self.tile_size, 3), dtype="uint8"
        )
        for y_tile in range(y_tiles):
            self.dlg.progressBar.setValue(100 * y_tile // y_tiles)
            for x_tile in range(x_tiles):
                x = self.canvas.extent().xMinimum() + x_tile * self.processing_scale
                y = self.canvas.extent().yMaximum() - y_tile * self.processing_scale
                upper_left = QgsPointXY(x, y)
                x = x + self.processing_scale
                y = y - self.processing_scale
                lower_right = QgsPointXY(x, y)
                bounding_box = QgsRectangle(upper_left, lower_right)
                map_tiles[y_tile * x_tiles + x_tile] = self.get_input(bounding_box)
        self.dlg.progressBar.setValue(100)

        # If client requests online training
        if self.dlg.inputUseServer.isChecked():
            if self.key == "nokey":
                self.dlg.inputKey.setPlainText(
                    "Offload feature requires key\nPlease input key with dashes."
                )
                return
            if self.tthread is None:
                self.dlg.inputKey.setPlainText("Uploading assets...")
                self.tthread = QThread()
                self.trainer = Trainer(
                    self.model,
                    epochs,
                    map_tiles,
                    self.tile_size,
                    self.pixel_size,
                    y_tiles,
                    x_tiles,
                    self.device,
                    self.key,
                )
                self.trainer.moveToThread(self.tthread)
                self.worker = self.trainer
                self.tthread.started.connect(self.trainer.run)
                self.trainer.finished.connect(self.tthread.quit)
                self.trainer.finished.connect(self.trainer.deleteLater)
                self.tthread.finished.connect(self.tthread.deleteLater)
                self.tthread.finished.connect(self.finished_remote_train)
                self.tthread.start()
                return

        # Offline training
        self.dlg.inputKey.setPlainText("Training... {}".format(message))
        map_tiles = np.swapaxes(map_tiles, -1, -2)
        map_tiles = np.swapaxes(map_tiles, -2, -3)
        for e in range(epochs):
            self.dlg.progressBar.setValue(100 * e // epochs)

            # Shuffle inputs
            np.random.shuffle(map_tiles)

            # Do training
            batch_size = 1
            for bb in range(0, y_tiles * x_tiles, batch_size):
                batch = map_tiles[bb : bb + batch_size]

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                tensor_in = (
                    torch.tensor(batch, dtype=torch.float32).to(self.device) / 255
                )
                outputs = self.model(tensor_in)

                # compute training reconstruction loss
                train_loss = criterion(outputs, tensor_in)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

            # Display example output
            example = outputs.cpu().detach().numpy()[0] * 255
            example = np.swapaxes(np.swapaxes(example, -3, -2), -2, -1)
            width = self.dlg.imageTraining.width()
            height = self.dlg.imageTraining.height()
            example = cv2.resize(example, (width, height))
            img_path = os.path.join(tempfile.gettempdir(), "qgis_img.jpg")
            cv2.imwrite(img_path, example)
            pix = QPixmap(img_path)
            self.dlg.imageTraining.setPixmap(pix)

        # Transfer weights to encoder
        self.encoder.load_state_dict(self.model.state_dict())

        self.dlg.progressBar.setValue(100)
        self.dlg.inputKey.setPlainText("Model trained!")

        return

    def write_raster_layer(self, raster_data, bounding_box):
        channels = 1
        raster_data = np.expand_dims(raster_data, axis=-1)

        driver = osgdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            os.path.join(tempfile.gettempdir(), "seg.tif"),
            raster_data.shape[1],
            raster_data.shape[0],
            channels,
            osgdal.GDT_Byte,
        )

        out_srs = osgdal.osr.SpatialReference()
        out_srs.ImportFromEPSG(self.canvas.layer(0).crs().postgisSrid())

        dataset.SetProjection(out_srs.ExportToWkt())

        dataset.SetGeoTransform(
            (
                bounding_box.xMinimum(),  # 0
                bounding_box.width() / raster_data.shape[1],  # 1
                0,  # 2
                bounding_box.yMaximum(),  # 3
                0,  # 4
                -bounding_box.height() / raster_data.shape[0],
            )
        )

        for c in range(channels):
            dataset.GetRasterBand(c + 1).WriteArray(
                raster_data[:, :, c]
            )  # Remove "T" if it's inverted.
        dataset = None

        raster_layer = QgsRasterLayer(
            os.path.join(tempfile.gettempdir(), "seg.tif"), "Segmentation map"
        )
        raster_layer.renderer().setOpacity(0.5)

        QgsProject.instance().addMapLayer(raster_layer, True)

    def finished_remote_predict(self):
        self.dlg.inputKey.setPlainText(f"Finished uploading to {self.predictor.ip}!")
        self.pthread = None

    def predict(self):
        message = ""
        if self.device == torch.device("cpu"):
            message = "WARNING: no gpu!"
        clusters = int(self.dlg.inputSegments.text())

        # Wait for other remote processes
        if self.dlg.inputUseServer.isChecked():
            if self.tthread is not None:
                self.dlg.inputKey.setPlainText("Waiting on remote training...")
                return
            elif self.pthread is not None:
                self.dlg.inputKey.setPlainText("Still predicting remotely...")
                return

        y_tiles = floor(self.canvas.extent().height() / self.processing_scale) + 1
        x_tiles = floor(self.canvas.extent().width() / self.processing_scale) + 1

        y = self.canvas.extent().yMaximum()
        x = self.canvas.extent().xMinimum()
        upper_left = QgsPointXY(x, y)
        y = y - self.processing_scale * y_tiles
        x = x + self.processing_scale * x_tiles
        lower_right = QgsPointXY(x, y)
        self.bounding_box = QgsRectangle(upper_left, lower_right)

        # Render and tile map
        self.dlg.inputKey.setPlainText("Rendering map...\nPlease don't move the canvas")
        tiles = np.empty(
            (y_tiles, x_tiles, self.tile_size, self.tile_size, 3), dtype="uint8"
        )
        for y_tile in range(y_tiles):
            self.dlg.progressBar.setValue(100 * y_tile // y_tiles)
            for x_tile in range(x_tiles):
                # Render and tile map
                y = self.canvas.extent().yMaximum() - y_tile * self.processing_scale
                x = self.canvas.extent().xMinimum() + x_tile * self.processing_scale
                upper_left = QgsPointXY(x, y)
                y = y - self.processing_scale
                x = x + self.processing_scale
                lower_right = QgsPointXY(x, y)
                bounding_box = QgsRectangle(upper_left, lower_right)
                tiles[y_tile, x_tile] = self.get_input(bounding_box)
        self.dlg.progressBar.setValue(100)

        # If client requests online prediction
        if self.dlg.inputUseServer.isChecked():
            if self.key == "nokey":
                self.dlg.inputKey.setPlainText(
                    "Offload feature requires key\nPlease input key with dashes."
                )
                return
            if self.pthread is None:
                self.dlg.inputKey.setPlainText("Uploading assets...")
                self.pthread = QThread()
                self.predictor = Predictor(
                    self.model,
                    tiles,
                    self.tile_size,
                    self.pixel_size,
                    y_tiles,
                    x_tiles,
                    self.device,
                    clusters,
                    self.key,
                )
                self.predictor.moveToThread(self.pthread)
                self.worker = self.predictor
                self.pthread.started.connect(self.predictor.run)
                self.predictor.finished.connect(self.pthread.quit)
                self.predictor.finished.connect(self.predictor.deleteLater)
                self.pthread.finished.connect(self.pthread.deleteLater)
                self.pthread.finished.connect(self.finished_remote_predict)
                self.pthread.start()
                return

        # Execute network
        tiles = np.swapaxes(tiles, -1, -2)
        tiles = np.swapaxes(tiles, -2, -3)
        self.dlg.inputKey.setPlainText("Encoding... {}".format(message))
        batch_size = 1
        test_vector = self.encoder.forward(
            torch.tensor(tiles[0, 0], dtype=torch.float32).to(self.device)
        )
        vectors = np.empty((y_tiles, x_tiles, *test_vector.shape[-3:]), dtype="uint8")
        for yy in range(0, y_tiles):
            self.dlg.progressBar.setValue(100 * yy // y_tiles)
            for xx in range(0, x_tiles, batch_size):
                batch = tiles[yy, xx : xx + batch_size]
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                vectors[yy, xx : xx + batch_size] = (
                    self.encoder.forward(batch / 255).cpu().detach().numpy() * 255 
                )

        # K means clustering
        self.dlg.inputKey.setPlainText("Segmenting...")
        self.dlg.progressBar.setValue(0)
        kmeans = cluster.KMeans(n_clusters=clusters)
        vectors = np.swapaxes(vectors, -3, -2)
        vectors = np.swapaxes(vectors, -2, -1)
        kmeans.fit(
            vectors.reshape(
                (
                    vectors.shape[-3] * vectors.shape[-2] * y_tiles * x_tiles,
                    vectors.shape[-1],
                )
            )
        )

        # Get mask from kmeans
        self.dlg.inputKey.setPlainText("Processing...")
        masks = np.zeros(
            (clusters, y_tiles * x_tiles * vectors.shape[-3] * vectors.shape[-2]),
            dtype="uint8",
        )
        for c in range(clusters):
            masks[c][kmeans.labels_ == c] = 255
        masks = masks.reshape((clusters, y_tiles, x_tiles, *vectors.shape[-3:-1]))

        # Process segment map
        segments = np.zeros(
            (y_tiles, x_tiles, self.tile_size, self.tile_size), dtype="uint8"
        )
        for y in range(y_tiles):
            for x in range(x_tiles):
                for c in range(clusters):
                    upscaled = cv2.resize(
                        masks[c, y, x],
                        (self.tile_size, self.tile_size),
                        cv2.INTER_LINEAR,
                    )
                    segments[y, x][upscaled > 0] = c * (255 // clusters)

        # Consolidate tiles into one big raster
        raster_data = np.empty(
            (self.tile_size * y_tiles, self.tile_size * x_tiles), dtype="uint8"
        )
        for yy in range(y_tiles):
            for xx in range(x_tiles):
                raster_data[
                    yy * self.tile_size : yy * self.tile_size + self.tile_size,
                    xx * self.tile_size : xx * self.tile_size + self.tile_size,
                ] = segments[yy, xx]

        self.write_raster_layer(raster_data, self.bounding_box)
        self.dlg.progressBar.setValue(100)
        self.dlg.inputKey.setPlainText("Map segmented!")

        return

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

    def reset_model(self):
        res_enum = {"low": 64, "medium": 32, "high": 16, "high+": 8, "very high": 4}

        if self.tthread is not None or self.pthread is not None:
            self.dlg.inputKey.setPlainText("Please don't alter model settings mid-run!")
            return

        input_res = self.dlg.inputResolution.currentText()
        self.pixel_size = res_enum[input_res]
        self.tile_size = 512

        self.model = AE(
            input_shape=(self.tile_size, self.tile_size),
            in_channels=3,
            pixel_size=self.pixel_size,
            decode=True,
        ).to(self.device)
        self.encoder = AE(
            input_shape=(self.tile_size, self.tile_size),
            in_channels=3,
            pixel_size=self.pixel_size,
            decode=False,
        ).to(self.device)

        self.dlg.inputKey.setPlainText("Model reset!")

    def set_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.encoder.load_state_dict(state_dict)

    def submit(self):
        res_enum = {"low": 64, "medium": 32, "high": 16, "high+": 8, "very high": 4}
        input_res = self.dlg.inputResolution.currentText()
        self.pixel_size = res_enum[input_res]

        key = self.dlg.inputKey.toPlainText()
        license_key = os.path.join(self.plugin_dir, "qgis_key")

        # Process special input
        if key == "delete_key":
            os.remove(license_key)
            self.dlg.inputKey.setPlainText("Key deleted!")
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
            self.dlg.inputKey.setPlainText(msg)
            self.key = key
        else:
            self.dlg.inputKey.setPlainText(msg)
            self.key = "nokey"

    # Get local and remote models and display in dropdown
    def render_models(self):
        res_list = ["high", "high+", "very_high"]
        local_models = glob(self.plugin_dir + "/*.torch")
        remote_models = [
            os.path.join(self.plugin_dir, f"{model_id}.torch")
            for model_id in res_list
        ]

        models = set(local_models + remote_models)

        # Add to dropdown
        self.dlg.inputLoadModel.clear()
        for model in models:
            self.dlg.inputLoadModel.addItem(os.path.basename(model).replace(".torch", ""))

    def download_model(self, model_path):
        # Download all remote models
        if self.key != "nokey":
            try:
                model_buffer = self.keygen_model(
                    os.path.basename(model_path), self.key
                )
            except ValueError as e:
                self.dlg.inputKey.setPlainText(f"Downloading models failed\n{e}")
                return
            with open(model_path, "wb") as f:
                f.write(model_buffer.read())
        else:
            self.dlg.inputKey.setPlainText(f"Downloading models requires key.")

    # Load selected model from local disk or repo
    def load_model(self):
        model_name = self.dlg.inputLoadModel.currentText() + ".torch"

        model_path = os.path.join(self.plugin_dir, model_name)
        if not os.path.exists(model_path):
            self.download_model(model_path)

        model = torch.load(
            model_path, map_location=self.device
        )

        try:
            self.set_model(model.state_dict())
        except RuntimeError:
            self.dlg.inputKey.setPlainText(
                f"Oops! Something went wrong. Try selecting a different resolution for this model."
            )
            self.render_models()
            return

        self.dlg.inputKey.setPlainText(f"{model_name} loaded!")

    def save_model(self):
        model_path = self.dlg.inputSaveModel.toPlainText() + ".torch"
        torch.save(self.model, os.path.join(self.plugin_dir, model_path))
        self.dlg.inputKey.setPlainText(f"Saved model {model_path}")

    def check_server(self):

        url = "http://qgis.quantcivil.ai:5000/buzz"
        available = True
        try:
            response = requests.get(url)
            if response.status_code != 200:
                available = False
        except:
            available = False
        
        if not available:
            self.dlg.inputKey.setPlainText("Offload server offline.")
            self.dlg.inputUseServer.clicked.disconnect()
            self.dlg.inputUseServer.setChecked(0)
            self.dlg.inputUseServer.clicked.connect(self.check_server)     

    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = SegmenterDialog()
            self.canvas = self.iface.mapCanvas()
            self.tthread = None
            self.pthread = None
            self.worker = None

            # Set up user interface interactions
            self.dlg.buttonTrain.clicked.connect(self.train)
            self.dlg.buttonPredict.clicked.connect(self.predict)
            self.dlg.buttonRefresh.clicked.connect(self.update_progress)
            self.dlg.inputResolution.currentTextChanged.connect(self.reset_model)
            self.dlg.inputKey.textChanged.connect(self.submit)
            self.dlg.inputLoadModel.textActivated.connect(self.load_model)
            self.dlg.buttonSaveModel.clicked.connect(self.save_model)
            self.dlg.inputUseServer.clicked.connect(self.check_server)     

            # Scale at which tiles are created
            self.processing_scale = 128

            # Set up model
            #  use gpu if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            self.render_models()
            self.reset_model()

            self.license_path = os.path.join(self.plugin_dir, "qgis_key")
            self.key = "nokey"
            if not os.path.exists(self.license_path):
                self.dlg.inputKey.setPlainText("Please input key with dashes")
            else:
                # Check license
                with open(self.license_path, "r") as f:
                    self.key = f.readline()
                act, msg = activate_license(self.key, self.keygen_account_id)
                if not act:
                    os.remove(self.license_path)
                    self.key = "nokey"
                self.dlg.inputKey.setPlainText(f"{msg}")

            # Render logo
            img_path = os.path.join(self.plugin_dir, "logo.png")
            pix = QPixmap(img_path)
            self.dlg.imageTraining.setPixmap(pix)

        # show the dialog
        self.dlg.show()
