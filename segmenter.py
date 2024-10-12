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
from qgis.core import (
    QgsTask,
    QgsApplication,
    QgsMessageLog,
    QgsProject,
    Qgis,
)

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .segmenter_dialog import SegmenterDialog
import os.path

from io import BytesIO
import requests
from osgeo import gdal

# Install necessary packages
import pkg_resources, pip
def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

if not is_package_installed("torch"):
    pip.main(["install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
if not is_package_installed("scikit-learn"):
    pip.main(["install", "scikit-learn"])
if not is_package_installed("numpy"):
    pip.main(["install", "numpy"])

import torch
from sklearn.cluster import KMeans
import numpy as np

from .funcs import predict_kmeans, predict_cnn
from .qgis_funcs import render_raster

TILE_SIZE = 512


# Multithreading stuff
class Task(QgsTask):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.result = None
        QgsMessageLog.logMessage("Task initialized", "Segmenter", level=Qgis.Info)

    def run(self):
        QgsMessageLog.logMessage("Running task", "Segmenter", level=Qgis.Info)
        try:
            self.result = self.function(*self.args)
            return True
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Exception in task: {e}", "Segmenter", level=Qgis.Critical
            )
            return False

    def finished(self, result):
        QgsMessageLog.logMessage("Task finished", "Segmenter", level=Qgis.Info)
        if result:
            # render raster
            render_raster(
                self.result,
                self.kwargs["layer"].extent(),
                f"{self.kwargs['layer'].name()}_{self.kwargs['model']}_{self.kwargs['num_segments']}_{self.kwargs['resolution']}",
                self.kwargs["canvas"].layer(0).crs().postgisSrid(),
            )
            
def run_task(function, *args, **kwargs):
    task = Task(function, *args, **kwargs)
    QgsApplication.taskManager().addTask(task)
    return task


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

        self.task = None

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
            "high": 4,
            "medium": 8,
            "low": 16,
        }

        # Get user specified resolution
        resolution = resolution_map[self.dlg.inputRes.currentText()]

        # Set up kwargs
        kwargs = {
            "layer": layer,
            "canvas": self.canvas,
            "dlg": self.dlg,
            "model": self.model,
            "num_segments": num_segments,
            "resolution": resolution,
        }

        # set up args
        if self.model == "kmeans":
            func = predict_kmeans
            args = (layer_array, num_segments, resolution)
        elif self.model == "cnn":
            func = predict_cnn
            args = (self.load_model(resolution), layer_array, num_segments, TILE_SIZE, self.device)

        # Run task
        self.task = run_task(func, *args, **kwargs)

        # Display error if task stops running after a little bit
        if self.task.waitForFinished(1):
            self.dlg.inputBox.setPlainText("An error occurred. Please try again.")

    # Load model from disk
    def load_model(self, model_name):
        # Load model into bytes object
        model_path = os.path.join(self.plugin_dir, f"models/model_{model_name}.pth")
        with open(model_path, "rb") as f:
            model_bytes = BytesIO(f.read())

        # Load torchscript model
        model = torch.jit.load(model_bytes)
        model.eval().to(self.device)

        return model

    # Process user input box
    def submit(self):
        return

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
        res_list = ["high", "medium", "low"]
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

            # Set device
            if torch.cuda.is_available(): # Cuda
                self.device = torch.device("cuda")
            else: # CPU
                self.device = torch.device("cpu")

            # Populate drop down menus
            self.render_models()
            self.render_layers()
            self.render_resolutions()

            # Set gpu message
            gpu_msg = "GPU available."
            if self.device == torch.device("cpu"):
                gpu_msg = "GPU not available. Using CPU instead."

            # Display message
            self.dlg.inputBox.setPlainText(gpu_msg)

            # Attach inputs
            self.dlg.inputBox.textChanged.connect(self.submit)
            self.dlg.buttonPredict.clicked.connect(self.predict)
            self.dlg.inputLoadModel.currentIndexChanged.connect(self.set_model)
            self.dlg.inputLoadModel.highlighted.connect(self.render_layers)

            # Render logo
            img_path = os.path.join(self.plugin_dir, "logo.png")
            pix = QPixmap(img_path)
            self.dlg.imageLarge.setPixmap(pix)

        # show the dialog
        self.render_layers()
        self.dlg.show()
