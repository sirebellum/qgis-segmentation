# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Segmenter
                                 A QGIS plugin
 This plugin segments the map into discrete buckets
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-05-26
        copyright            : (C) 2023 by Joshua Herrera
        email                : joshua.herrera@quantcivil.ai
        git sha              : $Format:%H$
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load Segmenter class from file Segmenter.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .segmenter import Segmenter
    return Segmenter(iface)
