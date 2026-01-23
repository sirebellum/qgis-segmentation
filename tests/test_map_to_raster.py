# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""QGIS-free unit tests for map-to-raster layer detection and parameter mapping."""

import pytest


class TestIsFileBackedGdalRaster:
    """Tests for is_file_backed_gdal_raster detection function."""

    def test_valid_gdal_geotiff_three_bands(self):
        """Valid 3-band GDAL GeoTIFF should return True."""
        from map_to_raster import is_file_backed_gdal_raster

        assert is_file_backed_gdal_raster("gdal", "/path/to/file.tif", 3) is True
        assert is_file_backed_gdal_raster("gdal", "/path/to/file.tiff", 3) is True
        assert is_file_backed_gdal_raster("GDAL", "/path/to/file.TIF", 3) is True

    def test_wrong_provider_returns_false(self):
        """Non-GDAL providers should return False."""
        from map_to_raster import is_file_backed_gdal_raster

        assert is_file_backed_gdal_raster("wms", "/path/to/file.tif", 3) is False
        assert is_file_backed_gdal_raster("xyz", "/path/to/file.tif", 3) is False
        assert is_file_backed_gdal_raster("wmts", "/path/to/file.tif", 3) is False

    def test_wrong_band_count_returns_false(self):
        """Non-3-band rasters should return False."""
        from map_to_raster import is_file_backed_gdal_raster

        assert is_file_backed_gdal_raster("gdal", "/path/to/file.tif", 1) is False
        assert is_file_backed_gdal_raster("gdal", "/path/to/file.tif", 4) is False
        assert is_file_backed_gdal_raster("gdal", "/path/to/file.tif", 0) is False

    def test_wrong_extension_returns_false(self):
        """Non-.tif/.tiff extensions should return False."""
        from map_to_raster import is_file_backed_gdal_raster

        assert is_file_backed_gdal_raster("gdal", "/path/to/file.png", 3) is False
        assert is_file_backed_gdal_raster("gdal", "/path/to/file.jpg", 3) is False
        assert is_file_backed_gdal_raster("gdal", "/path/to/file.ecw", 3) is False

    def test_none_values_return_false(self):
        """None provider or source should return False."""
        from map_to_raster import is_file_backed_gdal_raster

        assert is_file_backed_gdal_raster(None, "/path/to/file.tif", 3) is False
        assert is_file_backed_gdal_raster("gdal", None, 3) is False

    def test_path_with_layer_options(self):
        """Source with GDAL layer options should work."""
        from map_to_raster import is_file_backed_gdal_raster

        source = "/path/to/file.tif|layername=my_layer"
        assert is_file_backed_gdal_raster("gdal", source, 3) is True


class TestIsRenderableNonFileLayer:
    """Tests for is_renderable_non_file_layer detection function."""

    def test_wms_provider_is_renderable(self):
        """WMS provider should be detected as renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "wms", "http://example.com/wms") is True

    def test_wmts_provider_is_renderable(self):
        """WMTS provider should be detected as renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "wmts", "http://example.com/wmts") is True

    def test_xyz_provider_is_renderable(self):
        """XYZ tiles provider should be detected as renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "xyz", "http://example.com/xyz/{z}/{x}/{y}.png") is True

    def test_arcgis_provider_is_renderable(self):
        """ArcGIS server provider should be detected as renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "arcgismapserver", "http://example.com/arcgis") is True
        assert is_renderable_non_file_layer("RasterLayer", "arcgisfeatureserver", "http://example.com/arcgis") is True

    def test_vector_layer_is_renderable(self):
        """Vector layers should be detected as renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("VectorLayer", "ogr", "/path/to/file.shp") is True
        assert is_renderable_non_file_layer("VectorLayer", "postgres", "dbname=test") is True

    def test_gdal_local_file_not_renderable(self):
        """Local GDAL file should not be flagged as needing conversion."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "gdal", "/local/file.tif") is False

    def test_gdal_remote_vsicurl_is_renderable(self):
        """GDAL with /vsicurl/ remote source should be renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "gdal", "/vsicurl/http://example.com/data.tif") is True

    def test_gdal_http_source_is_renderable(self):
        """GDAL with HTTP source should be renderable."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", "gdal", "http://example.com/data.tif") is True
        assert is_renderable_non_file_layer("RasterLayer", "gdal", "https://example.com/data.tif") is True

    def test_none_provider_returns_false(self):
        """None provider should return False."""
        from map_to_raster import is_renderable_non_file_layer

        assert is_renderable_non_file_layer("RasterLayer", None, "/path/to/file.tif") is False


class TestBuildConvertMapToRasterParams:
    """Tests for build_convert_map_to_raster_params function."""

    def test_basic_params_with_crs(self):
        """Parameters should include extent, layer, and map units."""
        from map_to_raster import build_convert_map_to_raster_params

        extent = (100.0, 200.0, 150.0, 250.0, "EPSG:4326")
        params = build_convert_map_to_raster_params(extent, "my_layer_id")

        assert params["MAP_UNITS_PER_PIXEL"] == 1.0
        assert params["LAYERS"] == ["my_layer_id"]
        assert "100.0,150.0,200.0,250.0 [EPSG:4326]" in params["EXTENT"]
        assert params["OUTPUT"] == ""

    def test_params_without_crs(self):
        """Parameters should work without CRS."""
        from map_to_raster import build_convert_map_to_raster_params

        extent = (0.0, 0.0, 100.0, 100.0, None)
        params = build_convert_map_to_raster_params(extent, "layer_123")

        assert params["MAP_UNITS_PER_PIXEL"] == 1.0
        assert params["LAYERS"] == ["layer_123"]
        assert params["EXTENT"] == "0.0,100.0,0.0,100.0"

    def test_custom_map_units(self):
        """Custom map units should be respected."""
        from map_to_raster import build_convert_map_to_raster_params

        extent = (0.0, 0.0, 100.0, 100.0, None)
        params = build_convert_map_to_raster_params(extent, "layer", map_units_per_pixel=2.5)

        assert params["MAP_UNITS_PER_PIXEL"] == 2.5

    def test_default_map_units_is_one(self):
        """Default map units per pixel should be 1.0."""
        from map_to_raster import MAP_UNITS_PER_PIXEL_DEFAULT

        assert MAP_UNITS_PER_PIXEL_DEFAULT == 1.0


class TestWebServiceProviders:
    """Tests for web service provider detection."""

    def test_all_web_providers_in_set(self):
        """Verify all expected web providers are in the detection set."""
        from map_to_raster import WEB_SERVICE_PROVIDERS

        expected = {"wms", "wmts", "wcs", "xyz", "arcgismapserver", "arcgisfeatureserver", "wfs", "vectortile", "mbtiles"}
        for provider in expected:
            assert provider in WEB_SERVICE_PROVIDERS, f"Missing provider: {provider}"


class TestConvertMapToRasterAlgorithmId:
    """Tests for the algorithm ID constant."""

    def test_algorithm_id_is_native_rasterize(self):
        """Verify the algorithm ID is set correctly."""
        from map_to_raster import CONVERT_MAP_TO_RASTER_ALG_ID

        assert CONVERT_MAP_TO_RASTER_ALG_ID == "native:rasterize"


class TestExtractLayerMetadata:
    """Tests for extract_layer_metadata helper (with mocked layers)."""

    def test_extracts_basic_fields(self):
        """Metadata extraction should return expected keys."""
        from map_to_raster import extract_layer_metadata

        class MockProvider:
            def name(self):
                return "gdal"

        class MockLayer:
            def dataProvider(self):
                return MockProvider()

            def source(self):
                return "/path/to/file.tif"

            def bandCount(self):
                return 3

        meta = extract_layer_metadata(MockLayer())

        assert meta["layer_type"] == "MockLayer"
        assert meta["provider_name"] == "gdal"
        assert meta["source_path"] == "/path/to/file.tif"
        assert meta["band_count"] == 3

    def test_handles_missing_provider(self):
        """Should handle layers without provider gracefully."""
        from map_to_raster import extract_layer_metadata

        class MockLayer:
            def dataProvider(self):
                return None

            def source(self):
                return "/path/to/file.shp"

        meta = extract_layer_metadata(MockLayer())

        assert meta["provider_name"] is None
        assert meta["band_count"] == 0

    def test_handles_missing_bandCount(self):
        """Should handle layers without bandCount method."""
        from map_to_raster import extract_layer_metadata

        class MockProvider:
            def name(self):
                return "ogr"

        class MockVectorLayer:
            def dataProvider(self):
                return MockProvider()

            def source(self):
                return "/path/to/file.shp"

        meta = extract_layer_metadata(MockVectorLayer())

        assert meta["band_count"] == 0
        assert meta["provider_name"] == "ogr"
