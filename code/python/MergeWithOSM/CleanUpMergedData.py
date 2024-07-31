# Code creates new vector shapefile and new classified raster land cover datasets with designated drawing order.
# This code was originally written with only 4 classes in mind: 1 - Tree/Shrub, 2 - Grass/Pervious, 3 - Other Impervious
# 4 - Buildings/Structures. In order to accomodate more classes, you will need to alter the iterable list which loops over
# which classes to processes and in what order. This list can be found within the function 'process_land_cover'.

# Author: Eric Romero, PhD Candidate, UC Berkeley
# Last edit: 7/31/2024

import os
import sys
import numpy as np
from osgeo import gdal, ogr, osr

def vector_to_array(src_ds, src_layer_name, ref_raster_path, class_value):
    #NOTE (Eric): Open the reference raster file
    ref_ds = gdal.Open(ref_raster_path)
    ref_geo_transform = ref_ds.GetGeoTransform()
    ref_projection = ref_ds.GetProjection()
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize

    #NOTE (Eric): Open the source vector file
    src_layer = src_ds.GetLayerByName(src_layer_name)

    #NOTE (Eric): Filter the layer by class value
    src_layer.SetAttributeFilter(f"DN = {class_value}")

    #NOTE (Eric): Create the destination raster in-memory
    mem_driver = gdal.GetDriverByName('MEM')
    target_ds = mem_driver.Create('', x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(ref_geo_transform)
    target_ds.SetProjection(ref_projection)
    
    #NOTE (Eric): Rasterize the vector layer
    gdal.RasterizeLayer(target_ds, [1], src_layer, options=["ATTRIBUTE=DN"])
    
    array = target_ds.ReadAsArray()
    target_ds = None
    ref_ds = None

    return array

def burn_into_final_array(final_array, class_array, class_value):
    mask = class_array == class_value
    final_array[mask] = class_value
    return final_array

def array_to_raster(array, ref_raster_path, dst_raster_path):

    #NOTE (Eric): Set creation options
    creation_options = [
        'BIGTIFF=IF_NEEDED', 
        'COMPRESS=LZW', 
        'PREDICTOR=2', 
        'TILED=YES', 
        'BLOCKXSIZE=256', 
        'BLOCKYSIZE=256', 
        'SPARSE_OK=TRUE', 
        'NUM_THREADS=ALL_CPUS'
    ]

    #NOTE (Eric): Open the reference raster file
    ref_ds = gdal.Open(ref_raster_path)
    ref_geo_transform = ref_ds.GetGeoTransform()
    ref_projection = ref_ds.GetProjection()
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize

    #NOTE (Eric): Create the destination raster file
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(dst_raster_path, x_res, y_res, 1, gdal.GDT_Byte, options=creation_options)
    dst_ds.SetGeoTransform(ref_geo_transform)
    dst_ds.SetProjection(ref_projection)
    
    #NOTE (Eric): Write the array to the raster
    dst_ds.GetRasterBand(1).WriteArray(array)
    dst_ds.GetRasterBand(1).SetNoDataValue(0)
    dst_ds.FlushCache()
    dst_ds = None
    ref_ds = None

def process_land_cover(src_vector_path: str, ref_raster_path: str, new_vector_path: str, new_raster_path: str):

    """
    Function takes a merged shapefile with attribute 'DN' that represents different land cover classes and
    converts them into another shapefile + raster dataset with a cleaner drawing order (i.e., no objects occupying the same
    vector space).

    src_vector_path: str - path to source vector shapefile of merged polygon land cover data
    
    ref_raster_path: str - path to a reference raster dataset whose extents should approximately match those of the src_vector_path.
    This is a crucial variable and will be used for spatial referencing (spatial extents, spatial resolution, datum, etc.)

    new_vector_path: str - path to output vector dataset with cleaned drawing order

    new_raster_path: str - path to output raster dataset based on cleaned vector data

    """

    #NOTE (Eric): Open the source vector file
    src_ds = ogr.Open(src_vector_path)

    #NOTE (Eric): Create an empty final array
    ref_ds = gdal.Open(ref_raster_path)
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize
    final_array = np.zeros((y_res, x_res), dtype=np.uint8)

    #NOTE (Eric): Process each class
    for cls in [3, 4, 2, 1]:
        class_array = vector_to_array(src_ds, src_ds.GetLayer().GetName(), ref_raster_path, cls)
        final_array = burn_into_final_array(final_array, class_array, cls)

    #NOTE (Eric): Save the final array to a raster
    array_to_raster(final_array, ref_raster_path, new_raster_path)

    #NOTE (Eric): Re-vectorize the final raster
    raster_to_vector(new_raster_path, new_vector_path, 'land_cover')

def raster_to_vector(src_raster_path, dst_vector_path, dst_layer_name):
    #NOTE (Eric): Open the source raster file
    src_ds = gdal.Open(src_raster_path)
    src_band = src_ds.GetRasterBand(1)

    #NOTE (Eric): Create the destination vector file
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(dst_vector_path):
        driver.DeleteDataSource(dst_vector_path)
    dst_ds = driver.CreateDataSource(dst_vector_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjectionRef())
    dst_layer = dst_ds.CreateLayer(dst_layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    field = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(field)

    #NOTE (Eric): Polygonize the raster band
    mask_band = src_band.GetMaskBand()
    gdal.Polygonize(src_band, mask_band, dst_layer, 0, [], callback=None)

    #NOTE (Eric): Remove features with DN = 0
    dst_layer.SetAttributeFilter("DN = 0")
    dst_layer.StartTransaction()
    for feature in dst_layer:
        dst_layer.DeleteFeature(feature.GetFID())
    dst_layer.CommitTransaction()
    dst_layer.SetAttributeFilter(None)

    src_ds = None
    dst_ds = None

#NOTE (Eric): Example usage
if __name__ == "__main__":
    src_vector_path = sys.argv[1]
    ref_raster_path = sys.argv[2]
    new_vector_path = sys.argv[3]
    new_raster_path = sys.argv[4]
    process_land_cover(src_vector_path, ref_raster_path, new_vector_path, new_raster_path)
