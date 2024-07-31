# This code makes a comparison between tree cover classification results and 2018 tree classification results produced by EarthDefine.
# This analysis creates a confusion matrix which assumes that the EarthDefine dataset is like a "ground truth". 
# This code also creates a comparison between our classification and EarthDefine's in terms of total percent tree cover
# and total tree area. All input datasets are shapefiles and require a reference raster dataset to complete this processing,
# which should roughly match the extents of the shapefile. All input datasets are filtered by a clipping shapefile.
# Confusion matrix and %area/total area comparisons written to .csv files.

# Author: Eric Romero
# Last edit: 7/31/2024

import os
import numpy as np
from osgeo import gdal
from osgeo import ogr
import geopandas as gpd
import pandas as pd
import progressbar as pb
from os.path import isfile

gdal.UseExceptions()
ogr.UseExceptions()

gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS") 
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS") #Thank the GDAL gods for hopeless redundancy
gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES") # NOTE (Eric): This is necessary when using "invalid" cutline files for cropping/masking

def progress_callback(complete, message, data):
    percent = int(complete * 100)  # round to integer percent
    data.update(percent)  # update the progressbar
    return 1

def filter_and_polygonize(shp_path, field_name, field_value, ref_raster):
    #NOTE (Eric): Read the shapefile and filter features
    shp_ds = ogr.Open(shp_path)

    # Get the layer
    shp_layer = shp_ds.GetLayer()
    
    # Create a filter expression
    #filter_expression = f"{field_name} = '{field_value}'"
    
    # Apply the filter to the layer
    #shp_layer.SetAttributeFilter(filter_expression)
    
    #NOTE (Eric): Open the reference raster to get georeferencing information
    ref_ds = gdal.Open(ref_raster)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize
    data_type = ref_ds.GetRasterBand(1).DataType
    
    #NOTE (Eric): Create the in-memory output raster with the same georeferencing information
    mem_driver = gdal.GetDriverByName('MEM')
    raster_ds = mem_driver.Create('', x_res, y_res, 1, data_type)
    raster_ds.SetGeoTransform(gt)
    raster_ds.SetProjection(proj)
    
    #NOTE (Eric): Rasterize the filtered shapefile
    band = raster_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    gdal.RasterizeLayer(raster_ds, [1], shp_layer, options=["ATTRIBUTE=%s" % field_name])
    
    return raster_ds

def warp_image_match_res(src_ds, match_ds, resample_alg: str, cutline_fn=None):
    alpha = False
    dst_ds = gdal.GetDriverByName('MEM').Create('', match_ds.RasterXSize, match_ds.RasterYSize, src_ds.RasterCount, src_ds.GetRasterBand(1).DataType)

    match_srs = match_ds.GetSpatialRef()
    match_gt = match_ds.GetGeoTransform()
    dst_ds.SetGeoTransform(match_gt)
    dst_ds.SetProjection(match_srs.ExportToWkt())

    gdal_no_data_value = src_ds.GetRasterBand(1).GetNoDataValue()
    if gdal_no_data_value is None:
        gdal_no_data_value = 0
        
        for b in range(src_ds.RasterCount):
            band_n = src_ds.GetRasterBand(b+1)
            band_n.SetNoDataValue(gdal_no_data_value)

    pbar = pb.ProgressBar(maxval=100).start()

    if cutline_fn and os.path.isfile(cutline_fn):
        cutline_ds = ogr.Open(cutline_fn)
        cutline_layer = cutline_ds.GetLayer()
        cutline_ds_name = cutline_ds.GetName()
        cutline_layer_name = cutline_layer.GetName()
        
        if alpha:
            gdal.Warp(dst_ds, src_ds, format='MEM', multithread='YES',
                      srcAlpha=True, dstNodata=gdal_no_data_value, outputType=src_ds.GetRasterBand(1).DataType,
                      width=match_ds.RasterXSize, height=match_ds.RasterYSize, outputBoundsSRS=match_srs, outputBounds=match_gt,
                      srcSRS=src_ds.GetSpatialRef(), dstSRS=match_srs, resampleAlg=resample_alg, cutlineDSName=cutline_ds_name, cutlineLayer=cutline_layer_name,
                      callback=progress_callback, callback_data=pbar)
        else:
            gdal.Warp(dst_ds, src_ds, format='MEM', multithread='YES',
                      dstNodata=gdal_no_data_value, outputType=src_ds.GetRasterBand(1).DataType, width=match_ds.RasterXSize,
                      height=match_ds.RasterYSize, outputBoundsSRS=match_srs, outputBounds=match_gt, 
                      srcSRS=src_ds.GetSpatialRef(), dstSRS=match_srs, resampleAlg=resample_alg, cutlineDSName=cutline_ds_name,
                      cutlineLayer=cutline_layer_name, callback=progress_callback, callback_data=pbar)
    else:
        if alpha:
            gdal.Warp(dst_ds, src_ds, format='MEM', multithread='YES', srcAlpha=True,
                      dstNodata=gdal_no_data_value, outputType=src_ds.GetRasterBand(1).DataType, width=match_ds.RasterXSize,
                      height=match_ds.RasterYSize, outputBoundsSRS=match_srs, outputBounds=match_gt,
                      srcSRS=src_ds.GetSpatialRef(), dstSRS=match_srs, resampleAlg=resample_alg, callback=progress_callback, callback_data=pbar)
        else:
            gdal.Warp(dst_ds, src_ds, format='MEM', multithread='YES', 
                      dstNodata=gdal_no_data_value, outputType=src_ds.GetRasterBand(1).DataType,
                      width=match_ds.RasterXSize, height=match_ds.RasterYSize, outputBoundsSRS=match_srs,
                      outputBounds=match_gt, srcSRS=src_ds.GetSpatialRef(), dstSRS=match_srs,
                      resampleAlg=resample_alg, callback=progress_callback, callback_data=pbar)

    pbar.finish()
    
    return dst_ds

def PolygonAccuracy(confusion_mat_path: str, polygon_rast, class_rast, im_res: float):
    val_arr = polygon_rast.GetRasterBand(1).ReadAsArray()
    class_arr = class_rast.GetRasterBand(1).ReadAsArray()
    
    # NOTE (Eric): Extract masks for nodata and non-tree regions from our classification. This assumes that our tree class is a value of 1
    nodata_mask = (class_arr == 0)
    no_tree_mask = (class_arr > 1)

    #NOTE (Eric): Apply nodata mask to validation raster
    val_arr[nodata_mask] = 255
    
    #NOTE (Eric): Apply nodata and no-tree masks to calssification data
    class_arr[nodata_mask] = 255
    class_arr[no_tree_mask] = 0

    min_class = 0
    max_class = 1

    conf_mat = np.zeros((max_class + 1, max_class + 1))

    for i in range(0, max_class + 1):
        val_mask = val_arr == i
        class_masked = class_arr[val_mask]
        for j in range(0, max_class + 1):
            class_masked_ij = class_masked == j
            sum_ij = np.count_nonzero(class_masked_ij) * (im_res * im_res)
            conf_mat[j, i] = sum_ij

    np.savetxt(confusion_mat_path, conf_mat, delimiter=',')
    print('Confusion matrix saved to', confusion_mat_path)

def calculate_percent_tree_cover(raster_ds, out_csv_path):
    arr = raster_ds.GetRasterBand(1).ReadAsArray()
    total_pixels = arr.size
    tree_pixels = np.count_nonzero(arr == 1)
    percent_tree_cover = (tree_pixels / total_pixels) * 100
    
    df = pd.DataFrame({'Total Pixels': [total_pixels], 'Tree Pixels': [tree_pixels], 'Percent Tree Cover': [percent_tree_cover]})
    df.to_csv(out_csv_path, index=False)
    print('Percent tree cover saved to', out_csv_path)

def calculate_percent_tree_cover_comparison(class_ds, ref_ds, out_csv_path):

    #NOTE (Eric): Extract classification and EarthDefine arrays
    class_arr = class_ds.GetRasterBand(1).ReadAsArray()
    ref_arr = ref_ds.GetRasterBand(1).ReadAsArray()

    #NOTE (Eric): Extract area of classified raster
    nodata_mask = class_arr == 0
    valid_class_pixel_mask = class_arr > 0

    total_area = valid_class_pixel_mask.sum() * 0.6 * 0.6

    class_tree_area = np.count_nonzero(class_arr == 1) * 0.6 * 0.6
    ref_tree_area = np.count_nonzero(ref_arr == 1) * 0.6 * 0.6

    class_percent_tree_cover = (class_tree_area / total_area) * 100
    ref_percent_tree_cover = (ref_tree_area / total_area) * 100
    
    df = pd.DataFrame({'Tree Area - Our Study (m2)': [class_tree_area], 'Tree Area - EarthDefine (m2)': [ref_tree_area],
                        '%Tree Cover (This study)': [class_percent_tree_cover], '%Tree Cover (EarthDefine)': [ref_percent_tree_cover]})
    
    df.to_csv(out_csv_path, index=False)
    print('Percent tree cover saved to', out_csv_path)


if __name__ == "__main__": 

    """
    shapefile_1: str - full path to shapefile_1 which contains land cover classification that you want to compare to EarthDefine

    shapefile_2: str - full path to shapefile_2 which contains tree classification produced by EarthDefine

    ref_raster: str - full path to a reference raster dataset whose extents should approximately match those of the shapefile_1/2.
    This is a crucial variable and will be used for spatial referencing (spatial extents, spatial resolution, datum, etc.) of
    in-memory datasets

    city_name: str - name of city to be processed, should match formatting in your file system/file naming

    """

    from sys import argv

    try:

        shapefile_1 = argv[1]
        shapefile_2 = argv[2]
        ref_raster = argv[3]
        city_name = argv[4]

        #NOTE (Eric): Assert input files exist
        assert isfile(shapefile_1), f'[ERROR] File {shapefile_1} not found. Exiting.'
        assert isfile(shapefile_2), f'[ERROR] File {shapefile_2} not found. Exiting.'
        assert isfile(ref_raster), f'[ERROR] File {ref_raster} not found. Exiting.'

        #NOTE (Eric): Filter and polygonize shapefiles
        raster_1 = filter_and_polygonize(shapefile_1, 'DN', 1, ref_raster) #NOTE (Eric): Make sure you're referencing the right attribute!
        raster_2 = filter_and_polygonize(shapefile_2, 'gridcode', 1, ref_raster) #NOTE (Eric): Make sure you're referencing the right attribute!

        #NOTE (Eric): Warp rasters to match extents, resolutions, and projections
        warped_raster_2 = warp_image_match_res(raster_2, raster_1, 'nearest',
                                                cutline_fn=f"E:\\SummerGSR2024\\Schools_Joined_min5b\\{city_name}\\{city_name}_Schools_Buffer10m_WGS84.shp")
        
        warped_raster_1 = warp_image_match_res(raster_1, warped_raster_2, 'nearest',
                                                cutline_fn=f"E:\\SummerGSR2024\\Schools_Joined_min5b\\{city_name}\\{city_name}_Schools_Buffer10m_WGS84.shp")

        #NOTE (Eric): Generate confusion matrix
        confusion_matrix_path = f'E:\\SummerGSR2024\\2018TreeCoverComparison\\{city_name}-EarthDefineConfusionMatrix.csv' #NOTE (Eric): Path to confusion matrix .csv to be written
        PolygonAccuracy(confusion_matrix_path, warped_raster_2, warped_raster_1, 0.6) #NOTE (Eric): Last argument is image resolution in meters

        #NOTE (Eric): Calculate and export percent tree cover
        percent_tree_cover_csv_1 = f'E:\\SummerGSR2024\\2018TreeCoverComparison\\{city_name}-USDA-EarthDefine-TreeCover-Stats.csv' #NOTE (Eric): Path to percent tree cover and tree area .csv
        calculate_percent_tree_cover_comparison(warped_raster_1, warped_raster_2, percent_tree_cover_csv_1)

    except Exception as e:
        
        print(f'\n[ERROR] Expected python EarthDefineconfusionMatrixAndCoverStats.py /path/to/YourClassification.shp /path/to/EarthDefinePolygons.shp /path/to/ref_class_raster.tif, but got {argv[:]} instead. Exiting.')
        print(f'\n{e}')    


