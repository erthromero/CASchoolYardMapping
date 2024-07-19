# Code writte to perform object-based classification accuracy assessment of
# school yard classification in California urban centers
# Author: Eric Romero
# Date: 7/11/2023



from osgeo import gdal 
from os.path import isfile
import numpy as np

def PolygonAccuracy(confusion_mat_path: str, polygon_rast: str, class_rast: str, im_res: float):
    
    """""
    Function takes rasterized validation polygons that have been rasterized and performs polygon-based accuracy assessment of 
    classified raster. Assessment appropriate for geographic object-based image classifications.
    Geospatial extents and reference of polygon and class rasters must match, as well as classification schemas.

    confusion_mat_path: str - path to  output confusion matrix
    
    polygon_rast: path to rasterized validation polygons
    
    class_rast: path to classified raster
    
    im_res: float - image resolution in meters

    """""

    assert isfile(polygon_rast), f'[ERROR] File {polygon_rast} not found. Exiting.'
    assert isfile(class_rast), f'[ERROR] File {class_rast} not found. Exiting.'

    val_ds = gdal.Open(polygon_rast)
    class_ds = gdal.Open(class_rast)

    val_arr = val_ds.GetRasterBand(1).ReadAsArray()
    class_arr = class_ds.GetRasterBand(1).ReadAsArray()

    min_class = 1
    max_class = 3

    conf_mat = np.zeros((max_class,max_class))

    for i in range(1,max_class+1):
        val_mask = val_arr == i
        class_masked = class_arr[val_mask]
        for j in range(1,max_class+1):
            class_masked_ij = class_masked == j
            sum_ij = np.count_nonzero(class_masked_ij) * (im_res * im_res)
            conf_mat[j-1,i-1] = sum_ij
    
    np.savetxt(confusion_mat_path, conf_mat, delimiter=',')
    
    print('Complete.')

if __name__ == "__main__": 
    from sys import argv

    try:
        PolygonAccuracy(argv[1], argv[2], argv[3], float(argv[4]))

    except Exception as e:
        
        print(f'\n[ERROR] Expected python PolygonAccuracy.py /path/to/output_conf_mat.csv /path/to/validaton_polygons.tif /path/to/class_raster.tif im_res, but got {argv[:]} instead. Exiting.')
        print(f'\n{e}')    
