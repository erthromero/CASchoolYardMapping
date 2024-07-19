
#NOTE (Eric): Imports
import cv2
import time
import torch
import psutil
import warnings
import numpy as np
import torch.nn as nn
from os.path import isfile
from osgeo import gdal, ogr
from itertools import chain
from scipy.stats import describe
from skimage.exposure import equalize_adapthist
from pysnic.algorithms.snic import snic, compute_grid
from pysnic.metric.snic import create_augmented_snic_distance
from pysnic.ndim.operations_collections import nd_computations
from concurrent.futures import ProcessPoolExecutor, as_completed


from math import isnan as math_isnan
from matplotlib import pyplot as plt

# Ignore RuntimeWarnings during object statistics collection 
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ClassyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def monitor_resources():
    """Monitor system resources."""
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    #logging.debug(f"CPU Usage: {cpu_usage}%")
    #logging.debug(f"Memory Usage: {memory_info.percent}%")
    return cpu_usage, memory_info.percent

def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):

    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene

def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

def breakup_large_segments(segmentation1, segmentation2, max_segment_area, pixel_res):
    """
    Breaks large segments from SAM segments (segmentation1) into smaller
    over-segmented segments from SNIC (segmentation2)

    Parameters:
    segmentation1 (np.ndarray): The input segmentation array.
    This array is checked for large segments (SAM)

    segmentation2 (np.ndarray): The burn-in segmentation (SNIC). If
    the SAM segments are too large, SAM segment is broken up and
    these smaller 'over-segmented' segments get burned-in.
    
    max_segment_area: The maximum allowed area for segments in m^2.
    
    pixel_res: The resolution of a single pixel (should be equal in x and y dims)

    Returns:
    np.ndarray: The broken-up segmentation array.
    """

    # Label connected components in the segmentation
    min_seg_val = np.nanmin(segmentation1)
    max_seg_val = np.nanmax(segmentation1)

    for segment_label in range(min_seg_val, max_seg_val + 1):
        # Find the area of the current segment in pixels
        segment_area = np.sum(segmentation1 == segment_label) * (pixel_res*pixel_res)
        
        # Only keep segments that are smaller than the area threshold
        if segment_area >= max_segment_area:
            segmentation1[segmentation1 == segment_label] = segmentation2[segmentation1 == segment_label]
    
    return segmentation1

def burn_segmentation(segmentation1, segmentation2, foreground_mask):
    """
    Burn the results of segmentation2 onto segmentation1 using a foreground mask.
    
    Parameters:
    - segmentation1: np.array, first segmentation result
    - segmentation2: np.array, second segmentation result
    - foreground_mask: np.array, boolean array where True indicates foreground
    
    Returns:
    - combined_segmentation: np.array, the combined segmentation result
    """
    # Ensure the mask is boolean
    foreground_mask = foreground_mask.astype(bool)
    
    # Create a copy of the first segmentation to avoid modifying the original
    combined_segmentation = np.copy(segmentation1)
    
    # Use the mask to replace values in the first segmentation with the second segmentation
    combined_segmentation[foreground_mask] = segmentation2[foreground_mask]
    
    return combined_segmentation

def normalize_bands(arr):
    """
    Normalize each band of a ndim numpy array to the range [-1, 1].

    Parameters:
    arr (numpy.ndarray): ndim numpy array with shape (bands, height, width)

    Returns:
    numpy.ndarray: Normalized ndim numpy array with values scaled between 0 and 1
    """
    # Create an empty array with the same shape as the input array to store the normalized bands
    normalized_arr = np.empty_like(arr, dtype=np.float64)
    
    # Loop through each band (layer)
    for i in range(arr.shape[2]):
        
        #NOTE (Eric): Extract data
        band = arr[:,:,i]
       

        # mean_val = np.nanmean(band)
        # var_val = np.nanvar(band)
        min_val = band.min()
        max_val = band.max()
        
        #NOTE (Eric) Normalize the band by removing the mean and scaling to unit variance.
        # normalized_band = (band-mean_val) / var_val
        # normalized_arr[:,:,i] = normalized_band

        # Normalize the band to the range [-1, 1]
        normalized_band = 2 * (band - min_val) / (max_val - min_val) - 1

        #NOTE (Eric): Extract data and mask
        if isinstance(normalized_band, np.ma.MaskedArray):
            
            normalized_mask = normalized_band.mask
            normalized_band = normalized_band.data

            normalized_band[normalized_mask] = np.nan

        #NOTE (Eric): Normalize the image using a histogram equalization
        normalized_arr[:,:,i] = np.ma.array(equalize_adapthist(normalized_band, kernel_size=None, clip_limit=0.01, nbins=256), mask=normalized_mask)
    
    #NOTE (Eric): Uncomment if you want to visualize the normalized imagery
    # imshow_im = np.dstack([normalized_arr[:,:,3], normalized_arr[:,:,0], normalized_arr[:,:,1]])
    # plt.imshow(imshow_im)
    # plt.show()

    return normalized_arr

def extract_bounding_boxes(shapefile_path):
    """
    Extract the bounding boxes of features in a shapefile.

    Parameters:
    shapefile_path (str): Path to the shapefile

    Returns:
    list of tuples: List of bounding boxes, where each bounding box is represented as a tuple
                    (min_x, min_y, max_x, max_y)
    """
    # Open the shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    datasource = driver.Open(shapefile_path, 0)  # 0 means read-only

    if datasource is None:
        raise FileNotFoundError(f"Could not open {shapefile_path}")

    # Get the layer
    layer = datasource.GetLayer()

    # Loop through the features and extract bounding boxes
    bounding_boxes = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is not None:
            envelope = geom.GetEnvelope()  # Returns (min_x, max_x, min_y, max_y)
            bounding_boxes.append((envelope[0], envelope[2], envelope[1], envelope[3]))

    # Close the datasource
    datasource.Destroy()

    return bounding_boxes

def nrmlze_image_and_calc_indices(image_path: str):
    '''
    Function reads GeoTiff file as numpy array, calculates spectral indices,
    converts nodata and inf values to np.nan, and normalizes the array bands
    individually between -1 and 1.

    image_path: str - full path to GeoTiff image file

    '''

    assert isfile(image_path), f'[ERROR] File {image_path} not found. Exiting.'

    dataset = gdal.Open(image_path)

    #NOTE (Eric): Get nodata value
    nodataVal = dataset.GetRasterBand(1).GetNoDataValue()

    #NOTE (Eric): Extract band data and calculate indices
    bands = []
    for i in range(dataset.RasterCount):

        imageArr = dataset.GetRasterBand(i+1).ReadAsArray()
        
        #NOTE (Eric): Extract nodata and inf mask from imageArr
        cond1 = np.isnan(imageArr) # nan check
        cond2 = np.isinf(imageArr) # inf check
        cond3 = imageArr == nodataVal # nodata check 

        #NOTE (Eric): Conditional OR for masking
        noDataMask = (cond1 | cond2 | cond3)

        #NOTE (Eric): Mask the data
        maskedImage = np.ma.array(imageArr, mask=noDataMask)

        bands.append(maskedImage) 
    
    bands.append( (bands[3] - bands[0]) / (bands[3] + bands[0])) #NDVI
    bands.append( (bands[1] - bands[3]) / (bands[1] + bands[3])) #NDWI
    bands.append( ((bands[3] - bands[0]) / (bands[3] + bands[0])) * bands[3]) #NIRv 

    #NOTE (Eric): Stack the original RGBN bands and calculated indices into numpy array
    imageArr = np.dstack(bands)

    #NOTE (Eric): Mask the data again to handle nodata areas in the computed indices
    
    #NOTE (Eric): Extract nodata and inf mask from image
    cond1 = np.isnan(imageArr.data) # nan check
    cond2 = np.isinf(imageArr.data) # inf check
    cond3 = imageArr.data == nodataVal # nodata check 

    #NOTE (Eric): Conditional OR for masking
    noDataMask = (cond1 | cond2 | cond3)

    #NOTE (Eric): Mask the data
    maskedImage = np.ma.array(imageArr.data, mask=noDataMask)

    #NOTE (Eric): Normalize the image between -1 and 1
    nrmImage = normalize_bands(maskedImage)

    #NOTE (Eric): Extract the masked and normalized data as a numpy array (non-masked) and apply a nodata value
    image = nrmImage.data
    image[noDataMask] = np.nan


    return image

def nrmlze_image_and_calc_indices_glcm(image_path: str):
    '''
    Function reads GeoTiff file as numpy array, calculates spectral indices,
    converts nodata and inf values to np.nan, and normalizes the array bands
    individually between -1 and 1.

    image_path: str - full path to GeoTiff image file

    '''

    assert isfile(image_path), f'[ERROR] File {image_path} not found. Exiting.'

    dataset = gdal.Open(image_path)

    #NOTE (Eric): Get nodata value
    nodataVal = dataset.GetRasterBand(1).GetNoDataValue()

    #NOTE (Eric): Extract band data and calculate indices
    bands = []
    for i in range(dataset.RasterCount):

        imageArr = dataset.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
        
        #NOTE (Eric): Extract nodata and inf mask from imageArr
        cond1 = np.isnan(imageArr) # nan check
        cond2 = np.isinf(imageArr) # inf check
        cond3 = imageArr == nodataVal # nodata check 

        #NOTE (Eric): Conditional OR for masking
        noDataMask = (cond1 | cond2 | cond3)

        #NOTE (Eric): Mask the data
        maskedImage = np.ma.array(imageArr, mask=noDataMask)

        bands.append(maskedImage) 
    
    bands.append( ((bands[3] - bands[0]) / (bands[3] + bands[0])) * bands[3]) #NIRv
    bands.append( (bands[1] - bands[0]) / (bands[1] + bands[0] - bands[2])) #VARI VARI = (Green - Red)/ (Green + Red - Blue)
    bands.append( (bands[3] - bands[0]) / (bands[3] + bands[0])) #NDVI
    bands.append( fast_glcm_entropy(bands[1])) #Green entropy

    #NOTE (Eric): Stack the original RGBN bands and calculated indices into numpy array
    imageArr = np.dstack(bands)

    #NOTE (Eric): Mask the data again to handle nodata areas in the computed indices
    
    #NOTE (Eric): Extract nodata and inf mask from image
    cond1 = np.isnan(imageArr.data) # nan check
    cond2 = np.isinf(imageArr.data) # inf check
    cond3 = imageArr.data == nodataVal # nodata check 

    #NOTE (Eric): Conditional OR for masking
    noDataMask = (cond1 | cond2 | cond3)

    #NOTE (Eric): Mask the data
    maskedImage = np.ma.array(imageArr.data, mask=noDataMask)

    #NOTE (Eric): Normalize the image between -1 and 1
    nrmImage = normalize_bands(maskedImage)

    #NOTE (Eric): Extract the masked and normalized data as a numpy array (non-masked) and apply a nodata value
    image = nrmImage.data
    image[noDataMask] = np.nan


    return image

def nrmlze_image_and_calc_indices_subset(image_path: str, bounding_box):
    '''
    Function reads GeoTiff file and extracts as numpy array within specified bounding box.
    Function also calculates spectral indices, converts nodata and inf values to np.nan,
    and normalizes the array bands individually between -1 and 1. Raster and shapefile spatial reference
    should all be common.

    image_path: str - full path to GeoTiff image file
    bounding_box: tuple or list of bounding box coordinates for image subsetting (min_x, min_y, max_x, max_y)

    '''

    assert isfile(image_path), f'[ERROR] File {image_path} not found. Exiting.'

    #NOTE (Eric): Open the original raster dataset
    dataset = gdal.Open(image_path)

    #NOTE (Eric): Get src geotiff data type, nodata value, and spatial reference
    dataType = dataset.GetRasterBand(1).DataType
    nodataVal = dataset.GetRasterBand(1).GetNoDataValue()

    if nodataVal is None:   
        nodataVal = 0

    src_srs = dataset.GetSpatialRef()

    #NOTE (Eric): Gdal warp reference image to subset within bounding box (in-memory)
    gdal.Warp('/vsimem/raster.tif', dataset, multithread='YES', dstNodata = nodataVal,
                            outputType = dataType, outputBoundsSRS=src_srs, 
                            outputBounds=bounding_box, srcSRS=src_srs, dstSRS=src_srs)

    #NOTE (Eric): Extract the clipped dataset and close reference dataset
    dataset = None
    dataset = gdal.Open('/vsimem/raster.tif')

    #NOTE (Eric): Extract band data and calculate indices
    bands = []
    for i in range(dataset.RasterCount):

        imageArr = dataset.GetRasterBand(i+1).ReadAsArray()
        
        #NOTE (Eric): Extract nodata and inf mask from imageArr
        cond1 = np.isnan(imageArr) # nan check
        cond2 = np.isinf(imageArr) # inf check
        cond3 = imageArr == nodataVal # nodata check 

        #NOTE (Eric): Conditional OR for masking
        noDataMask = (cond1 | cond2 | cond3)

        #NOTE (Eric): Mask the data
        maskedImage = np.ma.array(imageArr, mask=noDataMask)

        bands.append(maskedImage) 
    
    bands.append( (bands[3] - bands[0]) / (bands[3] + bands[0])) #NDVI
    bands.append( (bands[1] - bands[3]) / (bands[1] + bands[3])) #NDWI
    bands.append( ((bands[3] - bands[0]) / (bands[3] + bands[0])) * bands[3]) #NIRv 

    #NOTE (Eric): Stack the original RGBN bands and calculated indices into numpy array
    imageArr = np.dstack(bands)

    #NOTE (Eric): Mask the data again to handle nodata areas in the computed indices
    
    #NOTE (Eric): Extract nodata and inf mask from image
    cond1 = np.isnan(imageArr.data) # nan check
    cond2 = np.isinf(imageArr.data) # inf check
    cond3 = imageArr.data == nodataVal # nodata check 

    #NOTE (Eric): Conditional OR for masking
    noDataMask = (cond1 | cond2 | cond3)

    #NOTE (Eric): Mask the data
    maskedImage = np.ma.array(imageArr.data, mask=noDataMask)

    #NOTE (Eric): Normalize the image between -1 and 1
    nrmImage = normalize_bands(maskedImage)

    #NOTE (Eric): Extract the masked and normalized data as a numpy array (non-masked) and apply a nodata value
    image = nrmImage.data
    image[noDataMask] = np.nan


    return image

def nrmlze_image_and_calc_indices_glcm_subset(image_path: str, bounding_box):
    '''
    Function reads GeoTiff file and extracts as numpy array within specified bounding box.
    Function also calculates spectral indices/glcm, converts nodata and inf values to np.nan,
    and normalizes the array bands individually between -1 and 1. Raster and shapefile spatial reference
    should all be common.

    image_path: str - full path to GeoTiff image file
    bounding_box: tuple or list of bounding box coordinates for image subsetting (min_x, min_y, max_x, max_y)

    '''

    assert isfile(image_path), f'[ERROR] File {image_path} not found. Exiting.'

    #NOTE (Eric): Open the original raster dataset
    dataset = gdal.Open(image_path)

    #NOTE (Eric): Get src geotiff data type, nodata value, and spatial reference
    dataType = dataset.GetRasterBand(1).DataType
    nodataVal = dataset.GetRasterBand(1).GetNoDataValue()

    if nodataVal is None:   
        nodataVal = 0

    src_srs = dataset.GetSpatialRef()

    #NOTE (Eric): Gdal warp reference image to subset within bounding box (in-memory)
    gdal.Warp('/vsimem/raster.tif', dataset, multithread='YES', dstNodata = nodataVal,
                            outputType = dataType, outputBoundsSRS=src_srs, 
                            outputBounds=bounding_box, srcSRS=src_srs, dstSRS=src_srs)

    #NOTE (Eric): Extract the clipped dataset and close reference dataset
    dataset = None
    dataset = gdal.Open('/vsimem/raster.tif')

    #NOTE (Eric): Extract band data and calculate indices
    bands = []
    for i in range(dataset.RasterCount):

        imageArr = dataset.GetRasterBand(i+1).ReadAsArray()
        
        #NOTE (Eric): Extract nodata and inf mask from imageArr
        cond1 = np.isnan(imageArr) # nan check
        cond2 = np.isinf(imageArr) # inf check
        cond3 = imageArr == nodataVal # nodata check 

        #NOTE (Eric): Conditional OR for masking
        noDataMask = (cond1 | cond2 | cond3)

        #NOTE (Eric): Mask the data
        maskedImage = np.ma.array(imageArr, mask=noDataMask)

        bands.append(maskedImage) 
    
    
    bands.append( ((bands[3] - bands[0]) / (bands[3] + bands[0])) * bands[3]) #NIRv
    bands.append( (bands[1] - bands[0]) / (bands[1] + bands[0] - bands[2])) #VARI VARI = (Green - Red)/ (Green + Red - Blue)
    bands.append( (bands[3] - bands[0]) / (bands[3] + bands[0])) #NDVI
    bands.append( fast_glcm_entropy(bands[1])) #Green entropy

    #NOTE (Eric): Stack the original RGBN bands and calculated indices into numpy array
    imageArr = np.dstack(bands)

    #NOTE (Eric): Mask the data again to handle nodata areas in the computed indices
    
    #NOTE (Eric): Extract nodata and inf mask from image
    cond1 = np.isnan(imageArr.data) # nan check
    cond2 = np.isinf(imageArr.data) # inf check
    cond3 = imageArr.data == nodataVal # nodata check 

    #NOTE (Eric): Conditional OR for masking
    noDataMask = (cond1 | cond2 | cond3)

    #NOTE (Eric): Mask the data
    maskedImage = np.ma.array(imageArr.data, mask=noDataMask)

    #NOTE (Eric): Normalize the image between -1 and 1
    nrmImage = normalize_bands(maskedImage)

    #NOTE (Eric): Extract the masked and normalized data as a numpy array (non-masked) and apply a nodata value
    image = nrmImage.data
    image[noDataMask] = np.nan


    return image

def collect_object_statistics(segment_id, image, segmentation):
    segment_pixels = image[segmentation == segment_id]
    object_features = segment_features(segment_pixels)
    return segment_id, object_features

def collect_object_statistics_batch(batch_segment_ids, image, segmentation):
    batch_objects = []
    batch_object_ids = []
    for id in batch_segment_ids:
        object_features = collect_object_statistics(id, image, segmentation)
        batch_objects.append(object_features)
        batch_object_ids.append(id)
    return batch_object_ids, batch_objects

def segment_features(segment_pixels):

    """
    Function collects spatial and band statistics within generated segments

    """

    features = []
    npixels, nbands = segment_pixels.shape
    
    for b in range(nbands):

        #NOTE (Eric): Intialize empty list to store band statistics for object
        band_stats = []

        #NOTE (Eric): Collect stats using scipy describe
        stats = describe(segment_pixels[:, b], nan_policy='omit')
        
        #NOTE (Eric): Extract type info from stats. If masked array then object data is nan
        # and we will apply blanket value to stats
        minmax_type = type(stats.minmax[0])
        mean_type = type(stats.mean)
        var_type = type(stats.variance)

        if minmax_type is np.ma.core.MaskedArray:
            band_stats.append(0.0)
            band_stats.append(0.0)

        else:
            band_stats.append(stats.minmax[0])
            band_stats.append(stats.minmax[1])

        if mean_type is np.ma.core.MaskedArray:
            band_stats.append(0.0)
        
        else:
            band_stats.append(stats.mean)

        if var_type is np.ma.core.MaskedArray:
            band_stats.append(0.0)
        
        else:
            band_stats.append(stats.variance)
        
        # if math_isnan(stats.skewness):
        #     band_stats.append(0.0)
        
        # else:
        #     band_stats.append(stats.skewness)

        # if math_isnan(stats.kurtosis):
        #     band_stats.append(0.0)
        
        # else:
        #     band_stats.append(stats.kurtosis)

        if npixels == 1:
            # in this case the variance, skewness, and kurtosis = nan, change them to 0.0
            band_stats[3] = 0.0
            # band_stats[4] = 0.0
            # band_stats[5] = 0.0
        features += band_stats
    
    return(features)
    
def split_into_batches(segment_ids, batch_size):
    return [segment_ids[i:i + batch_size] for i in range(0, len(segment_ids), batch_size)]

def process_segments_cpu(segment_ids, image, segmentation, batch_size):
    objects = []
    object_ids = []
    print('\nBeginning batch processing')
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(collect_object_statistics_batch, batch, image, segmentation)
                   for batch in split_into_batches(segment_ids, batch_size)]
        for future in as_completed(futures):
            try:
                batch_object_ids, batch_objects = future.result(timeout=None)  # Add a timeout to each future
                objects.extend(batch_objects)
                object_ids.extend(batch_object_ids)
            except Exception as e:
                print(f"\nException in future result: {e}")
    print('\nBatch processing complete')
    return objects, object_ids

def applyClassifier(image_path: str, modelPath: str,
                   classFilename: str, subsetShapefile=False, saveSegments: bool = False,
                   gpu: bool = False, batchSize: int = 10, segmentsFn:str = 'outClassSegments.tif'):
    '''
    Function to apply Random Forest classifier. Function segments input image using simple non-iterative clustering (SNIC) algorithm
    and subsets the image if subsetting shapefile is provided. Segmentation is then used to collect megapixel object statistics.
    Classification is then applied to megapixel objects based on their statistics and the saved model being used. 
    Output classification is saved to a GeoTiff raster dataset.

    image_path: str - full path to GeoTiff image file to classify
    
    modelPath: str - full path to saved Random Forest classifier model
    
    classFilename: str - full path to output location for classified GeoTiff image
    
    subsetShapefile: str (optional) - full path to shapefile which will be used to subset the image and clip the output imagery     
    
    saveSegments: bool (optional) - boolean to decide if segments from SNIC should be saved to disk, defaults to False

    gpu: bool (optional) - Boolean to use GPU hardware or not, defaults to False
    
    batchSize: int (optional) - number of batches to run simultaneously, defaults to 10 -- be careful with large values

    segmentsFn: str (optional) - name of the output GeoTiff filename for the image segments used to collect image statistics before classification
    '''

    # NOTE (Eric): SNIC parameters
    numSegments = 22000
    compactness = 0.03 #Very little compactness gives our segments great felixbility in their shape

    if not isfile(subsetShapefile):
    
        # NOTE (Eric): Read the image
        image = nrmlze_image_and_calc_indices_glcm(image_path)
        print(f"\nImage shape: {image.shape}")

        #NOTE (Eric): Extract nodata and nodata mask
        ref_ds = gdal.Open(image_path)
        nodataVal = ref_ds.GetRasterBand(1).GetNoDataValue()
        ref_band = ref_ds.GetRasterBand(1)
        image_arr = ref_band.ReadAsArray()
        nodataMask = np.array(image_arr == nodataVal)
        image_arr = None

        #NOTE (Eric): Extract bands specifically for SAM segmentation (N,R,G)
        seg_im = np.dstack([image[:,:,3], image[:,:,0], image[:,:,1]])
        number_of_pixels = seg_im.shape[0] * seg_im.shape[1]
        
        # NOTE (Eric): Compute grid (SNIC)
        grid = compute_grid(seg_im.shape, numSegments)
        seeds = list(chain.from_iterable(grid))
        seed_len = len(seeds)

        # NOTE (Eric): Choose a distance metric (SNIC)
        distance_metric = create_augmented_snic_distance(seg_im.shape, seed_len, compactness)

        print(f"\nNumber of bands: {seg_im.shape[2]}, Number of seeds: {len(seeds)}, Compactness: {compactness}")

        #NOTE (Eric): Apply image segmentation with SNIC
        segmentation, distances, centroids = snic(seg_im.tolist(), seeds,
        compactness, nd_computations["nd"], distance_metric,
        update_func=lambda num_pixels: print("\nSegments processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))

        #NOTE (Eric): Convert SNIC segments to np.array
        segmentation = np.array(segmentation)

        print('\nSegmentation complete.')

        #NOTE (Eric): Apply nodata mask to segments
        segmentation[nodataMask] = -9999

        if saveSegments is True:
            driver = gdal.GetDriverByName('GTIFF')
            ref_ds = gdal.Open(image_path)
            XSize = ref_ds.RasterXSize
            YSize = ref_ds.RasterYSize
            out_ds = driver.Create(segmentsFn + f'_SAM-SNIC-{numSegments}Segments.tif', XSize, YSize,
                                    1, gdal.GDT_Int32)
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            out_ds.SetSpatialRef(ref_ds.GetSpatialRef())

            out_ds.GetRasterBand(1).SetNoDataValue(-9999)
            out_ds.GetRasterBand(1).WriteArray(np.array(segmentation))
            out_ds.GetRasterBand(1).FlushCache()
            out_ds.FlushCache()
            out_ds = None
        
        #NOTE (Eric): Collect spatial and band statistics from each segment
        # and store it with its unique segment id while ignoring nodata segments

        segment_ids = np.unique(segmentation)
        segment_ids_mask = segment_ids != -9999
        segment_ids = segment_ids[segment_ids_mask]

        #NOTE (Eric): Time the object statistics process
        print('\nBeginning statistics collection process')
        t1 = time.time()

        # Parallelize the statistics collection process
        objects, object_ids = process_segments_cpu(segment_ids, image, segmentation, batch_size=batchSize)

        t2 = time.time()
        time_elapsed = round(t2-t1, ndigits=2)
        
        print(f'\nTime to collect object statistics: {time_elapsed} seconds')

        #NOTE (Eric): Apply the classifier on the objects
        print(f'\nClassifying image {image_path}')
        objects = np.array([features for _, features in objects])
        
        #NOTE (Eric): Set neural network parameters
        input_size = objects.shape[1]
        hidden_size = 100
        output_size = 3 #NOTE (Eric): Number of output classes (1 - Trees/Shrubs, 2 - Grasses/Pervious, 3 - Urban/Impervious)

        # Initialize the model
        model = ClassyNN(input_size, hidden_size, output_size)

        # Load the trained model's state
        model.load_state_dict(torch.load(modelPath))
        model.eval()  # Set the model to evaluation mode

        # Assuming X_new is your new input data for prediction
        X_tensor = torch.tensor(objects, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():  # No need to compute gradients for prediction
            outputs = model(X_tensor)
            _, predicted_labels = torch.max(outputs, 1)  # Get the predicted labels

        # If you need to convert the labels back to the original range [1, num_classes]
        predicted_labels += 1

        # Convert to numpy array if needed
        predicted_labels = predicted_labels.numpy()

        clf = np.copy(segmentation)
        for segment_id, klass in zip(segment_ids, predicted_labels):
            clf[clf == segment_id] = klass

        print('\nPrediction applied to numpy array')
        mask = np.ones_like(clf)
        final_clf = np.multiply(clf, mask)
        final_clf[nodataMask] = 255

        #NOTE (Eric): Write the classification to disk
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(classFilename + '.tif', final_clf.shape[1], final_clf.shape[0], 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
        out_ds.SetProjection(ref_ds.GetProjection())
        out_ds.SetSpatialRef(ref_ds.GetSpatialRef())
        out_ds.GetRasterBand(1).SetNoDataValue(255)
        out_ds.GetRasterBand(1).WriteArray(final_clf)
        
        out_ds.FlushCache()
        out_ds = None
        


    if isfile(subsetShapefile):
        
        #NOTE (Eric): Extract the bounding boxes of all the subsetting shapefile features
        bounding_boxes = extract_bounding_boxes(subsetShapefile)

        #NOTE (Eric): Count number of bounding boxes for progress tracking
        numBoundingBoxes = len(bounding_boxes)
        
        #NOTE (Eric): Loop through the shapefiles and extract the training data
        for bb, bounding_box in enumerate(bounding_boxes):
            
            # NOTE (Eric): Read the image subset
            image = nrmlze_image_and_calc_indices_glcm_subset(image_path, bounding_box)
            print(f"\nImage shape: {image.shape}")

            #NOTE (Eric): Extract bands specifically for SAM segmentation (N,R,G)
            seg_im = np.dstack([image[:,:,3], image[:,:,0], image[:,:,1]])
            number_of_pixels = seg_im.shape[0] * seg_im.shape[1]

            # NOTE (Eric): Compute grid (SNIC)
            grid = compute_grid(seg_im.shape, numSegments)
            seeds = list(chain.from_iterable(grid))
            seed_len = len(seeds)

            # NOTE (Eric): Choose a distance metric (SNIC)
            distance_metric = create_augmented_snic_distance(seg_im.shape, seed_len, compactness)

            print(f"\nNumber of bands: {seg_im.shape[2]}, Number of seeds: {len(seeds)}, Compactness: {compactness}")

            #NOTE (Eric): Apply image segmentation with SNIC
            segmentation, distances, centroids = snic(seg_im.tolist(), seeds,
            compactness, nd_computations["nd"], distance_metric,
            update_func=lambda num_pixels: print("\nSegments processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))

            #NOTE (Eric): Convert SNIC segments to np.array
            segmentation = np.array(segmentation)

            print('\nSegmentation complete.')
          
            #NOTE (Eric): Open the original raster dataset and subset it with the current bounding box
            ref_ds = gdal.Open(image_path)

            #NOTE (Eric): Get src geotiff data type, nodata value, and spatial reference
            dataType = ref_ds.GetRasterBand(1).DataType
            nodataVal = ref_ds.GetRasterBand(1).GetNoDataValue()

            if nodataVal is None:   
                nodataVal = 0

            src_srs = ref_ds.GetSpatialRef()

            #NOTE (Eric): Gdal warp reference image to subset within bounding box (in-memory)
            gdal.Warp('/vsimem/raster.tif', ref_ds, multithread='YES', dstNodata = nodataVal,
                                    outputType = dataType, outputBoundsSRS=src_srs, 
                                    outputBounds=bounding_box, srcSRS=src_srs, dstSRS=src_srs)

            #NOTE (Eric): Extract the clipped dataset and close reference dataset
            ref_ds = None
            ref_ds = gdal.Open('/vsimem/raster.tif')

            #NOTE (Eric): Extract nodata and nodata mask
            nodataVal = ref_ds.GetRasterBand(1).GetNoDataValue()
            ref_band = ref_ds.GetRasterBand(1)
            image_arr = ref_band.ReadAsArray()
            nodataMask = np.array(image_arr == nodataVal)
            image_arr = None

            #NOTE (Eric): Apply nodata mask to segments
            segmentation[nodataMask] = -9999

            if saveSegments is True:
                driver = gdal.GetDriverByName('GTIFF')
                XSize = ref_ds.RasterXSize
                YSize = ref_ds.RasterYSize
                out_ds = driver.Create(segmentsFn + f'_SAM-SNIC-{numSegments}-{bb+1}.tif', XSize, YSize,
                                        1, gdal.GDT_Int32)
                out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
                out_ds.SetProjection(ref_ds.GetProjection())
                out_ds.SetSpatialRef(ref_ds.GetSpatialRef())

                out_ds.GetRasterBand(1).SetNoDataValue(-9999)
                out_ds.GetRasterBand(1).WriteArray(np.array(segmentation))
                out_ds.GetRasterBand(1).FlushCache()
                out_ds.FlushCache()
                out_ds = None
            
            #NOTE (Eric): Collect spatial and band statistics from each segment
            # and store it with its unique segment id while ignoring nodata segments

            segment_ids = np.unique(segmentation)
            segment_ids_mask = segment_ids != -9999
            segment_ids = segment_ids[segment_ids_mask]

            #NOTE (Eric): Time the object statistics process
            print('\nBeginning statistics collection process')

            #NOTE (Eric): Time the object statistics process
            print('\nBeginning statistics collection process')
            t1 = time.time()

            # Parallelize the statistics collection process
            objects, object_ids = process_segments_cpu(segment_ids, image, segmentation, batch_size=batchSize)


            t2 = time.time()
            time_elapsed = round(t2-t1, ndigits=2)
            
            print(f'\nTime to collect object statistics for grid subset {bb+1}: {time_elapsed} seconds')
            print(f'\n{round(((bb+1)/numBoundingBoxes)*100,2)}% of imagery processed')
        
            #NOTE (Eric): Apply the classifier on the objects
            print(f'\nClassifying image {image_path} subset {bb+1}')
            objects = np.array([features for _, features in objects])
            
            #NOTE (Eric): Set neural network parameters
            input_size = objects.shape[1]
            hidden_size = 100
            output_size = 3 #NOTE (Eric): Number of output classes (1 - Trees/Shrubs, 2 - Grasses/Pervious, 3 - Urban/Impervious)

            # Initialize the model
            model = ClassyNN(input_size, hidden_size, output_size)

            # Load the trained model's state
            model.load_state_dict(torch.load(modelPath))
            model.eval()  # Set the model to evaluation mode

            # Assuming X_new is your new input data for prediction
            X_tensor = torch.tensor(objects, dtype=torch.float32)

            # Make predictions
            with torch.no_grad():  # No need to compute gradients for prediction
                outputs = model(X_tensor)
                _, predicted_labels = torch.max(outputs, 1)  # Get the predicted labels

            # If you need to convert the labels back to the original range [1, num_classes]
            predicted_labels += 1

            # Convert to numpy array if needed
            predicted_labels = predicted_labels.numpy()

            clf = np.copy(segmentation)
            for segment_id, klass in zip(segment_ids, predicted_labels):
                clf[clf == segment_id] = klass

            print('\nPrediction applied to numpy array')
            mask = np.ones_like(clf)
            final_clf = np.multiply(clf, mask)
            final_clf[nodataMask] = 255

            #NOTE (Eric): Write the classification to disk
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(classFilename +f'_SAM-SNIC-{numSegments}-{bb+1}.tif', final_clf.shape[1], final_clf.shape[0], 1, gdal.GDT_Byte)
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            out_ds.SetSpatialRef(ref_ds.GetSpatialRef())
            out_ds.GetRasterBand(1).SetNoDataValue(255)
            out_ds.GetRasterBand(1).WriteArray(final_clf)
            
            out_ds.FlushCache()
            out_ds = None
            break
            
        

def main():
    '''
    Main function to read data, segment the image, and run the model classification.
    '''

    start_time = time.time()

    #NOTE (Eric): Define image and shapefile paths to iterate over classification - these need to match 1-to-1 (i.e., Oakland cannot match w/ LA)
    image_paths = ["E:\\SummerGSR2024\\NAIP\\LosAngeles\\LosAngelesNAIP2022.tif",
                    "E:\\SummerGSR2024\\NAIP\\Oakland\\OaklandNAIP2022.tif",
                      "E:\\SummerGSR2024\\NAIP\\Sacramento\\SacramentoNAIP2022.tif"]
    
    grid_paths = [r"E:\SummerGSR2024\Grids\LosAngeles\Classification\LosAngeles500mGridClassificationSubset.shp",
                  r"E:\SummerGSR2024\Grids\Oakland\Classification\Oakland500mGridClassificationSubset.shp",
                  r"E:\SummerGSR2024\Grids\Sacramento\Classification\Sacramento500mGridClassificationSubset.shp"]
    
    city_names = ["LosAngeles", "Oakland", "Sacramento"]
    
    #NOTE (Eric): Define output model paths for saving to disk
    model_path = "E:\\SummerGSR2024\\MLModels\\LA-Oak-Sac\\LA-Oak-Sac-SchoolYards-SNIC-GLCM-NN.pt"    
    assert isfile(model_path), f'[ERROR] Model file {model_path} not found. Exiting.'


    #NOTE (Eric): Iterate over the image and grid subset shapefile data for each urban center
    for image_path, grid_shp, city_name in  zip(image_paths, grid_paths, city_names):

        #NOTE (Eric): Assert that the files we're using actually exist
        assert isfile(image_path), f'\n[ERROR] Image file {image_path} not found for {city_name}. Exiting.'
        assert isfile(grid_shp), f'\n[ERROR] Subset grid shapefile {grid_shp} not found for {city_name}. Exiting.'
        
        #NOTE (Eric): Define output file names and paths for the classification and its segmentation
        outClassFilename = f"E:\\SummerGSR2024\\ImClassification\\{city_name}\\NN\\{city_name}ElemntarySchoolClassification"
        output_segments_path = f"E:\\SummerGSR2024\\ImSegmentation\\{city_name}\\Classification\\NN\\{city_name}ClassSegments"

        #NOTE (Eric): Book keeping on which image we're processing
        print(f'\nProcessing: {city_name}\nImage file: {image_path}')

        #NOTE (Eric): Run the classification code
        applyClassifier(image_path, model_path, outClassFilename, subsetShapefile=grid_shp,
                        saveSegments=True, segmentsFn=output_segments_path)

    

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\nTotal Classification time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()