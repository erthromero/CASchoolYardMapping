/*###########################################################################

USDA School Yard Mapping Project
Author: Eric Romero, PhD Candidate UC Berkeley
Last Edit: 6/6/2024

#############################################################################                          
                              NOTE
Script to mosiac and clip NAIP imagery from 2022 over designated AOI
This NAIP imagery will be used locally to collect, veryify, and perform
validation of image classification over select elementary school yards in 
California urban centers

Change AOI geomtery import as needed to alter bounds of NAIP mosaic

###########################################################################*/


//NOTE (Eric): Load four 2022 NAIP over AOI
var naip2022 = ee.ImageCollection('USDA/NAIP/DOQQ')
  .filterBounds(AOI)
  .filterDate('2022-01-01', '2022-12-31');

//NOTE (Eric): Spatially mosaic the images in the collection and display.
var mosaic = naip2022.mosaic();
Map.addLayer(mosaic, {}, 'spatial mosaic');
Map.addLayer(AOI);

//NOTE (Eric): Name the image. Change this as needed
var imName = 'OaklandNAIP2022';

//NOTE (Eric): Export Image to Drive
Export.image.toDrive({image: mosaic, description: imName, folder:'USDASchoolYards', fileNamePrefix:imName,
scale: 0.6, region:AOI, maxPixels:1e13, fileFormat: 'GeoTIFF'});