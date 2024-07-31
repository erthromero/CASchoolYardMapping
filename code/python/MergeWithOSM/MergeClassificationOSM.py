# This code creates a merged shapefile between two vector datasets, both of which represent land cover data with an attribute labeled
# 'DN'. The two shapefiles are provided as arguments to this code, as well as the output filename of the merged shapefile. The first input shapefile
# must have 'DN' classes 1, 2, and 3. The second input shapefile must have 'DN' class 4. This can all be edited in the code below to accept any number of
# classes, just edit the variable 'sort_order'. Originally, I had written this code in an attempt to have certain classes supercede others in the drawing 
# order (e.g., I wanted DN class 1 - Tree/Shrub to supercede DN class 4 - Buildings/Structures). This ultimately failed as it seems the Python tools available 
# for vector geometries don't allow for this or make it more difficult than I would have liked to attempt. Now, this code simply merges the two vector data into one, 
# similar to the "Merge Vector Layers" tool in QGIS. In order to create an improved drawing order, where buildings appear below trees, you have to first run this code
# and then run the "CleanUpMergedData.py" code after.

# Author: Eric Romero, PhD Candidate, UC Berkeley
# Last edited: 7/31/2024

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from os.path import isfile

def mergeClassyOSMChangeDrawingOrder(shp1: str, shp2: str, shp3: str):

    """
    Function merges two shapefiles into a third output shapefile. Shapefiles must contain a shared attribure 'DN' which
    represents land cover. Similar functioning to "Merge Vector Layers" tool in QGIS, but specified for land cover class-
    ification in urban elementary school campuses in CA for USDA. If you want to have more 'DN' classes be merged, just add them
    to the dictionary in the 'sort_order' variable (e.g., if you have a new DN value '5' which represents a land cover class,
    your dictionary would look like this: sort_order = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}).

    shp1: str - full path to first land cover shapefile
    shp2: str - full path to second land cover shapefile
    shp3: str - full path to output merged shapefile

    """

    #NOTE (Eric): Assert shapefiles exist
    assert isfile(shp1), f'[ERROR] File {shp1} not found. Exiting.'
    assert isfile(shp2), f'[ERROR] File {shp2} not found. Exiting.'

    #NOTE (Eric):Read the shapefiles
    shapefile1 = gpd.read_file(shp1)
    shapefile2 = gpd.read_file(shp2)

    #NOTE (Eric):Merge the shapefiles
    merged = gpd.GeoDataFrame(pd.concat([shapefile1, shapefile2], ignore_index=True))

    #NOTE (Eric): Define the custom sort order
    sort_order = {1: 1, 2: 2, 3: 3, 4: 4}

    #NOTE (Eric):Sort the merged DataFrame based on 'DN' values
    merged['sort_order'] = merged['DN'].map(sort_order)
    merged_sorted = merged.sort_values(by='sort_order')
    
    #NOTE (Eric):Write the resulting shapefile
    merged_sorted.to_file(shp3)

if __name__ == "__main__": 
    from sys import argv

    shp1 = argv[1]
    shp2 = argv[2]
    shp3 = argv[3]

    try:
        mergeClassyOSMChangeDrawingOrder(shp1, shp2, shp3)

    except Exception as e:
        print(f'\n[ERROR] Expected python MergeClassificationsOSM.py /path/to/ClassificationPolygons.shp /path/to/OSM-Buildings.shp /path/to/outputMergedPolygons.shp, but got {argv[:]} instead. Exiting.')
        print(f'\n{e}')
