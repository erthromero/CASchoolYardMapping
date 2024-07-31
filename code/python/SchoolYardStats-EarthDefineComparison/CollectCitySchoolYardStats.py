# Code calculates land cover areas and percent land cover for all school yards within three cities in California. 
# Summary statistics of these calculations are written to .csv files.

# Author: Eric Romero
# Last edit: 7/31/2024

import geopandas as gpd
import pandas as pd

def calculate_city_land_cover_areas(school_yards_path, land_cover_path, utm_crs):
    #NOTE (Eric): Read shapefiles
    school_yards = gpd.read_file(school_yards_path)
    land_cover = gpd.read_file(land_cover_path)

    #NOTE (Eric): Ensure both datasets have the same CRS
    school_yards = school_yards.to_crs(land_cover.crs)

    #NOTE (Eric): Reproject to UTM CRS
    school_yards = school_yards.to_crs(utm_crs)
    land_cover = land_cover.to_crs(utm_crs)

    #NOTE (Eric): Perform spatial join
    joined = gpd.sjoin(land_cover, school_yards, how="inner")

    #NOTE (Eric): Calculate area of each land cover polygon
    joined['area'] = joined.geometry.area

    #NOTE (Eric): Group by land cover class - land cover class is designated by attribute 'DN'
    total_area_by_class = joined.groupby('DN')['area'].sum().reset_index()

    #NOTE (Eric): Filter out DN = 255 (nodata regions)
    total_area_by_class = total_area_by_class[total_area_by_class['DN'] != 255]

    #NOTE (Eric): Calculate total area
    total_area = total_area_by_class['area'].sum()

    #NOTE (Eric): Calculate percent area
    total_area_by_class['percent_area'] = (total_area_by_class['area'] / total_area) * 100

    #NOTE (Eric): Rename rows based on 'DN' values
    class_names = {
        1: 'Trees',
        2: 'Grasses/Pervious',
        3: 'Impervious',
        4: 'Buildings/Structures'
    }
    total_area_by_class['LandCover'] = total_area_by_class['DN'].map(class_names)

    #NOTE (Eric): Select and rename columns
    total_area_by_class = total_area_by_class[['LandCover', 'area', 'percent_area']]
    total_area_by_class.columns = ['LandCover', 'Area (m2)', '%Area']

    #NOTE (Eric): Ensure correct data types
    total_area_by_class['Area (m2)'] = total_area_by_class['Area (m2)'].astype(float)
    total_area_by_class['%Area'] = total_area_by_class['%Area'].astype(float)

    return total_area_by_class

#NOTE (Eric): Paths to shapefiles
out_csv_path = 'E:\\SummerGSR2024\\SummarySchoolYardStatistics\\AllCityYards\\Smooth\\CleanAndSmooth\\'
cities = ["LosAngeles", "Oakland", "Sacramento"]
utm_epsgs = ['EPSG:32611', 'EPSG:32610', 'EPSG:32610'] #Conversion for meaningful areas
results = {}

for city, utm_epsg in zip(cities, utm_epsgs):
    school_yards_path = f"E:\\SummerGSR2024\\Schools_Joined_min5b\\{city}\\{city}_Schools_Buffer10m_WGS84.shp"
    land_cover_path = f"E:\\SummerGSR2024\\Classification-OSM-Merged\\{city}\\Smooth\\CleanAndSmooth\\{city}Schools-Classified-OSM-CleanSmoothMerged.shp"
    result = calculate_city_land_cover_areas(school_yards_path, land_cover_path, utm_epsg)
    results[city] = result
    
    #NOTE (Eric): Write to CSV
    result.to_csv(f"{out_csv_path}{city}_city_land_cover_areas.csv", index=False)

#NOTE (Eric): Display results
for city, result in results.items():
    print(f"City: {city}")
    print(result)
