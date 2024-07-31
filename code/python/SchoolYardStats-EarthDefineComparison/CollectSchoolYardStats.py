# Code calculates land cover areas and percent land cover for individual school yards within three cities in California. 
# Summary statistics of these calculations are written to .csv files.

# Author: Eric Romero
# Last edit: 7/31/2024

import geopandas as gpd
import pandas as pd

def calculate_school_yard_land_cover_areas(school_yards_path, land_cover_path, utm_crs):
    
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

    #NOTE (Eric): Group by school yard and land cover class
    grouped = joined.groupby(['School', 'DN'])['area'].sum().reset_index()

    #NOTE (Eric): Filter out DN = 255 (nodata regions)
    grouped = grouped[grouped['DN'] != 255]

    #NOTE (Eric): Merge with school yards to get total area for each school yard
    school_yard_areas = school_yards.copy()
    school_yard_areas['total_area'] = school_yard_areas.geometry.area

    merged = grouped.merge(school_yard_areas[['School', 'total_area']], on='School')

    #NOTE (Eric): Calculate percent area
    merged['percent_area'] = (merged['area'] / merged['total_area']) * 100

    #NOTE (Eric): Rename rows based on 'DN' values
    class_names = {
        1: 'Trees',
        2: 'Grasses/Pervious',
        3: 'Impervious',
        4: 'Buildings/Structures'
    }
    merged['LandCover'] = merged['DN'].map(class_names)

    #NOTE (Eric): Select and rename columns
    merged = merged[['School', 'LandCover', 'area', 'percent_area']]
    merged.columns = ['School', 'LandCover', 'Area (m2)', '%Area']

    #NOTE (Eric): Ensure correct data types
    merged['Area (m2)'] = merged['Area (m2)'].astype(float)
    merged['%Area'] = merged['%Area'].astype(float)

    return merged

#NOTE (Eric): Paths to shapefiles
results_by_school_yard = {}
out_csv_path = 'E:\\SummerGSR2024\\SummarySchoolYardStatistics\\IndvYards\\Smooth\\CleanAndSmooth\\'
cities = ["LosAngeles", "Oakland", "Sacramento"]
utm_epsgs = ['EPSG:32611', 'EPSG:32610', 'EPSG:32610'] #NOTE (Eric): Conversion for meaningful areas
results = {}

for city, utm_epsg in zip(cities, utm_epsgs):
    school_yards_path = f"E:\\SummerGSR2024\\Schools_Joined_min5b\\{city}\\{city}_Schools_Buffer10m_WGS84.shp"
    land_cover_path = f"E:\\SummerGSR2024\\Classification-OSM-Merged\\{city}\\Smooth\\CleanAndSmooth\\{city}Schools-Classified-OSM-CleanSmoothMerged.shp"
    result = calculate_school_yard_land_cover_areas(school_yards_path, land_cover_path, utm_epsg)
    results_by_school_yard[city] = result
    
    #NOTE (Eric): Write to CSV
    result.to_csv(f"{out_csv_path}{city}_school_yard_land_cover_areas.csv", index=False)

#NOTE (Eric): Display results
for city, result in results_by_school_yard.items():
    print(f"City: {city}")
    print(result)
