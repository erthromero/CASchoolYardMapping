# CASchoolYardMapping
This repository houses all source code and documentation related to image classification of school yards in major California urban centers by researchers with UC Berkeley and the USDA.

# Instructions

Step 0 : Creating a suitable processing environment

        - It is reccommended to use Anaconda to construct the local Python processing
          environment for this process.

        - In this repository, download and use the file TrainLassifySchoolsNAIP.yml 
          to create a suitable environment for this image processing task. Instructions
          on how to do this can be found here: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

Step 1: Javascript export of NAIP RGBN aerial imagery in Google Earth Engine (GEE)

          ##################### IMPORTANT NOTE ##########################
          You will need a GEE account with access to the javascript
          code editor interface. Sign up and access to this
          interface can be gained here: https://earthengine.google.com/
          ###############################################################


        - Use the following link to open the NAIP mosaic image downloader
          https://code.earthengine.google.com/12e095bff916f8b95c2a58fe7ee8c03a?noload=true

        - Alternatively, you can naviagte to the file in this repository: 
          code/javascript/MosaicClipDownloadNAIP.js and copy its contents into a blank
          GEE code editor script

        - Run the NAIP mosaic image downloader script with a custom AOI geometry and 
          selected image dates

        - After having run the code, navigate to the 'Tasks' pane in the GEE code editor 
          interface, this will give you the option to export your image to your Google Drive with whatever account is associated GEE. From your Google Drive you can then download the imagery to your local machine.

Step 2: Model Training

        ############################ IMPORTANT NOTE ####################################
        You will need to have a pre-collected "ground truth" training datseet for
        the following model training steps. It is reccomended to do this in GIS 
        software, either ArcGIS Pro, ArcMap, or QGIS. Training point data should have 
        an attribute table with one integer type attribute for unique land use land cover classes labelled as integers corresponding to the land cover determined via visual inspection in NAIP imagery. A nice tutorial on this collection can be found as part of the following tutorial: https://youtu.be/p4hVA5dvV0g?feature=shared.

        It is required that for each city that will be processed, a polygon shapefile grid of 500m2 processing zones be created over areas in the NAIP imagery only where valid training data exists. There are nice tutorials for doing this in both ArcGIS Pro (https://pro.arcgis.com/en/pro-app/latest/help/production/clearing-grids/create-a-grid.htm) or QGIS (https://youtu.be/UO9CvUS66Xk?feature=shared)
        ##################################################################################
        
        - Navigate to folder /code/python/Training in this repository and select either  
          the Train-MultiCity-RFClassifier-SNIC-GLCM.py (Random Forest Model) or the 
          Train-MultiCity-NNClassifier-SNIC-GLCM.py (Neural Network Model). I recomend using the Random Forest for simplicity. The neural nework was more of a test which was not used in any of the final products from this project.

        - The code can be run after editing just a few variables in the 'main()' function 
          to align with the file structure of your local system:

            image_paths: list - list of strings which point to NAIP image file paths on local machine

            training_shapefile_paths: list - list of strings which point to shapefile paths that contain ESRI shapefiles of point vector training data.
                
            grid_paths: list - list of strings which point to 500 m2 grid shapefiles for iterating over and subsetting NAIP imagery

            city_names: list - list of strings denoting the cities which will be processed. These will also point the code to file paths for the output segmentation and classified image filenames (see variable 'output_segments_path')

            output_model_path: string - string which points to path of the output model file to be saved

Step 3: Image Classification

        ############################ IMPORTANT NOTE ####################################
        It is required that for each city that will be processed, a polygon shapefile grid of 500m2 processing zones be created over areas in the NAIP imagery where processing is desired to take place. There are nice tutorials for doing this in both ArcGIS Pro (https://pro.arcgis.com/en/pro-app/latest/help/production/clearing-grids/create-a-grid.htm) or QGIS (https://youtu.be/UO9CvUS66Xk?feature=shared)
        ##################################################################################
        
        - Navigate to folder /code/python/Classification/ in this repository and select 
          either  the Classify-MultiCity-RFClassifier-SNIC-GLCM.py (Random Forest Model) or the Classify-MultiCity-NNClassifier-SNIC-GLCM.py (Neural Network Model) depending on which kind of model you chose to train. Random Forest is the recommended model.

        - The code can be run after editing just a few variables in the 'main()' function 
          to align with the file structure of your local system:

            image_paths: list - list of strings which point to NAIP image file paths on local machine
        
            grid_paths: list - list of strings which point to 500 m2 grid shapefiles for iterating over and subsetting NAIP imagery

            city_names: list - list of strings denoting the cities which will be processed. These will also point the code to file paths for the output segmentation and 
            classified image filenames (see variables 'outClassFilename' and 'output_segments_path')

            model_path: string - file path to the saved model on local machine

        - Once the code has run, your file system will have many classified images 
          depending on the number of gridded subsets you have per processing region. I reccomend merging and clipping these regions together as necessary in either QGIS or ArcGIS software.


Step 4: Validation

        ############################ IMPORTANT NOTE ####################################
        You will need to have a pre-collected "ground truth" validation datset for
        the following model validation steps. It is reccomended to do this in GIS 
        software, either ArcGIS Pro, ArcMap, or QGIS. Validation data should be in the form of polygon shapefiles, which can then be converted into a raster dataset of integer values. The integer values of the validation raster should match the unique classification schema of the classified image rasters. 
        
        Nice documentaiton for converting polygon to raster data can be found for both QGIS (https://docs.qgis.org/3.34/en/docs/user_manual/processing_algs/gdal/vectorconversion.html#rasterize-overwrite-with-attribute) as well as ArcGIS Pro (https://pro.arcgis.com/en/pro-app/latest/tool-reference/conversion/polygon-to-raster.htm). I reccomend first making a polygon shapefile with ana attribute table schema matching the training data schema (e.g., an attribute LULC which contains the land cover class integer for each validation polygon), and then converting this polygon to a validation land cover raster with one of the methods above. 

        As a final note, the dimensions as spatial extents of the classified image raster and the validation raster MUST be IDENTICAL.
        ##################################################################################

        - Navigate to /code/python/Validation/ and use the code file PolygonAccuracy.py to perform the validation. 

        - This code requires 4 input variables:

            confusion_mat_path: str - path to  output confusion matrix
        
            polygon_rast: path to rasterized validation polygons
        
            class_rast: path to classified raster
        
            im_res: float - image resolution in meters
        
        - Once the code has run, you will have a confusion matrix .csv file which can be 
          used to calculate additional accuracy statistics. A nice tutorial for accuracy metric calculations can be found here: https://gsp.humboldt.edu/olm/Courses/GSP_216/lessons/accuracy/metrics.html


