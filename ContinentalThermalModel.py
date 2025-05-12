def cltm(fname_config_file):
    
    ## Import the modules that the model will use
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from osgeo import gdal, osr
    import configparser
    import os
    from permetrics.regression import RegressionMetric
    from tqdm import tqdm
    from datetime import datetime
    from netCDF4 import Dataset
    
    # Get the current time as the start time
    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')   # Get current time
    print('Model start time:', start_time)
    print('')

    # Now read configuration file
    config = configparser.ConfigParser(comment_prefixes = ('#', ';'))    # Remove comment
    config.read(fname_config_file)

    # Removing comments after each variable
    for section in config.sections():
        for key, value in config.items(section):
            config.set(section, key, value.split('#')[0].strip())

    # Accessing parameters from each section:
    # Model boundary parameters
    GeographicBoundary = {
        'lat_max': float(config.get('GeographicBoundary', 'lat_max')),
        'lat_min': float(config.get('GeographicBoundary', 'lat_min')),
        'lon_max': float(config.get('GeographicBoundary', 'lon_max')),
        'lon_min': float(config.get('GeographicBoundary', 'lon_min')),
        'maximum_pixel_size': int(config.get('GeographicBoundary', 'maximum_pixel_size'))}
    lat_max = GeographicBoundary['lat_max']                           # Maximum latitude (North); °
    lat_min = GeographicBoundary['lat_min']                           # Minimum latitude (South); °
    lon_max = GeographicBoundary['lon_max']                           # Maximum longitude (East); °
    lon_min = GeographicBoundary['lon_min']                           # Minimum longitude (West); °
    maximum_pixel_size = GeographicBoundary['maximum_pixel_size']     # Maximum pixel size when quadtree setting is on; m

    # Select to use which temperature-depth constraints
    ConstraintSelection = {
        'Run_D95': config.getboolean('ConstraintSelection', 'Run_D95'),
        'Run_Curie': config.getboolean('ConstraintSelection', 'Run_Curie'),
        'Run_Moho': config.getboolean('ConstraintSelection', 'Run_Moho'),
        'Run_LAB': config.getboolean('ConstraintSelection', 'Run_LAB')}
    Run_D95 = ConstraintSelection['Run_D95']         # Decide to run D95 or not
    Run_Curie = ConstraintSelection['Run_Curie']     # Decide to run Curie depth or not
    Run_Moho = ConstraintSelection['Run_Moho']       # Decide to run Moho dataset or not
    Run_LAB = ConstraintSelection['Run_LAB']         # Decide to run LAB or not

    # D95 modeling parameters
    if Run_D95 == True:
        D95 = {
            'fname_EQ_catalog': config.get('D95', 'fname_EQ_catalog'),
            'min_EQ_cutoff': int(config.get('D95', 'min_EQ_cutoff')),
            'quadtree': config.getboolean('D95', 'quadtree'),
            'fixed_pixel_size': int(config.get('D95', 'fixed_pixel_size'))}
        fname_EQ_catalog = D95['fname_EQ_catalog']         # File name to earthquake catalog (csv)
        min_EQ_cutoff = D95['min_EQ_cutoff']               # Minimum earthquake cutoff of each pixel
        quadtree = D95['quadtree']                         # Quadtree setting; on = True, off = False
        fixed_pixel_size = D95['fixed_pixel_size']         # Fixed size when quadtree setting is off; m

    # Input constraints for thermal model
    InputData = {
        'fname_heatflow': config.get('InputData', 'fname_heatflow'), 
        'fname_Curie': config.get('InputData', 'fname_Curie'),
        'fname_MohoT': config.get('InputData', 'fname_MohoT'),
        'fname_MohoT_uncer': config.get('InputData', 'fname_MohoT_uncer'),
        'fname_Mohoz': config.get('InputData', 'fname_Mohoz'),
        'fname_LAB': config.get('InputData', 'fname_LAB')}
    fname_heatflow = InputData['fname_heatflow']             # File name to surface heat flow data (tif)
    fname_Curie = InputData['fname_Curie']                   # File name to Curie depth (tif)
    fname_MohoT = InputData['fname_MohoT']                   # File name to Moho temperature data (tif)
    fname_MohoT_uncer = InputData['fname_MohoT_uncer']       # File name to Moho temperature uncertainty data (tif)
    fname_Mohoz = InputData['fname_Mohoz']                   # File name to crustal thickness data (tif)
    fname_LAB = InputData['fname_LAB']                       # File name to LAB depth (tif)

    # Thermal model parameter
    ModelParameters = {
        'Curie_z_uncer': float(config.get('ModelParameters', 'Curie_z_uncer')),
        'Curie_depth_T': float(config.get('ModelParameters', 'Curie_depth_T')),
        'Curie_T_uncer': float(config.get('ModelParameters', 'Curie_T_uncer')),
        'Iteration': int(config.get('ModelParameters', 'Iteration')),
        'T0_range': np.array([float(x) for x in config.get('ModelParameters', 'T0_range').split(',')]),
        'k_range': np.array([float(x) for x in config.get('ModelParameters', 'k_range').split(',')]),
        'H0_range': np.array([float(x) for x in config.get('ModelParameters', 'H0_range').split(',')]),
        'qm_range': np.array([float(x) for x in config.get('ModelParameters', 'qm_range').split(',')]),
        'goodfit': float(config.get('ModelParameters', 'goodfit')),
        'moderatefit': float(config.get('ModelParameters', 'moderatefit')),
        'D95_T_config': float(config.get('ModelParameters', 'D95_T_config')),
        'Max_goodfit_profile': int(config.get('ModelParameters', 'Max_goodfit_profile')),
        'LAB_T_config': float(config.get('ModelParameters', 'LAB_T_config')),
        'Uncertainty_box_fitting': config.getboolean('ModelParameters', 'Uncertainty_box_fitting'),
        'Diffusion_2D': config.getboolean('ModelParameters', 'Diffusion_2D'),
        'Diffusion_time': float(config.get('ModelParameters', 'Diffusion_time'))}
    Curie_z_uncer = ModelParameters['Curie_z_uncer']              # Curie depth uncertainty; km
    Curie_depth_T = ModelParameters['Curie_depth_T']                      # Curie depth temperature; °C
    Curie_T_uncer = ModelParameters['Curie_T_uncer']                      # Curie depth temperature uncertainty; °C
    Iteration = ModelParameters['Iteration']                              # Number of iterations during thermal modeling
    T0_range = ModelParameters['T0_range']                                # Range of surface temperature; °C
    k_range = ModelParameters['k_range']                                  # Range of thermal conductivity; W m-1 °C-1
    H0_range = ModelParameters['H0_range']                                # Range of surface radiogenic heat production; W m-3
    qm_range = ModelParameters['qm_range']                                # Range of mantle heat flow; W m-2
    goodfit = ModelParameters['goodfit']                                  # Good-fit cutoff for NRMSE
    moderatefit = ModelParameters['moderatefit']                          # Moderate-fit cutoff for NRMSE
    D95_T_config = ModelParameters['D95_T_config']                        # D95 temperature; °C
    Max_goodfit_profile = ModelParameters['Max_goodfit_profile']          # Maximum number of good-fit profiles until the model is satisfied
    LAB_T_config = ModelParameters['LAB_T_config']                        # LAB temperature; °C
    Uncertainty_box_fitting = ModelParameters['Uncertainty_box_fitting']  # Option if the model use temperature-depth uncertainty fitting
    Diffusion_2D = ModelParameters['Diffusion_2D']                        # Option if model will conduct 2D lateral heat diffusion at each depth slice
    Diffusion_time = ModelParameters['Diffusion_time']                    # Total lateral heat diffusion time; Myr

    ## Here resample heat flow data
    # Here set model geographic extents
    maximum_pixel_size_deg = (maximum_pixel_size / 1000) / 111                          # Largest bin width; meter to unit [=] degree
    minimum_pixel_size = maximum_pixel_size / 8                                         # Smallest bin width; unit [=] m
    minimum_pixel_size_deg = (minimum_pixel_size / 1000) / 111                          # Smallest bin width; meter to unit [=] degree 
    
    # Longitude and latitude model boundary
    print('Longitude bound:', lon_min, lon_max)
    print('Latitude bound:', lat_min, lat_max)
    print('Model resolution:', minimum_pixel_size_deg, 'degree')
    print('')
    
    # Resample heat flow data from USGS: Mordensky and DeAngelo, 2023
    # https://www.sciencebase.gov/catalog/item/63090a9cd34e3b967a8c19c4
    # Reproject heat flow data from NAD 1983_Albers to WGS 84
    window = (lon_min, lat_max, lon_max, lat_min + minimum_pixel_size_deg)    # Define cropping window
    heatflow = gdal.Open(fname_heatflow)
    warp = gdal.Warp('', heatflow, format = 'MEM', dstSRS = 'EPSG:4326',
                outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
    print('Heat flow data dimension:')
    print('x Dimension:', warp.RasterXSize, ' ', 'y dimension:', warp.RasterYSize, ' ', 'Pixel resolution (°):', warp.GetGeoTransform()[1])
    print(' ')

    # Save tif as array
    hf = warp.ReadAsArray()               # Read tif as numpy array
    hf = np.flipud(hf)                    # Flip data upside-down
    hf = np.where(hf < -10, np.nan, hf)   # Replace value smaller -10 with NaN
    warp = None                           # Close tif

    # Define grid shape
    shape = np.shape(hf)

    # Resample Curie depth data
    if Run_Curie:
        ## Here resample curie depth data
        Curie_depth = gdal.Open(fname_Curie)
        warp = gdal.Warp('', Curie_depth, format = 'MEM', dstSRS = 'EPSG:4326',
                    outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
        print('Curie depth data dimension:')
        print('x Dimension:', warp.RasterXSize, ' ', 'y dimension:', warp.RasterYSize, ' ', 'Pixel resolution (°):', warp.GetGeoTransform()[1])
        print(' ')
    
        # Save tif as array
        Curie_depth_arr = warp.ReadAsArray()                                        # Read tif as numpy array
        Curie_depth_arr = np.flipud(Curie_depth_arr)                                # Flip data upside-down
        Curie_depth_arr = np.where(Curie_depth_arr < -10, np.nan, Curie_depth_arr)  # Replace value smaller -10 with NaN
        warp = None                                                                 # Close tif

    # Resample Moho data
    if Run_Moho:
        ## Here resample all input constraint data
        # Resample Moho temperature data (default with Schutt et al., 2018 Geology)
        # https://pubs.geoscienceworld.org/gsa/geology/article-abstract/46/3/219/525801/Moho-temperature-and-mobility-of-lower-crust-in
        # Reproject Moho temperature data to WGS 84
        Moho_T = gdal.Open(fname_MohoT)
        window = (lon_min, lat_max, lon_max, lat_min + minimum_pixel_size_deg)    # Define cropping window
        warp = gdal.Warp('', Moho_T, format = 'MEM', dstSRS = 'EPSG:4326',
                    outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
        print('Moho temperature data dimension:')
        print('x Dimension:', warp.RasterXSize, ' ', 'y dimension:', warp.RasterYSize, ' ', 'Pixel resolution (°):', warp.GetGeoTransform()[1])
        print(' ')
    
        # Save tif as array
        Moho_T_arr = warp.ReadAsArray()                              # Read tif as numpy array
        Moho_T_arr = np.flipud(Moho_T_arr)                           # Flip data upside-down
        Moho_T_arr = np.where(Moho_T_arr < -10, np.nan, Moho_T_arr)  # Replace value smaller -10 with NaN
        warp = None                                                  # Close tif

        # Resample Moho depth data (default with Buehler and Shearer, 2017 JGR:SE)
        # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB013265
        # Reproject Moho depth data to WGS 84
        zm = gdal.Open(fname_Mohoz)
        warp = gdal.Warp('', zm, format = 'MEM', dstSRS = 'EPSG:4326',
                    outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
        print('Moho depth data dimension:')
        print('x Dimension:', warp.RasterXSize, ' ', 'y dimension:', warp.RasterYSize, ' ', 'Pixel resolution (°):', warp.GetGeoTransform()[1])
        print(' ')
    
        # Save tif as array
        zm_arr = warp.ReadAsArray()                     # Read tif as numpy array
        zm_arr = np.flipud(zm_arr)                      # Flip data upside-down
        zm_arr = np.where(zm_arr < -10, np.nan, zm_arr) # Replace value smaller -10 with NaN
        warp = None                                     # Close tif

        # Resample Moho temperature uncertainty data (default with Schutt et al., 2018 Geology)
        # https://pubs.geoscienceworld.org/gsa/geology/article-abstract/46/3/219/525801/Moho-temperature-and-mobility-of-lower-crust-in
        # Reproject Moho temperature data to WGS 84
        Moho_T_uncer = gdal.Open(fname_MohoT_uncer)
        warp = gdal.Warp('', Moho_T_uncer, format = 'MEM', dstSRS = 'EPSG:4326',
                    outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
        print('Moho temperature uncertainty dimension:')
        print('x Dimension:', warp.RasterXSize, ' ', 'y dimension:', warp.RasterYSize, ' ', 'Pixel resolution (°):', warp.GetGeoTransform()[1])
        print(' ')
    
        # Save tif as array
        Moho_T_uncer_arr = warp.ReadAsArray()                                           # Read tif as numpy array
        Moho_T_uncer_arr = np.flipud(Moho_T_uncer_arr)                                  # Flip data upside-down
        Moho_T_uncer_arr = np.where(Moho_T_uncer_arr < -10, np.nan, Moho_T_uncer_arr)   # Replace value smaller -10 with NaN
        warp = None                                                                     # Close tif

    # Here resample the LAB data
    if Run_LAB:     
        LAB = gdal.Open(fname_LAB)
        warp = gdal.Warp('', LAB, format = 'MEM', dstSRS = 'EPSG:4326',
                    outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
        print('LAB depth dimension:')
        print('x Dimension:', warp.RasterXSize, ' ', 'y dimension:', warp.RasterYSize, ' ', 'Pixel resolution (°):', warp.GetGeoTransform()[1])
        print(' ')
    
        # Save tif as array
        LAB_arr = warp.ReadAsArray()                            # Read tif as numpy array
        LAB_arr = np.flipud(LAB_arr)                            # Flip data upside-down
        LAB_arr = np.where(LAB_arr < -10, np.nan, LAB_arr)      # Replace value smaller -10 with NaN
        warp = None                                             # Close tif
        
    if Run_D95:
        ## Here prepare for D95 modeling
        # Load Seismic data
        EQ_data = pd.read_csv(fname_EQ_catalog)   # Read combined earthquake catalog
        EQ_lon = EQ_data['lon']                   # Extract longitude
        EQ_lat = EQ_data['lat']                   # Extract latitude
        EQ_DEPTH = EQ_data['dep']                 # Extract hypocenter depth
    
        # Set D-value: Percentile for defining seismogenic depth; 95%
        D95 = 95
    
        # For numbering
        nx = 0
        ny = 0
    
        # Run D95 with quadtree structure if input configuration allows
        # Binning earthquake data and calculate D-depth: Quadtree structure
        if quadtree == True:
            # Define empty array for D95 binning
            N = np.zeros(shape)
            D_depth = np.zeros(shape)
            D95_uncer = np.zeros(shape)
    
            # Boolean array for quad tree structure method; to evaluate if each pixel has enough earthquake data (n >= 30)
            bool_tf = np.array((True, False))
    
            # Define quad tree sizes (x by x)
            quad_size = 1
            quad_size_2 = quad_size * 2                                               # Second-order tree shape
            quad_size_4 = quad_size_2 * 2                                             # Third-order tree shape
            quad_size_8 = quad_size_4 * 2                                             # Fourth-order tree shape
    
            ## Here start D95 modeling
            # Search goes spatially from North to South and West to East for each row
            # North to south (y direction)
            # ***FIRST ORDER TREE***
            print('Start D95 calculation with quadtree structure.')
            print('')
            for yi in tqdm(np.linspace(lat_max, lat_min, int(np.ceil(shape[0] / quad_size_8)))):       # (start, end, length) 
                # West to east (x direction)
                for xi in np.linspace(lon_min, lon_max, int(np.ceil(shape[1] / quad_size_8))):         # (start, end, length)
    
                    # Initial pixel size equal to maximum pixel size
                    increment_adapt = maximum_pixel_size_deg
                    # Find the matching earthquake latitude in that bin
                    ind_lat = np.where((EQ_lat <= yi) & (EQ_lat > (yi - increment_adapt)))
                    # Find the matching earthquake longitude in that bin
                    ind_lon = np.where((EQ_lon >= xi) & (EQ_lon < (xi + increment_adapt)))
                    ind_latlon = np.intersect1d(ind_lat, ind_lon)                      # Find the matching latitude-longitude pairs

                    # If the amount of earthquake at the parent bin is less than 30
                    if len(ind_latlon) < min_EQ_cutoff: 
                        for yf in range(nx, nx + 8, 1):
                            for xf in range(ny, ny + 8, 1):
                                # If the index location is smaller than the array size
                                if (yf <= (shape[1] - 1)) and (xf <= (shape[0] - 1)):
                                    D_depth[xf][yf] = np.nan        # Define that pixel as NaN
                                    D95_uncer[xf][yf] = np.nan      # Define that pixel as NaN
        
                    # If the number of earthquake in the parent bin is greater than 120, proceed to calculate D95 in higher-order tree structure
                    elif len(ind_latlon) > (min_EQ_cutoff * 4):
            
                        # ***SECOND ORDER TREE***
                        increment_adapt = increment_adapt / 2                                     # Update pixel size in second-order tree; Size_max / 2
        
                        ind_latlon_2 = np.empty((quad_size_2, quad_size_2), dtype = object)       # 2 x 2 grid within a larger grid for storing EQ indices
    
                        ind_latlon_2_bool = np.zeros((quad_size_2, quad_size_2))                  # 2 x 2 grid within a larger grid for filtering
    
                        nf = 1      # Factor to make sure it is binned in the correct location
            
                        # Loop through 2 x 2 grid within the larger grid
                        for yf in range(0, quad_size_2, 1):
                            for xf in range(0, quad_size_2, 1):
                                # Find the matching earthquake latitude in that bin
                                ind_lat_2 = np.where((EQ_lat <= (yi - (increment_adapt * yf))) & (EQ_lat > (yi - (increment_adapt * (yf + nf)))))
                                # Find the matching earthquake longitude in that bin
                                ind_lon_2 = np.where((EQ_lon >= (xi + (increment_adapt * xf))) & (EQ_lon < (xi + (increment_adapt * (xf + nf)))))
                                ind_latlon_2[xf][yf] = np.intersect1d(ind_lat_2, ind_lon_2)     # Find the matching latitude-longitude pairs
    
                                # Second-order tree filtering: Make sure each grid has at less 30 earthquakes
                                if len(ind_latlon_2[xf][yf]) >= min_EQ_cutoff:    # If that grid is greater or equal to 30 earthquakes
                                    ind_latlon_2_bool[xf][yf] = True       # Define that grid as 'True'
                                else:                                      # If that grid is less than 30 earthquakes
                                    ind_latlon_2_bool[xf][yf] = False      # Define that grid as 'Flase'
                        
                        # Project 2 x 2 grid into a 8 x 8 grid within the larger grid
                        ind_latlon_2_v2 = ind_latlon_2[np.repeat(np.arange(ind_latlon_2.shape[0]), 4), :]
                        ind_latlon_2_v2 = ind_latlon_2_v2[:, np.repeat(np.arange(ind_latlon_2_v2.shape[1]), 4)]
    
                        # Second-order tree filtering result
                        ind_latlon_2_bool_inters = np.intersect1d(ind_latlon_2_bool, bool_tf)
    
                        # If each grid in a 2 x 2 setting has at least 30 earthquakes --> proceed to third-order tree (4 x 4 grid)
                        if len(ind_latlon_2_bool_inters) == 1 and ind_latlon_2_bool_inters[0] == True:
                
                            # ***THIRD ORDER TREE***
                            increment_adapt = increment_adapt / 2
            
                            ind_latlon_4 = np.empty((quad_size_4, quad_size_4), dtype = object)       # 4 x 4 grid within a larger grid for storing EQ indices
    
                            ind_latlon_4_bool = np.zeros((quad_size_4, quad_size_4))                  # 4 x 4 grid within a larger grid for filtering
                
                            # Loop through third-order tree, a 4 x 4 grid within the larger grid
                            for yf in range(0, quad_size_4, 1):
                                for xf in range(0, quad_size_4, 1):  
                                    # Find the matching earthquake latitude in that bin
                                    ind_lat_4 = np.where((EQ_lat <= (yi - (increment_adapt * yf))) & (EQ_lat > (yi - (increment_adapt * (yf + nf)))))
                                    # Find the matching earthquake longitude in that bin
                                    ind_lon_4 = np.where((EQ_lon >= (xi + (increment_adapt * xf))) & (EQ_lon < (xi + (increment_adapt * (xf + nf)))))
                                    ind_latlon_4[xf][yf] = np.intersect1d(ind_lat_4, ind_lon_4)       # Find the matching latitude-longitude pairs
                        
                                    # Third-order tree (4 x 4 grid) filtering: Make sure each grid has at less 30 earthquakes
                                    if len(ind_latlon_4[xf][yf]) > min_EQ_cutoff:   # If that grid is greater or equal to 30 earthquakes
                                        ind_latlon_4_bool[xf][yf] = True     # Define that grid as 'True'
                                    else:                                    # If that grid is less than 30 earthquakes
                                        ind_latlon_4_bool[xf][yf] = False    # Define that grid as 'Flase'
                            
                            # Project 4 x 4 grid into a 8 x 8 grid within the larger grid            
                            ind_latlon_4_v2 = ind_latlon_4[np.repeat(np.arange(ind_latlon_4.shape[0]), 2), :]
                            ind_latlon_4_v2 = ind_latlon_4_v2[:, np.repeat(np.arange(ind_latlon_4_v2.shape[1]), 2)]
    
                            # Third-order tree filtering result
                            ind_latlon_4_bool_inters = np.intersect1d(ind_latlon_4_bool, bool_tf)
                
                            # If each grid in a 4 x 4 setting (third-order tree) has at least 30 earthquakes --> proceed to fourth-order tree (8 x 8 grid)
                            if len(ind_latlon_4_bool_inters) == 1 and ind_latlon_4_bool_inters[0] == True:
                    
                                ## ***FOURTH-ORDER TREE***
                                increment_adapt = increment_adapt / 2
                
                                ind_latlon_8 = np.empty((quad_size_8, quad_size_8), dtype = object)  # 8 x 8 grid within a larger grid for storing EQ indices
                    
                                ind_latlon_8_bool = np.zeros((quad_size_8, quad_size_8))                 # 8 x 8 grid within a larger grid for filtering 
    
                                # Loop through fourth-order tree, a 8 x 8 grid within the larger grid
                                for yf in range(0, quad_size_8, 1):
                                    for xf in range(0, quad_size_8, 1):
                                        # Find the matching earthquake latitude in that bin
                                        ind_lat_8 = np.where((EQ_lat <= (yi - (increment_adapt * yf))) & (EQ_lat > (yi - (increment_adapt * (yf + nf)))))
                                        # Find the matching earthquake longitude in that bin
                                        ind_lon_8 = np.where((EQ_lon >= (xi + (increment_adapt * xf))) & (EQ_lon < (xi + (increment_adapt * (xf + nf)))))
                                        ind_latlon_8[xf][yf] = np.intersect1d(ind_lat_8, ind_lon_8)     # Find the matching latitude-longitude pairs
    
                                        # Third order tree (8 x 8 grid) filtering: Make sure each grid has at less 30 earthquakes
                                        if len(ind_latlon_8[xf][yf]) >= min_EQ_cutoff:  # If that grid is greater or equal to 30 earthquakes
                                            ind_latlon_8_bool[xf][yf] = True    # Define that grid as 'True'
                                        else:                                   # If that grid is less than 30 earthquakes
                                            ind_latlon_8_bool[xf][yf] = False   # Define that grid as 'Flase'
                                
                                # Fourth-order tree filtering result
                                ind_latlon_8_bool_inters = np.intersect1d(ind_latlon_8_bool, bool_tf)
    
                                # If each grid in an 8 x 8 setting (fourth-order tree) has at least 30 earthquakes --> Calculate D95 in fourth-order tree
                                if len(ind_latlon_8_bool_inters) == 1 and ind_latlon_8_bool_inters[0] == True:
                                    # Loop through actual model location
                                    for yf in range(nx, nx + quad_size_8, 1):        # Loop through y direction (North to South)
                                        for xf in range(ny, ny + quad_size_8, 1):    # Loop through x direction (West to East)
                                            if (yf <= (shape[1] - 1)) and (xf <= (shape[0] - 1)):
                                                # Get the correct index
                                                x_ind = xf - ny        
                                                y_ind = yf - nx
                                
                                                N[xf][yf] = len(ind_latlon_8[x_ind][y_ind])                       # Get the amount of earthquake at that bin  
                                                sorted_depth = np.sort(EQ_DEPTH.iloc[ind_latlon_8[x_ind][y_ind]])     # Sort depth in ascending order
                                                D_depth[xf][yf] = np.percentile(sorted_depth, D95)                    # Calculate the D95 depth
    
                                                # Calculate uncertainty with bootstrap resampling
                                                D_depth_resample = np.zeros((1000))    # Define an empty array with a length of 1000 to store sampled D95
                                                resample_n = len(sorted_depth)         # Define the amount of resample for iteration
    
                                                # Start resampling
                                                for i in range(1000):    # Resample for 1000 times
                                                    resample_dep = np.random.choice(sorted_depth, size = resample_n, replace = True)#Resample earthquake depth 
                                                    resample_dep = np.sort(resample_dep)                                            # Sort the resample depth
                                                    D_depth_resample[i] = np.percentile(resample_dep, D95) # Calculate D95 and append it to the resampled array
    
                                                std = np.std(D_depth_resample)# Get the standard deviation from the resampled distribution as uncertainty
                                                D95_uncer[xf][yf] = std          # Append uncertainty to empty array
                                
                                # If each grid in an 8 x 8 setting (fourth-order tree) doesn't have enough EQ --> Calculate D95 in third-order tree
                                else:
                                    # Loop through actual model location
                                    for yf in range(nx, nx + quad_size_8, 1):      # Loop through y direction (North to South)
                                        for xf in range(ny, ny + quad_size_8, 1):  # Loop through x direction (West to East)
                                            if (yf <= (shape[1] - 1)) and (xf <= (shape[0] - 1)):
                                                # Get the correct index
                                                x_ind = xf - ny
                                                y_ind = yf - nx
                                
                                                N[xf][yf] = len(ind_latlon_4_v2[x_ind][y_ind])                     # Get the amount of earthquake at that bin 
                                                sorted_depth = np.sort(EQ_DEPTH.iloc[ind_latlon_4_v2[x_ind][y_ind]])   # Sort depth in ascending order
                                                D_depth[xf][yf] = np.percentile(sorted_depth, D95)                     # Calculate and append the D95 depth
    
                                                # Calculate uncertainty with bootstrap resampling
                                                D_depth_resample = np.zeros((1000)) # Define an empty array with a length of 1000 to store sampled D95
                                                resample_n = len(sorted_depth)      # Define the amount of resample for iteration
    
                                                # Start resampling
                                                for i in range(1000):     # Resample for 1000 times
                                                    resample_dep = np.random.choice(sorted_depth, size = resample_n, replace = True) # Resample hypocenter
                                                    resample_dep = np.sort(resample_dep)                   # Sort the resample depth
                                                    D_depth_resample[i] = np.percentile(resample_dep, D95) # Calculate D95 and append it to the resampled array
                        
                                                std = np.std(D_depth_resample)   # Get the standard deviation from the resampled distribution as uncertainty
                                                D95_uncer[xf][yf] = std          # Append uncertainty to empty array
                                
                            # If each grid in an 4 x 4 setting (third-order tree) doesn't have at least 30 earthquakes --> Calculate D95 in second-order tree              
                            else:
                                # Loop through actual model location
                                for yf in range(nx, nx + quad_size_8, 1):        # Loop through y direction (North to South)
                                    for xf in range(ny, ny + quad_size_8, 1):    # Loop through x direction (West to East)
                                        if (yf <= (shape[1] - 1)) and (xf <= (shape[0] - 1)):
                                            # Get the correct index
                                            x_ind = xf - ny
                                            y_ind = yf - nx
                            
                                            N[xf][yf] = len(ind_latlon_2_v2[x_ind][y_ind])                       # Get the amount of earthquake at that bin 
                                            sorted_depth = np.sort(EQ_DEPTH.iloc[ind_latlon_2_v2[x_ind][y_ind]]) # Sort depth in ascending order
                                            D_depth[xf][yf] = np.percentile(sorted_depth, D95)                   # Calculate and append the D95 depth
    
                                            # Calculate uncertainty with bootstrap resampling
                                            D_depth_resample = np.zeros((1000))   # Define an empty array with a length of 1000 to store sampled D95
                                            resample_n = len(sorted_depth)        # Define the amount of resample for iteration
    
                                            # Start resampling
                                            for i in range(1000):       # Resample for 1000 times
                                                resample_dep = np.random.choice(sorted_depth, size = resample_n, replace = True) # Resample earthquake depth
                                                resample_dep = np.sort(resample_dep)                                             # Sort the resample depth
                                                D_depth_resample[i] = np.percentile(resample_dep, D95)  # Calculate D95 and append it to the resampled array
                        
                                            std = np.std(D_depth_resample)     # Get the standard deviation from the resampled distribution as uncertainty
                                            D95_uncer[xf][yf] = std            # Append uncertainty to empty array
                            
                        # If each grid in an 2 x 2 setting (second-order tree) doesn't have at least 30 earthquakes --> Calculate D95 in first-order tree   
                        else:
                            # Loop through actual model location
                            for yf in range(nx, nx + quad_size_8, 1):       # Loop through y direction (North to South)
                                for xf in range(ny, ny + quad_size_8, 1):   # Loop through x direction (West to East)
                                    if (yf <= (shape[1] - 1)) and (xf <= (shape[0] - 1)):
                                        N[xf][yf] = len(ind_latlon)                           # Get the amount of earthquake at that bin 
                                        sorted_depth = np.sort(EQ_DEPTH.iloc[ind_latlon])     # Sort depth in ascending order
                                        D_depth[xf][yf] = np.percentile(sorted_depth, D95)    # Calculate and append the D95 depth
    
                                        # Calculate uncertainty with bootstrap resampling
                                        D_depth_resample = np.zeros((1000))                  # Define an empty array with a length of 1000 to store sampled D95
                                        resample_n = len(sorted_depth)                       # Define the amount of resample for iteration
    
                                        # Start resampling
                                        for i in range(1000):         # Resample for 1000 times
                                            # Resample earthquake depth
                                            resample_dep = np.random.choice(sorted_depth, size = resample_n, replace = True) 
                                            # Sort the resample depth
                                            resample_dep = np.sort(resample_dep)                                             
                                            # Calculate D95 and append it to the resampled array
                                            D_depth_resample[i] = np.percentile(resample_dep, D95)                  

                                        # Get the standard deviation from the resampled distribution as reported uncertainty
                                        std = np.std(D_depth_resample)
                                        # Append uncertainty to empty array
                                        D95_uncer[xf][yf] = std               
    
                    else:
                        # Loop through actual model location
                        for yf in range(nx, nx + quad_size_8, 1):        # Loop through y direction (North to South)
                            for xf in range(ny, ny + quad_size_8, 1):    # Loop through x direction (West to East)
                                if (yf <= (shape[1] - 1)) and (xf <= (shape[0] - 1)):
                                
                                    N[xf][yf] = len(ind_latlon)                            # Get the amount of earthquake at that bin 
                                    sorted_depth = np.sort(EQ_DEPTH.iloc[ind_latlon])      # Sort depth in ascending order
                                    D_depth[xf][yf] = np.percentile(sorted_depth, D95)     # Calculate and append the D95 depth
    
                                    # Calculate uncertainty with bootstrap resampling
                                    D_depth_resample = np.zeros((1000))         # Define an empty array with a length of 1000 to store sampled D95
                                    resample_n = len(sorted_depth)              # Define the amount of resample for iteration
    
                                    # Start resampling
                                    for i in range(1000):        # Resample for 1000 times
                                        # Resample earthquake depth
                                        resample_dep = np.random.choice(sorted_depth, size = resample_n, replace = True)
                                        # Sort the resample depth
                                        resample_dep = np.sort(resample_dep)
                                        # Calculate D95 and append it to the resampled array
                                        D_depth_resample[i] = np.percentile(resample_dep, D95)

                                    # Get the standard deviation from the resampled distribution as reported uncertainty
                                    std = np.std(D_depth_resample)
                                    D95_uncer[xf][yf] = std          # Append uncertainty to empty array
    
                    # Update ny during each loop from West to East
                    nx = nx + quad_size_8
        
                # Reset ny after looping through each row    
                nx = 0
    
                # Update nx after during each loop from North to South
                ny = ny + quad_size_8
    
            ## Here export D95 modeling result
            # Convert empty bin to nan
            D_depth = np.where(D_depth == 0, np.nan, D_depth)
            D95_uncer = np.where(D95_uncer == 0, np.nan, D95_uncer)
            N = np.where(N == 0, np.nan, N)
    
        # If user select fixed bin width D95 modeling
        elif quadtree == False:
            ## Here set model geographic extents
            fixed_pixel_size_deg = (fixed_pixel_size / 1000) / 111                  # Fixed bin width; meter to unit [=] degree 

            fixed_shape = [len(np.arange(lat_max, lat_min + fixed_pixel_size_deg, -fixed_pixel_size_deg)), \
                           len(np.arange(lon_min, lon_max + fixed_pixel_size_deg, -fixed_pixel_size_deg))]
                           
            # Longitude and latitude model boundary
            print('Longitude bound:', lon_min, lon_max)
            print('Latitude bound:', lat_min, lat_max)
            print('D95 resolution:', fixed_pixel_size_deg, 'degree')
            print('')
    
            # Create empty array to store D95 result
            N = np.zeros(fixed_shape)
            D_depth = np.zeros(fixed_shape)
            D95_uncer = np.zeros(fixed_shape)
    
            ## Search goes spatially from North to South and West to East for each row
            # North to south (y direction)
            print('Start D95 calculation with fixed pixel size.')
            print('')
            for yi in tqdm(np.linspace(lat_max, lat_min, fixed_shape[0])):                       # (start, end, shape) 
                # West to east (x direction)
                for xi in np.linspace(lon_min, lon_max, fixed_shape[1]):                         # (start, end, shape)
            
                    ind_lat = np.where((EQ_lat <= yi) & (EQ_lat > (yi - fixed_pixel_size)))
                    ind_lon = np.where((EQ_lon >= xi) & (EQ_lon < (xi + fixed_pixel_size)))
            
                    ind_latlon = np.intersect1d(ind_lat, ind_lon)
                    N[nx][ny] = len(ind_latlon)
            
                    if len(ind_latlon) >= min_EQ_cutoff:
                        sorted_depth = np.sort(EQ_DEPTH.iloc[ind_latlon])                 # Sort depth in ascending order
                        D_depth[nx][ny] = np.percentile(sorted_depth, D95)                # Calculate the D95 depth
    
                        # Calculate uncertainty with bootstrap resampling
                        D_depth_resample = np.zeros((1000))                  # Define an empty array with a length of 1000 to store sampled D95
                        resample_n = len(sorted_depth)                       # Define the amount of resample for iteration
    
                        # Start resampling
                        for i in range(1000):        # Resample for 1000 times
                            resample_dep = np.random.choice(sorted_depth, size = resample_n, replace = True)    # Resample earthquake depth
                            resample_dep = np.sort(resample_dep)                                                # Sort the resample depth
                            D_depth_resample[i] = np.percentile(resample_dep, D95)                 # Calculate D95 and append it to the resampled array
                        
                        std = np.std(D_depth_resample) # Get the standard deviation from the resampled distribution as reported uncertainty
                        D95_uncer[xf][yf] = std        # Append uncertainty to empty array
            
                ## Update ny during each loop from West to East
                ny = ny + 1
        
            ## Reset ny after looping through each row    
            ny = 0
    
            ## Update nx after during each loop from North to South
            nx = nx + 1
    
            # Convert empty bin to nan
            D_depth = np.where(D_depth == 0, np.nan, D_depth)
            D95_uncer = np.where(D95_uncer == 0, np.nan, D95_uncer)
            N = np.where(N == 0, np.nan, N)
    
            # Here create a GeoTIF for fixed pixel size D95
            D_depth_shape = np.shape(D_depth)    # Get shape of D95
            height = D_depth_shape[0]            # Define height dimension
            width = D_depth_shape[1]             # Define width dimension
            bands = 1                            # Only 1 band
            
            # Get geotransform
            geotransform = (lon_min, fixed_pixel_size_deg, 0, lat_max, 0, -fixed_pixel_size_deg)  
            
            # Create a tif and define its spatial information
            driver = gdal.GetDriverByName('MEM')                    
            D95_tif = driver.Create('', width, height, bands, gdal.GDT_Float32)    # Create a tif
            D95_tif.SetGeoTransform(geotransform)
            
            # Set coordinate system and insert data
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)                                               # Set the spatial reference to EPSG:4326 (WGS 84)
            D95_tif.SetProjection(srs.ExportToWkt())
            D95_tif.GetRasterBand(1).WriteArray(D_depth)
    
            # Resample D95 to minimum pixel size
            D95_tif = gdal.Warp('', D95_tif, format = 'MEM', dstSRS = 'EPSG:4326',
                            outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
            # Save tif as array
            D_depth = D95_tif.ReadAsArray()                     # Read tif as numpy array
            D_depth = np.flipud(D_depth)                        # Flip data upside-down
            D_depth = np.where(D_depth < -10, np.nan, D_depth)  # Replace value smaller -10 with NaN
    
            D95_tif = None
    
            # Here create a GeoTIF for fixed pixel size D95 uncertainty
             # Get geotransform
            geotransform = (lon_min, fixed_pixel_size_deg, 0, lat_max, 0, -fixed_pixel_size_deg)  
            
            # Create a tif and define its spatial information
            driver = gdal.GetDriverByName('MEM')                    
            D95_uncer_tif = driver.Create('', width, height, bands, gdal.GDT_Float32)    # Create a tif
            D95_uncer_tif.SetGeoTransform(geotransform)
            
            # Set coordinate system and insert data
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)                                                     # Set the spatial reference to EPSG:4326 (WGS 84)
            D95_uncer_tif.SetProjection(srs.ExportToWkt())
            D95_uncer_tif.GetRasterBand(1).WriteArray(D95_uncer)
    
            # Resample D95 uncertainty to minimum pixel size
            D95_uncer_tif = gdal.Warp('', D95_uncer_tif, format = 'MEM', dstSRS = 'EPSG:4326',
                            outputBounds = window, xRes = minimum_pixel_size_deg, yRes = minimum_pixel_size_deg)
    
            # Save tif as array
            D95_uncer = D95_uncer_tif.ReadAsArray()                   # Read tif as numpy array
            D95_uncer = np.flipud(D95_uncer)                          # Flip data upside-down
            D95_uncer = np.where(D95_uncer < -10, np.nan, D95_uncer)  # Replace value smaller -10 with NaN
            D95_uncer_tif = None                                      # Close tif

    ## Here prepare for thermal modeling
    # Make empty arrays to store results
    Crustal_T = np.empty(shape, dtype = object)               # Make an empty 3D array for appending crustal temperature
    R2 = np.zeros(shape)                                      # Make an empty array for appending R2
    dTdz = np.zeros(shape)                                    # Make an empty array for appending geothermal gradient
    dTdz_250_max = np.zeros(shape)                            # Make an empty array for appending peak geotherms below 250C
    T_20km = np.zeros(shape)                                  # Make an empty array for appending temperature at 25km depth
    Qs_model = np.zeros(shape)                                # Make an empty array for appending modeled surface heat flow
    Qc_model = np.zeros(shape)                                # Make an empty array for appending heat flow contributed from heat production
    Qm_model = np.zeros(shape)                                # Make an empty array for appending modeled mantle heat flow
    Qc_Qs = np.zeros(shape)                                   # Make an empty array to store Qc/Qs ratio

    # Make an empty array for appending 1 or 0 depending on if pixel is constructed with D95
    # 0 is without D95; 1 is with D95
    With_or_withoutD95_mask = np.full(shape, np.nan) 

    # initialize default length of good-fit results
    ind_good_len = 0

    ## Here run thermal model
    # Now loop through all data one by one (nested loop)
    print('Start thermal modeling:')
    print('')
    for i_y in tqdm(range(shape[0])):                # Loop through rows (in y direction)
        for i_x in range(shape[1]):                  # Loop through columns (in x direction)

            # List to check NaN
            conditions = []
            condition_indices = []
            
            # Define data from each bin
            Qs = hf[i_y][i_x] / 1000                  # Surface heat flow; mW m-2 to W m-2
            conditions.append(~np.isnan(Qs))          # If Qs is not NaN --> True
            condition_indices.append('Qs')            # Add the index name for Qs
            
            if Run_D95:
                D95_depth = np.round(D_depth[i_y][i_x] * 1000, -3)         # D95; km to m and round to 1000th
                D95_depth_uncer = np.round(D95_uncer[i_y][i_x] * 1000, -3) # D95 uncertainty; km to m and round to 1000th
                D95_temp = D95_T_config                                    # D95 temperature; °C
                D95_temp_uncer = 100                                       # D95 temperature uncertainty; °C
                conditions.append(~np.isnan(D95_depth_uncer))              # If D95 is not NaN --> True
                condition_indices.append('D95')                            # Add the index name for mid-crustal depth
                
            if Run_Curie:
                Curie_depth = np.round(Curie_depth_arr[i_y][i_x] * 1000, -3) # Curie depth; km to m and round to 1000th
                Curie_depth_uncer = np.round(Curie_z_uncer * 1000, -3)       # Curie depth uncertainty; m
                Curie_temp = Curie_depth_T                                   # Curie depth temperature; °C
                Curie_temp_uncer = Curie_T_uncer                             # Curie depth temperature uncertainty; °C
                conditions.append(~np.isnan(Curie_depth))                    # If Curie depth is not NaN --> True
                condition_indices.append('Curie_depth')                      # Add the index name for mid-crustal depth
                 
            if Run_Moho:
                Moho_z = np.round(zm_arr[i_y][i_x] * 1000, -3)         # Crustal thickness; km to m and round to 1000th
                Moho_z_uncer = 7000                                    # Crustal thickness uncertainty; m
                Moho_T = Moho_T_arr[i_y][i_x]                          # Moho temperature; °C
                Moho_T_uncer = Moho_T_uncer_arr[i_y][i_x]              # Moho temperature uncertainty; °C
                conditions.append(~np.isnan(Moho_z))                   # If Moho depth is not NaN --> True
                condition_indices.append('Moho_z')                     # Add the index name for Moho depth
                conditions.append(~np.isnan(Moho_T))                   # If Moho temperature is not NaN --> True
                condition_indices.append('Moho_T')                     # Add the index name for Moho temperature
                
            if Run_LAB:
                LAB_depth = np.round(LAB_arr[i_y][i_x] * 1000, -3)  # LAB depth; km to m and round to 1000th
                LAB_depth_uncer = 7000                              # LAB depth uncertainty; m
                LAB_temp = LAB_T_config                             # LAB temperature; °C
                LAB_temp_uncer = 150                                # LAB temperature uncertainty; °C
                conditions.append(~np.isnan(LAB_depth))             # If LAB depth is not NaN --> True
                condition_indices.append('LAB_depth')               # Add the index name for LAB depth

            # If D95 is deeper and hotter than the Moho --> set the data to NaN
            if (Run_Moho) and (Run_D95) and (~np.isnan(D95_depth)) and (~np.isnan(Moho_z)) and (~np.isnan(Moho_T)):
                if (D95_depth >= Moho_z) or (D95_temp >= Moho_T):
                    D95_depth = np.nan
                    # Update condition list
                    if 'D95' in condition_indices:
                        conditions[condition_indices.index('D95')] = False

            # If Curie is deeper and hotter than the Moho --> set the data to NaN
            if Run_Moho and Run_Curie and (~np.isnan(Curie_depth)) and (~np.isnan(Moho_z)) and (~np.isnan(Moho_T)):
                if (Curie_depth >= Moho_z) or (Curie_temp >= Moho_T):
                    Curie_depth = np.nan
                    # Update condition list
                    if 'Curie_depth' in condition_indices:
                        conditions[condition_indices.index('Curie_depth')] = False

            # If the Moho is deeper and hotter than LAB --> set the data to NaN
            if Run_Moho and Run_LAB and (~np.isnan(Moho_z)) and (~np.isnan(Moho_T)) and (~np.isnan(LAB_depth)):
                if (Moho_z >= LAB_depth) or (Moho_T >= LAB_temp):
                    Moho_z = np.nan
                    # Update condition list
                    if 'Moho_z' in condition_indices:
                        conditions[condition_indices.index('Moho_z')] = False

            # If D95 is deeper than Curie depth --> set Curie depth to NaN
            if Run_D95 and Run_Curie and (~np.isnan(D95_depth)) and (~np.isnan(Curie_depth)):
                if (D95_depth >= Curie_depth):
                    Curie_depth = np.nan
                    # Update condition list
                    if 'Curie_depth' in condition_indices:
                        conditions[condition_indices.index('Curie_depth')] = False

            # If D95 is deeper than LAB --> set D95 to NaN
            if Run_D95 and Run_LAB and (~np.isnan(D95_depth)) and (~np.isnan(LAB_depth)):
                if (D95_depth >= LAB_depth):
                    D95_depth = np.nan
                    # Update condition list
                    if 'D95' in condition_indices:
                        conditions[condition_indices.index('D95')] = False

            # If Curie depth is deeper than LAB --> set Curie depth to NaN
            if Run_Curie and Run_LAB and (~np.isnan(Curie_depth)) and (~np.isnan(LAB_depth)):
                if (Curie_depth >= LAB_depth):
                    Curie_depth = np.nan
                    # Update condition list
                    if 'Curie_depth' in condition_indices:
                        conditions[condition_indices.index('Curie_depth')] = False

            # If Curie depth is negative value --> set Curie depth to NaN
            if Run_Curie and (~np.isnan(Curie_depth)):
                if (Curie_depth <= 0):
                    Curie_depth = np.nan
                    # Update condition list
                    if 'Curie_depth' in condition_indices:
                        conditions[condition_indices.index('Curie_depth')] = False

            # First pads Crustal_T array (the array to store thermal profiles) with NaN
            Crustal_T[i_y][i_x] = np.array([np.nan])

            conditions_v2 = []
            condition_indices_v2 = []
            for iii in range(len(conditions)):
                if conditions[iii] == True:
                    conditions_v2.append(conditions[iii])
                    condition_indices_v2.append(condition_indices[iii])
            
            # If data is NaN --> don't run them
            if all(conditions):

                # Set two arrays for temperature and depth observations based on active datasets
                T_obs = [10]  # Surface temperature = 10 °C
                z_obs = [0]   # Surface depth = 0 m
                T_obs_uncer = [0]
                z_obs_uncer = [0]

                # Append D95 depth and temperature, and their uncertainties to the observed list
                if Run_D95:
                    T_obs.append(D95_temp)
                    z_obs.append(D95_depth)
                    T_obs_uncer.append(D95_temp_uncer)
                    z_obs_uncer.append(D95_depth_uncer)

                # Append Curie depth and temperature, and their uncertainties to the observed list
                if Run_Curie:
                    T_obs.append(Curie_temp)
                    z_obs.append(Curie_depth)
                    T_obs_uncer.append(Curie_temp_uncer)
                    z_obs_uncer.append(Curie_depth_uncer)
                
                # Append Moho constraint to the observed list
                if Run_Moho:
                    T_obs.append(Moho_T)
                    z_obs.append(Moho_z) 
                    T_obs_uncer.append(Moho_T_uncer)
                    z_obs_uncer.append(Moho_z_uncer)

                # Append LAB constraint to the observed list
                if Run_LAB:
                    T_obs.append(LAB_temp)
                    z_obs.append(LAB_depth)
                    T_obs_uncer.append(LAB_temp_uncer)
                    z_obs_uncer.append(LAB_depth_uncer)

                T_obs = np.array(T_obs)
                z_obs = np.array(z_obs)
                T_obs_uncer = np.array(T_obs_uncer)
                z_obs_uncer = np.array(z_obs_uncer)

                # Define ranges of values for unknown variables (Based on Moho depth defined here)
                if Run_Moho and (Run_LAB == False):
                    hr_range = np.array([0, Moho_z])                  # e-folding distance (radiogenic heat decay length scale); m
                elif Run_Moho and Run_LAB:
                    hr_range = np.array([0, Moho_z])                  # e-folding distance (radiogenic heat decay length scale); m
                elif (Run_Moho == False) and Run_LAB:
                    hr_range = np.array([0, 30000])                   # e-folding distance (radiogenic heat decay length scale); m
                
                z = np.arange(0, z_obs[-1] + 1000, 1000)   # Depth array based on Moho depth; m

                # Make empty arrays to store results later
                T_sim = np.zeros((Iteration, len(z)))     # For temperature as a function of depth simulation results
                fit_array = np.zeros((Iteration))         # For NRMSE coefficient simulation results
                hr_array = np.zeros((Iteration))          # For e-folding distance results
                k_array = np.zeros((Iteration))           # For thermal conductivity results
                H0_array = np.zeros((Iteration))          # For radiogenic heat production rate at surface
                qm_array = np.zeros((Iteration))          # For mantle heat flux; W/m^2
                qs_array = np.zeros((Iteration))          # For surface heat flux; W/m^2
                Hr_total_array = np.zeros((Iteration))    # For heat flow from heat production; W/m^2
            
                # Run the fitting model now
                # Loop n times depending on the 'Iteration'
                for i in range(Iteration):       
                
                    # Make random thermal properties
                    T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                    k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                    hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                    H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                    if Run_LAB == False:
                        qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2
                        
                    # Set basal condition at LAB depth at 1300 °C
                    elif Run_LAB:
                        # Mantle heat flow; W/m^2
                        qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k) 

                    # Calculate crustal heat production as a function of depth assuming exponential decay
                    Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))

                    qs_model = qm + Hc                                 # Calculate modeled surface heat flow (qs_model)

                    qs_array[i] = qs_model                             # Append modeled surface heat flow

                    qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation

                    # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                    if qs_delta <= 10:
                        # Calculate conductive steady-state temperature as a function of depth
                        T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                        if Uncertainty_box_fitting:
                            # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                            if len(z_obs) == 2:
                                T_obs_sim_base = T[np.where((z >= (z_obs[-1] - z_obs_uncer[-1])))]
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if len(T_obs_sim_base_check[0]) > 0:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break
                            
                            elif len(z_obs) == 3:
                                # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                T_obs_sim_mid = T[np.where((z >= (z_obs[1] - z_obs_uncer[1])) & (z <= (z_obs[1] + z_obs_uncer[1])))]
                                T_obs_sim_base = T[np.where(z >= (z_obs[-1] - z_obs_uncer[-1]))]
                                T_obs_sim_mid_check = np.where((T_obs_sim_mid >= (T_obs[1] - T_obs_uncer[1])) & 
                                                              (T_obs_sim_mid <= (T_obs[1] + T_obs_uncer[1])))
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if (len(T_obs_sim_mid_check[0]) > 0) and (len(T_obs_sim_base_check[0]) > 0):
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                            elif len(z_obs) == 4:
                                # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                T_obs_sim_mid_1 = T[np.where((z >= (z_obs[1] - z_obs_uncer[1])) & (z <= (z_obs[1] + z_obs_uncer[1])))]
                                T_obs_sim_mid_2 = T[np.where((z >= (z_obs[2] - z_obs_uncer[2])) & (z <= (z_obs[2] + z_obs_uncer[2])))]
                                T_obs_sim_base = T[np.where(z >= (z_obs[-1] - z_obs_uncer[-1]))]
                                T_obs_sim_mid_1_check = np.where((T_obs_sim_mid_1 >= (T_obs[1] - T_obs_uncer[1])) & 
                                                                 (T_obs_sim_mid_1 <= (T_obs[1] + T_obs_uncer[1])))
                                T_obs_sim_mid_2_check = np.where((T_obs_sim_mid_2 >= (T_obs[2] - T_obs_uncer[2])) & 
                                                                 (T_obs_sim_mid_2 <= (T_obs[2] + T_obs_uncer[2])))
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))
                    
                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if (len(T_obs_sim_mid_1_check[0]) > 0) and \
                                   (len(T_obs_sim_mid_2_check[0]) > 0) and \
                                   (len(T_obs_sim_base_check[0]) > 0):
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break
                                        
                            elif len(z_obs) == 5:
                                # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                T_obs_sim_mid_1 = T[np.where((z >= (z_obs[1] - z_obs_uncer[1])) & (z <= (z_obs[1] + z_obs_uncer[1])))]
                                T_obs_sim_mid_2 = T[np.where((z >= (z_obs[2] - z_obs_uncer[2])) & (z <= (z_obs[2] + z_obs_uncer[2])))]
                                T_obs_sim_mid_3 = T[np.where((z >= (z_obs[3] - z_obs_uncer[3])) & (z <= (z_obs[3] + z_obs_uncer[3])))]
                                T_obs_sim_base = T[np.where(z >= (z_obs[-1] - z_obs_uncer[-1]))]
                                T_obs_sim_mid_1_check = np.where((T_obs_sim_mid_1 >= (T_obs[1] - T_obs_uncer[1])) & \
                                                                 (T_obs_sim_mid_1 <= (T_obs[1] + T_obs_uncer[1])))
                                T_obs_sim_mid_2_check = np.where((T_obs_sim_mid_2 >= (T_obs[2] - T_obs_uncer[2])) & \
                                                                 (T_obs_sim_mid_2 <= (T_obs[2] + T_obs_uncer[2])))
                                T_obs_sim_mid_3_check = np.where((T_obs_sim_mid_3 >= (T_obs[3] - T_obs_uncer[3])) & \
                                                                 (T_obs_sim_mid_3 <= (T_obs[3] + T_obs_uncer[3])))
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))
                    
                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if (len(T_obs_sim_mid_1_check[0]) > 0) and \
                                   (len(T_obs_sim_mid_2_check[0]) > 0) and \
                                   (len(T_obs_sim_mid_3_check[0]) > 0) and \
                                   (len(T_obs_sim_base_check[0]) > 0):
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        elif Uncertainty_box_fitting == False:
                            # Make an empty list to find the corresponding temperature at all depth constraints
                            T_obs_sim = np.zeros((len(z_obs)))
                        
                            # Loop through the length of the empty list
                            for ii in range(len(z_obs)):     
                                T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
    
                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                            evaluator = RegressionMetric(T_obs, T_obs_sim)
                            GOF2 = evaluator.normalized_root_mean_square_error()
    
                            # Append the simulation results to empty arrays
                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                            hr_array[i] = hr                                   # Append the e-folding distance
                            T_sim[i,:] = T                                     # Append the temperature simulation 
                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                            k_array[i] = k                                     # Append thermal conductivity
                            qm_array[i] = qm                                   # Append mantle heat flux
                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
    
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
    
                            # ***If the amount of good fit is more user setting, break the loop***
                            if ind_good_len >= Max_goodfit_profile:
                                break

                # If there's simulation result -> then find the good fit results
                if ind_good_len > 0:
            
                    # Find the good fit simulation results
                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                    good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                    good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                    good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                    good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                    good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                    good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                    good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                    good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production

                    # Find the moderate fit simulation results
                    ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))         # Find the index of moderate fit
                    moderate_T = T_sim[ind_mod[0], :]                                        # Find moderate fit thermal profiles

                    # Depends on how many good fit thermal profiles --> export mean good fit; calculate their R2 and crustal linear geotherms
                    # If there's more than 1 good fit thermal profiles
                    # Find the mean good fit temperature at constraint depths
                    T_pred = []
                    for depth in z_obs:
                        T_pred.append(good_T_mean[int(depth / 1000)])
                    np.array(T_pred)
                    
                    SSR = np.sum((T_pred - T_obs) ** 2)                  # Calculate sum squared regression
                    TSS = np.sum(((T_obs - np.mean(T_obs)) ** 2))        # Calculate total sum of squares
                    R_square = 1 - (SSR / TSS)                           # Calculate R2
                    R2[i_y,i_x] = R_square                               # Append R2 to corresponding location

                    # Calculate linear geothermal gradient
                    Base_z_sim = z[-1] / 1000                         # Get Moho depth
                    geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim  # Calculate linear geothermal gradient
                    dTdz[i_y,i_x] = geotherm                          # Append crustal-scale geothermal gradient to corresponding location

                    # Calculate peak geothermal gradient from 0 to 250 °C
                    index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                    T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                    z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                    if len(T_250C) > 1:                                 # If there's result
                        dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz

                    # Get temperature at 20km depth
                    if len(good_T_mean) >= 21:
                        T_20km[i_y][i_x] = good_T_mean[20]
                    else:
                        T_20km[i_y][i_x] = np.nan

                    # Save mean good fit thermal profiles at each bin in a 3d array
                    Crustal_T[i_y][i_x] = good_T_mean

                    # Save D95 mask (1 is with all constraints, 2 is with all constraints except one, 3 is with all constraints except two
                    #                4 is with only one constraint)
                    With_or_withoutD95_mask[i_y][i_x] = 1

                    # Save modeled heat flow results
                    Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                    Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                    Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                    Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000)

                # If output has no result -> run the model without mid-crustal constraint
                elif ind_good_len == 0:
                    if len(z_obs) == 2:
                        # Save result as NaN at each bin
                        Crustal_T[i_y][i_x] = np.array([np.nan])
                        R2[i_y][i_x] = np.nan
                        dTdz[i_y][i_x] = np.nan 
                        dTdz_250_max[i_y][i_x] = np.nan 
                        T_20km[i_y][i_x] = np.nan
                        Qs_model[i_y][i_x] = np.nan
                        Qc_model[i_y][i_x] = np.nan
                        Qm_model[i_y][i_x] = np.nan
                        Qc_Qs[i_y][i_x] = np.nan
           
                        # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                        #                       3 is with all constraints except two
                        #                       4 is with only one constrain
                        With_or_withoutD95_mask[i_y][i_x] = np.nan
                        
                    elif len(z_obs) == 3:
                        # Set new arrays for temperature and depth observations for modeling that reduces 1 constraint
                        index = [0, -1]
                        T_obs_v2 = np.array(T_obs)[index]
                        z_obs_v2 = np.array(z_obs)[index]
                        T_obs_uncer_v2 = np.array(T_obs_uncer)[index]
                        z_obs_uncer_v2 = np.array(z_obs_uncer)[index]

                        # Run the fitting model now
                        # Loop n times depending on the 'Iteration' parameter
                        for i in range(Iteration):       
                    
                            # Make random thermal properties
                            T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                            k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                            hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                            H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                            if Run_LAB == False:
                                qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2
                                
                            # Set basal condition at LAB depth at 1300 °C
                            elif Run_LAB:
                                # Mantle heat flow; W/m^2
                                qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
        
                            # Calculate crustal heat production as a function of depth assuming exponential decay
                            Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))
        
                            qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model)
    
                            qs_array[i] = qs_model                       # Append modeled surface heat flow
    
                            qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
    
                            # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                            if qs_delta <= 10:
                                # Calculate conductive steady-state temperature as a function of depth
                                T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                                if Uncertainty_box_fitting:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_base = T[np.where(z >= (z_obs_v2[-1] - z_obs_uncer_v2[-1]))]
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v2[-1] - T_obs_uncer_v2[-1])) & 
                                                                    (T_obs_sim_base <= (T_obs_v2[-1] + T_obs_uncer_v2[-1])))

                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if len(T_obs_sim_base_check[0]) > 0:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v2)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v2)):     
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                                elif Uncertainty_box_fitting == False:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs_v2)))
                            
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs_v2)):
                                        T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
        
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs_v2  , T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
        
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                # Find the amount of good fit
        
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        # If there's simulation result -> then find the good fit results
                        if ind_good_len > 0:
                
                            # Find the good fit simulation results
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                            good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                            good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                            good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                            good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                            good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                            good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                            good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                            good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
    
                            # Find the moderate fit simulation results
                            ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                            moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
    
                            # Export mean good fit; calculate their R2 and linear geotherms
                            # Calculate R2
                            T_pred = []
                            for depth in z_obs_v2:
                                T_pred.append(good_T_mean[int(depth / 1000)])
                            np.array(T_pred)
                            SSR = np.sum((T_pred - T_obs_v2) ** 2)               # Calculate sum squared regression
                            TSS = np.sum(((T_obs_v2 - np.mean(T_obs_v2)) ** 2))  # Calculate total sum of squares
                            R_square = 1 - (SSR / TSS)                           # Calculate R2
                            R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
    
                            # Calculate linear geothermal gradient
                            Base_z_sim = z[-1] / 1000                          # Get Moho depth
                            geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                            dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
    
                            # Calculate peak geothermal gradient from 0 to 250C
                            index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                            T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                            z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                            if len(T_250C) > 1:                                 # If there's result
                                dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
    
                            # Get temperature at 20km depth
                            if len(good_T_mean) >= 21:
                                T_20km[i_y][i_x] = good_T_mean[20]
                            else:
                                T_20km[i_y][i_x] = np.nan

                            # Save mean good fit thermal profiles at each bin in a 3d array
                            Crustal_T[i_y][i_x] = good_T_mean
    
                            # Save D95 mask (1 is with all constraints, 2 is with all constraints except one, 3 is with all constraints except two
                            #                4 is with only one constraint)
                            With_or_withoutD95_mask[i_y][i_x] = 4
                            
                            # Save modeled heat flow results
                            Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                            Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                            Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                            Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
    
                        # If no result
                        elif ind_good_len == 0:
                            # Save result as NaN at each bin
                            Crustal_T[i_y][i_x] = np.array([np.nan])
                            R2[i_y][i_x] = np.nan
                            dTdz[i_y][i_x] = np.nan 
                            dTdz_250_max[i_y][i_x] = np.nan 
                            T_20km[i_y][i_x] = np.nan
                            Qs_model[i_y][i_x] = np.nan
                            Qc_model[i_y][i_x] = np.nan
                            Qm_model[i_y][i_x] = np.nan
                            Qc_Qs[i_y][i_x] = np.nan
    
                            # Save D95 mask (1 is with all constraints, 2 is with all constraints except one, 3 is with all constraints except two
                            #                4 is with only one constraint)
                            With_or_withoutD95_mask[i_y][i_x] = np.nan

                    elif len(z_obs) == 4:
                        # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                        index = [0, 2, -1]
                        T_obs_v2 = np.array(T_obs)[index]
                        z_obs_v2 = np.array(z_obs)[index]
                        T_obs_uncer_v2 = np.array(T_obs_uncer)[index]
                        z_obs_uncer_v2 = np.array(z_obs_uncer)[index]

                        # Run the fitting model now
                        # Loop n times depending on the 'Iteration' parameter
                        for i in range(Iteration):       
                    
                            # Make random thermal properties
                            T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                            k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                            hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                            H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                            if Run_LAB == False:
                                qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2
                                
                            # Set basal condition at LAB depth at 1300 °C
                            elif Run_LAB:
                                # Mantle heat flow; W/m^2
                                qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
        
                            # Calculate crustal heat production as a function of depth assuming exponential decay
                            Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))

                            qs_model = qm + Hc                                 # Calculate modeled surface heat flow (qs_model) 
    
                            qs_array[i] = qs_model                             # Append modeled surface heat flow
    
                            qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
    
                            # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                            if qs_delta <= 10:
                                # Calculate conductive steady-state temperature as a function of depth
                                T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                                if Uncertainty_box_fitting:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_mid = T[np.where(z >= (z_obs_v2[1] - z_obs_uncer_v2[1]))]
                                    T_obs_sim_base = T[np.where(z >= (z_obs_v2[-1] - z_obs_uncer_v2[-1]))]
                                    T_obs_sim_mid_check = np.where((T_obs_sim_mid >= (T_obs_v2[1] - T_obs_uncer_v2[1])) & 
                                                                    (T_obs_sim_mid <= (T_obs_v2[1] + T_obs_uncer_v2[1])))
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v2[-1] - T_obs_uncer_v2[-1])) & 
                                                                    (T_obs_sim_base <= (T_obs_v2[-1] + T_obs_uncer_v2[-1])))

                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if (len(T_obs_sim_mid_check[0]) > 0) and len(T_obs_sim_base_check[0]) > 0:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v2)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v2)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                                elif Uncertainty_box_fitting == False:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs_v2)))
                            
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs_v2)):
                                        T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
        
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
        
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                # Find the amount of good fit
        
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        # If there's simulation result -> then find the good fit results
                        if ind_good_len > 0:
                
                            # Find the good fit simulation results
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                            good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                            good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                            good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                            good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                            good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                            good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                            good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                            good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
    
                            # Find the moderate fit simulation results
                            ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                            moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
    
                            # Export mean good fit; calculate their R2 and linear geotherms
                            # Calculate R2
                            T_pred = []
                            for depth in z_obs_v2:
                                T_pred.append(good_T_mean[int(depth / 1000)])
                            np.array(T_pred)
                            SSR = np.sum((T_pred - T_obs_v2) ** 2)               # Calculate sum squared regression
                            TSS = np.sum(((T_obs_v2 - np.mean(T_obs_v2)) ** 2))  # Calculate total sum of squares
                            R_square = 1 - (SSR / TSS)                           # Calculate R2
                            R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
    
                            # Calculate linear geothermal gradient
                            Base_z_sim = z[-1] / 1000                          # Get Moho depth
                            geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                            dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
    
                            # Calculate peak geothermal gradient from 0 to 250C
                            index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                            T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                            z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                            if len(T_250C) > 1:                                 # If there's result
                                dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
    
                            # Get temperature at 20km depth
                            if len(good_T_mean) >= 21:
                                T_20km[i_y][i_x] = good_T_mean[20]
                            else:
                                T_20km[i_y][i_x] = np.nan

                            # Save mean good fit thermal profiles at each bin in a 3d array
                            Crustal_T[i_y][i_x] = good_T_mean
    
                                            
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = 3

                            # Save modeled heat flow results
                            Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                            Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                            Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                            Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
    
                        # If no result
                        elif ind_good_len == 0:
                            # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                            index = [0, -1]
                            T_obs_v3 = np.array(T_obs_v2)[index]
                            z_obs_v3 = np.array(z_obs_v2)[index]
                            T_obs_uncer_v3 = np.array(T_obs_uncer_v2)[index]
                            z_obs_uncer_v3 = np.array(z_obs_uncer_v2)[index]

                            # Run the fitting model now
                            # Loop n times depending on the 'Iteration' parameter
                            for i in range(Iteration):       
                        
                                # Make random thermal properties
                                T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                                k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                                hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                                H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                                if Run_LAB == False:
                                    qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                                # Set basal condition at LAB depth at 1300 °C
                                elif Run_LAB:
                                    # Mantle heat flow; W/m^2
                                    qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
            
                                # Calculate crustal heat production as a function of depth assuming exponential decay
                                Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))

                                qs_model = qm + Hc                                 # Calculate modeled surface heat flow (qs_model) 
        
                                qs_array[i] = qs_model                             # Append modeled surface heat flow
        
                                qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
        
                                # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                                if qs_delta <= 10:
                                    # Calculate conductive steady-state temperature as a function of depth
                                    T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 
    
                                    if Uncertainty_box_fitting:
                                        # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                        T_obs_sim_base = T[np.where(z >= (z_obs_v3[-1] - z_obs_uncer_v3[-1]))]
                                        T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v3[-1] - T_obs_uncer_v3[-1])) & 
                                                                        (T_obs_sim_base <= (T_obs_v3[-1] + T_obs_uncer_v3[-1])))
    
                                        # If the modeled geotherm is within the temperature-depth uncertainty box
                                        if len(T_obs_sim_base_check[0]) > 0:
                                            # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                            T_obs_sim = np.zeros((len(z_obs_v3)))
                                        
                                            # Loop through the length of the empty list
                                            for ii in range(len(z_obs_v3)):
                                                T_obs_sim[ii] = T[np.where(z == z_obs_v3[ii])] # Find the temperature from the simulation at the observed depth
                    
                                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                            evaluator = RegressionMetric(T_obs_v3, T_obs_sim)
                                            GOF2 = evaluator.normalized_root_mean_square_error()
                    
                                            # Append the simulation results to empty arrays
                                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                            hr_array[i] = hr                                   # Append the e-folding distance
                                            T_sim[i,:] = T                                     # Append the temperature simulation 
                                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                                            k_array[i] = k                                     # Append thermal conductivity
                                            qm_array[i] = qm                                   # Append mantle heat flux
                                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
                    
                                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                    
                                            # ***If the amount of good fit is more user setting, break the loop***
                                            if ind_good_len >= Max_goodfit_profile:
                                                break
    
                                    elif Uncertainty_box_fitting == False:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v3)))
                                
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v3)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v3[ii])]   # Find the temperature from the simulation at the observed depth
            
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v3, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
            
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                # Find the amount of good fit
            
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break
    
                            # If there's simulation result -> then find the good fit results
                            if ind_good_len > 0:
                    
                                # Find the good fit simulation results
                                ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                                good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                                good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                                good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                                good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                                good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                                good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                                good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                                good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
        
                                # Find the moderate fit simulation results
                                ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                                moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
        
                                # Export mean good fit; calculate their R2 and linear geotherms
                                # Calculate R2
                                T_pred = []
                                for depth in z_obs_v3:
                                    T_pred.append(good_T_mean[int(depth / 1000)])
                                np.array(T_pred)
                                SSR = np.sum((T_pred - T_obs_v3) ** 2)               # Calculate sum squared regression
                                TSS = np.sum(((T_obs_v3 - np.mean(T_obs_v3)) ** 2))  # Calculate total sum of squares
                                R_square = 1 - (SSR / TSS)                           # Calculate R2
                                R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
        
                                # Calculate linear geothermal gradient
                                Base_z_sim = z[-1] / 1000                          # Get Moho depth
                                geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                                dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
        
                                # Calculate peak geothermal gradient from 0 to 250C
                                index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                                T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                                z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                                if len(T_250C) > 1:                                 # If there's result
                                    dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
        
                                # Get temperature at 20km depth
                                if len(good_T_mean) >= 21:
                                    T_20km[i_y][i_x] = good_T_mean[20]
                                else:
                                    T_20km[i_y][i_x] = np.nan

                                # Save mean good fit thermal profiles at each bin in a 3d array
                                Crustal_T[i_y][i_x] = good_T_mean
        
                                            
                                # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                                #                       3 is with all constraints except two
                                #                       4 is with only one constrain
                                With_or_withoutD95_mask[i_y][i_x] = 4
    
                                # Save modeled heat flow results
                                Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                                Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                                Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                                Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000)
        
                            # If no result
                            elif ind_good_len == 0:
                                # Save result as NaN at each bin
                                Crustal_T[i_y][i_x] = np.array([np.nan])
                                R2[i_y][i_x] = np.nan
                                dTdz[i_y][i_x] = np.nan
                                dTdz_250_max[i_y][i_x] = np.nan
                                T_20km[i_y][i_x] = np.nan
                                Qs_model[i_y][i_x] = np.nan
                                Qc_model[i_y][i_x] = np.nan
                                Qm_model[i_y][i_x] = np.nan
                                Qc_Qs[i_y][i_x] = np.nan
        
                                # Save constraint mask (1 is with all constraints, 2 is with all constraints except one, 3 is with all constraints except two
                                #                       4 is with only one constrain
                                With_or_withoutD95_mask[i_y][i_x] = np.nan
                    
                    elif len(z_obs) == 5:
                        # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                        index = [0, 2, 3, 4]
                        T_obs_v2 = np.array(T_obs)[index]
                        z_obs_v2 = np.array(z_obs)[index]
                        T_obs_uncer_v2 = np.array(T_obs_uncer)[index]
                        z_obs_uncer_v2 = np.array(z_obs_uncer)[index]

                        # Run the fitting model now
                        # Loop n times depending on the 'Iteration' parameter
                        for i in range(Iteration):    
                    
                            # Make random thermal properties
                            T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                            k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                            hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                            H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                            if Run_LAB == False:
                                qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                            # Set basal condition at LAB depth at 1300 °C
                            elif Run_LAB:
                                # Mantle heat flow; W/m^2
                                qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
        
                            # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                            Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))      # Calculate crustal heat production as a function of depth assuming exponential decay
        
                            qs_model = qm + Hc                            # Calculate modeled surface heat flow (qs_model)
    
                            qs_array[i] = qs_model                        # Append modeled surface heat flow
    
                            qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
    
                            # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                            if qs_delta <= 10:
                                # Calculate conductive steady-state temperature as a function of depth
                                T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr))))

                                if Uncertainty_box_fitting:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_mid_1 = T[np.where((z >= (z_obs_v2[1] - z_obs_uncer_v2[1])) &
                                                                (z >= (z_obs_v2[1] + z_obs_uncer_v2[1])))]
                                    T_obs_sim_mid_2 = T[np.where((z >= (z_obs_v2[2] - z_obs_uncer_v2[2])) &
                                                                (z >= (z_obs_v2[2] & z_obs_uncer_v2[2])))]
                                    T_obs_sim_base = T[np.where(z >= (z_obs_v2[-1] - z_obs_uncer_v2[-1]))]
                                    T_obs_sim_mid_1_check = np.where((T_obs_sim_mid >= (T_obs_v2[1] - T_obs_uncer_v2[1])) &
                                                                    (T_obs_sim_mid <= (T_obs_v2[1] + T_obs_uncer_v2[1])))
                                    T_obs_sim_mid_2_check = np.where((T_obs_sim_mid >= (T_obs_v2[2] - T_obs_uncer_v2[2])) &
                                                                    (T_obs_sim_mid <= (T_obs_v2[2] + T_obs_uncer_v2[2])))
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v2[-1] - T_obs_uncer_v2[-1])) &
                                                                    (T_obs_sim_base <= (T_obs_v2[-1] + T_obs_uncer_v2[-1])))

                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if (len(T_obs_sim_mid_1_check[0]) > 0) and \
                                       (len(T_obs_sim_mid_2_check[0]) > 0) and \
                                       (len(T_obs_sim_base_check[0]) > 0):
                                        
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v2)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v2)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                                elif Uncertainty_box_fitting == False:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs_v2)))
                            
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs_v2)):
                                        T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
        
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
        
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                # Find the amount of good fit
        
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        # If there's simulation result -> then find the good fit results
                        if ind_good_len > 0:
                
                            # Find the good fit simulation results
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))
                            good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                            good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                            good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                            good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                            good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                            good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                            good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                            good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
    
                            # Find the moderate fit simulation results
                            ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                            moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
    
                            # Export mean good fit; calculate their R2 and linear geotherms
                            # Calculate R2
                            T_pred = []
                            for depth in z_obs_v2:
                                T_pred.append(good_T_mean[int(depth / 1000)])
                            np.array(T_pred)
                            SSR = np.sum((T_pred - T_obs_v2) ** 2)               # Calculate sum squared regression
                            TSS = np.sum(((T_obs_v2 - np.mean(T_obs_v2)) ** 2))  # Calculate total sum of squares
                            R_square = 1 - (SSR / TSS)                           # Calculate R2
                            R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
    
                            # Calculate linear geothermal gradient
                            Base_z_sim = z[-1] / 1000                          # Get Moho depth
                            geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                            dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
    
                            # Calculate peak geothermal gradient from 0 to 250C
                            index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                            T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                            z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                            if len(T_250C) > 1:                                 # If there's result
                                dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
    
                            # Get temperature at 20km depth
                            if len(good_T_mean) >= 21:
                                T_20km[i_y][i_x] = good_T_mean[20]
                            else:
                                T_20km[i_y][i_x] = np.nan

                            # Save mean good fit thermal profiles at each bin in a 3d array
                            Crustal_T[i_y][i_x] = good_T_mean
    
                                            
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = 2

                            # Save modeled heat flow results
                            Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                            Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                            Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                            Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model)) * 1000

                        
                        # If no result
                        elif ind_good_len == 0:
                            # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                            index = [0, 2, -1]
                            T_obs_v3 = np.array(T_obs_v2)[index]
                            z_obs_v3 = np.array(z_obs_v2)[index]
                            T_obs_uncer_v3 = np.array(T_obs_uncer_v2)[index]
                            z_obs_uncer_v3 = np.array(z_obs_uncer_v2)[index]

                            # Run the fitting model now
                            # Loop n times depending on the 'Iteration' parameter
                            for i in range(Iteration):       
                        
                                # Make random thermal properties
                                T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                                k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                                hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                                H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                                if Run_LAB == False:
                                    qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                                # Set basal condition at LAB depth at 1300 °C
                                elif Run_LAB:
                                    # Mantle heat flow; W/m^2
                                    qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
            
                                # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                                Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))      # Calculate crustal heat production as a function of depth assuming exponential decay
            
                                qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model) 
        
                                qs_array[i] = qs_model                       # Append modeled surface heat flow
        
                                qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
        
                                # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                                if qs_delta <= 10:
                                    # Calculate conductive steady-state temperature as a function of depth
                                    T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr))))
    
                                    if Uncertainty_box_fitting:
                                        # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                        T_obs_sim_base = T[np.where(z >= (z_obs_v3[-1] - z_obs_uncer_v3[-1]))]
                                        T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v3[-1] - T_obs_uncer_v3[-1])) &
                                                                        (T_obs_sim_base <= (T_obs_v3[-1] + T_obs_uncer_v3[-1])))
    
                                        # If the modeled geotherm is within the temperature-depth uncertainty box
                                        if len(T_obs_sim_base_check[0]) > 0:
                                            # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                            T_obs_sim = np.zeros((len(z_obs_v3)))
                                        
                                            # Loop through the length of the empty list
                                            for ii in range(len(z_obs_v3)):
                                                T_obs_sim[ii] = T[np.where(z == z_obs_v3[ii])]   # Find the temperature from the simulation at the observed depth
                    
                                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                            evaluator = RegressionMetric(T_obs_v3, T_obs_sim)
                                            GOF2 = evaluator.normalized_root_mean_square_error()
                    
                                            # Append the simulation results to empty arrays
                                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                            hr_array[i] = hr                                   # Append the e-folding distance
                                            T_sim[i,:] = T                                     # Append the temperature simulation 
                                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                                            k_array[i] = k                                     # Append thermal conductivity
                                            qm_array[i] = qm                                   # Append mantle heat flux
                                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
                    
                                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                    
                                            # ***If the amount of good fit is more user setting, break the loop***
                                            if ind_good_len >= Max_goodfit_profile:
                                                break
    
                                    elif Uncertainty_box_fitting == False:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v3)))
                                
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v3)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v3[ii])]   # Find the temperature from the simulation at the observed depth
            
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v3, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
            
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                # Find the amount of good fit
            
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break
    
                            # If there's simulation result -> then find the good fit results
                            if ind_good_len > 0:
                    
                                # Find the good fit simulation results
                                ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                                good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                                good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                                good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                                good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                                good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                                good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                                good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                                good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
        
                                # Find the moderate fit simulation results
                                ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                                moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
        
                                # Export mean good fit; calculate their R2 and linear geotherms
                                # Calculate R2
                                T_pred = []
                                for depth in z_obs_v3:
                                    T_pred.append(good_T_mean[int(depth / 1000)])
                                np.array(T_pred)
                                SSR = np.sum((T_pred - T_obs_v3) ** 2)               # Calculate sum squared regression
                                TSS = np.sum(((T_obs_v3 - np.mean(T_obs_v3)) ** 2))  # Calculate total sum of squares
                                R_square = 1 - (SSR / TSS)                           # Calculate R2
                                R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
        
                                # Calculate linear geothermal gradient
                                Base_z_sim = z[-1] / 1000                          # Get Moho depth
                                geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                                dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
        
                                # Calculate peak geothermal gradient from 0 to 250C
                                index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                                T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                                z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                                if len(T_250C) > 1:                                 # If there's result
                                    dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
        
                                # Get temperature at 20km depth
                                if len(good_T_mean) >= 21:
                                    T_20km[i_y][i_x] = good_T_mean[20]
                                else:
                                    T_20km[i_y][i_x] = np.nan

                                # Save mean good fit thermal profiles at each bin in a 3d array
                                Crustal_T[i_y][i_x] = good_T_mean
        
                                            
                                # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                                #                       3 is with all constraints except two
                                #                       4 is with only one constrain
                                With_or_withoutD95_mask[i_y][i_x] = 3
    
                                # Save modeled heat flow results
                                Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                                Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                                Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                                Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
        
                            # If no result
                            elif ind_good_len == 0:
                                # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                                index = [0, -1]
                                T_obs_v4 = np.array(T_obs_v3)[index]
                                z_obs_v4 = np.array(z_obs_v3)[index]
                                T_obs_uncer_v4 = np.array(T_obs_uncer_v3)[index]
                                z_obs_uncer_v4 = np.array(z_obs_uncer_v3)[index]
                                
                                # Run the fitting model now
                                # Loop n times depending on the 'Iteration' parameter
                                for i in range(Iteration):     
                            
                                    # Make random thermal properties
                                    T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                                    k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                                    hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                                    H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                                    if Run_LAB == False:
                                        qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                                    # Set basal condition at LAB depth at 1300 °C
                                    elif Run_LAB:
                                        # Mantle heat flow; W/m^2
                                        qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
                
                                    # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                                    # Calculate crustal heat production as a function of depth, assuming exponential decay
                                    Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))
                                    
                                    qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model) 
            
                                    qs_array[i] = qs_model                       # Append modeled surface heat flow
            
                                    qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
            
                                    # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                                    if qs_delta <= 10:
                                        # Calculate conductive steady-state temperature as a function of depth
                                        T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 
        
                                        if Uncertainty_box_fitting:
                                            # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                            T_obs_sim_base = T[np.where(z >= (z_obs_v4[-1] - z_obs_uncer_v4[-1]))]
                                            T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v4[-1] - T_obs_uncer_v4[-1])) & 
                                                                            (T_obs_sim_base <= (T_obs_v4[-1] + T_obs_uncer_v4[-1])))
        
                                            # If the modeled geotherm is within the temperature-depth uncertainty box
                                            if len(T_obs_sim_base_check[0]) > 0:
                                                # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                                T_obs_sim = np.zeros((len(z_obs_v4)))
                                            
                                                # Loop through the length of the empty list
                                                for ii in range(len(z_obs_v4)):
                                                    # Find the temperature from the simulation at the observed depth
                                                    T_obs_sim[ii] = T[np.where(z == z_obs_v4[ii])]
                        
                                                # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                                evaluator = RegressionMetric(T_obs_v4, T_obs_sim)
                                                GOF2 = evaluator.normalized_root_mean_square_error()
                        
                                                # Append the simulation results to empty arrays
                                                fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                                hr_array[i] = hr                                   # Append the e-folding distance
                                                T_sim[i,:] = T                                     # Append the temperature simulation 
                                                H0_array[i] = H0                                   # Append radiogenic heat production rate
                                                k_array[i] = k                                     # Append thermal conductivity
                                                qm_array[i] = qm                                   # Append mantle heat flux
                                                Hr_total_array[i] = Hc                             # Append heat flow from heat production
                        
                                                ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                                ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                        
                                                # ***If the amount of good fit is more user setting, break the loop***
                                                if ind_good_len >= Max_goodfit_profile:
                                                    break
        
                                        elif Uncertainty_box_fitting == False:
                                            # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                            T_obs_sim = np.zeros((len(z_obs_v4)))
                                    
                                            # Loop through the length of the empty list
                                            for ii in range(len(z_obs_v4)):
                                                T_obs_sim[ii] = T[np.where(z == z_obs_v4[ii])] # Find the temperature from the simulation at the observed depth
                
                                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                            evaluator = RegressionMetric(T_obs_v4, T_obs_sim)
                                            GOF2 = evaluator.normalized_root_mean_square_error()
                
                                            # Append the simulation results to empty arrays
                                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                            hr_array[i] = hr                                   # Append the e-folding distance
                                            T_sim[i,:] = T                                     # Append the temperature simulation 
                                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                                            k_array[i] = k                                     # Append thermal conductivity
                                            qm_array[i] = qm                                   # Append mantle heat flux
                                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                            ind_good_len = len(ind_good[0])                                # Find the amount of good fit
                
                                            # ***If the amount of good fit is more user setting, break the loop***
                                            if ind_good_len >= Max_goodfit_profile:
                                                break
        
                                # If there's simulation result -> then find the good fit results
                                if ind_good_len > 0:
                        
                                    # Find the good fit simulation results
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                                    good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                                    good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                                    good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                                    good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                                    good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                                    good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                                    good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                                    good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
            
                                    # Find the moderate fit simulation results
                                    ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                                    moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
            
                                    # Export mean good fit; calculate their R2 and linear geotherms
                                    # Calculate R2
                                    T_pred = []
                                    for depth in z_obs_v4:
                                        T_pred.append(good_T_mean[int(depth / 1000)])
                                    np.array(T_pred)
                                    SSR = np.sum((T_pred - T_obs_v4) ** 2)               # Calculate sum squared regression
                                    TSS = np.sum(((T_obs_v4 - np.mean(T_obs_v4)) ** 2))  # Calculate total sum of squares
                                    R_square = 1 - (SSR / TSS)                           # Calculate R2
                                    R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
            
                                    # Calculate linear geothermal gradient
                                    Base_z_sim = z[-1] / 1000                          # Get Moho depth
                                    geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                                    dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
            
                                    # Calculate peak geothermal gradient from 0 to 250C
                                    index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                                    T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                                    z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                                    if len(T_250C) > 1:                                 # If there's result
                                        dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
            
                                    # Get temperature at 20km depth
                                    if len(good_T_mean) >= 21:
                                        T_20km[i_y][i_x] = good_T_mean[20]
                                    else:
                                        T_20km[i_y][i_x] = np.nan
    
                                    # Save mean good fit thermal profiles at each bin in a 3d array
                                    Crustal_T[i_y][i_x] = good_T_mean
            
                                            
                                    # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                                    #                       3 is with all constraints except two
                                    #                       4 is with only one constrain
                                    With_or_withoutD95_mask[i_y][i_x] = 4
        
                                    # Save modeled heat flow results
                                    Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                                    Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                                    Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                                    Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 

                                # If no result
                                elif ind_good_len == 0:
                                    # Save result as NaN at each bin
                                    Crustal_T[i_y][i_x] = np.array([np.nan])
                                    R2[i_y][i_x] = np.nan
                                    dTdz[i_y][i_x] = np.nan 
                                    dTdz_250_max[i_y][i_x] = np.nan 
                                    T_20km[i_y][i_x] = np.nan
                                    Qs_model[i_y][i_x] = np.nan
                                    Qc_model[i_y][i_x] = np.nan
                                    Qm_model[i_y][i_x] = np.nan
                                    Qc_Qs[i_y][i_x] = np.nan
        
                                            
                                    # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                                    #                       3 is with all constraints except two
                                    #                       4 is with only one constrain
                                    With_or_withoutD95_mask[i_y][i_x] = np.nan

            # When the basal boundary condition is at LAB and some data is NaN, it uses all constraints except the data = NaN
            elif Run_LAB and ('Qs' in condition_indices_v2) and ('LAB_depth' in condition_indices_v2) and \
                 (('D95' in condition_indices_v2) or ('Curie_depth' in condition_indices_v2) or \
                 (('Moho_z' in condition_indices_v2) and ('Moho_T' in condition_indices_v2))):
                
                # Set two arrays for temperature and depth observations based on active datasets
                T_obs = [10]  # Surface temperature = 10 °C
                z_obs = [0]   # Surface depth = 0 m
                T_obs_uncer = [0]
                z_obs_uncer = [0]

                # Append D95 depth and temperature, and their uncertainties to the observed list
                if 'D95' in condition_indices_v2:
                    T_obs.append(D95_temp)
                    z_obs.append(D95_depth)
                    T_obs_uncer.append(D95_temp_uncer)
                    z_obs_uncer.append(D95_depth_uncer)

                # Append Curie depth and temperature, and their uncertainties to the observed list
                if 'Curie_depth' in condition_indices_v2:
                    T_obs.append(Curie_temp)
                    z_obs.append(Curie_depth)
                    T_obs_uncer.append(Curie_temp_uncer)
                    z_obs_uncer.append(Curie_depth_uncer)
                
                # Append Moho constraint to the observed list
                if ('Moho_z' in condition_indices_v2) and ('Moho_T' in condition_indices_v2):
                    T_obs.append(Moho_T)
                    z_obs.append(Moho_z) 
                    T_obs_uncer.append(Moho_T_uncer)
                    z_obs_uncer.append(Moho_z_uncer)

                # Append LAB constraint to the observed list
                T_obs.append(LAB_temp)
                z_obs.append(LAB_depth)
                T_obs_uncer.append(LAB_temp_uncer)
                z_obs_uncer.append(LAB_depth_uncer)

                T_obs = np.array(T_obs)
                z_obs = np.array(z_obs)
                T_obs_uncer = np.array(T_obs_uncer)
                z_obs_uncer = np.array(z_obs_uncer)

                # Define ranges of values for unknown variables (Based on Moho depth defined here)
                if Run_Moho and Run_LAB:
                    hr_range = np.array([0, Moho_z])       # e-folding distance (radiogenic heat decay length scale); m
                elif (Run_Moho == False) and Run_LAB:
                    hr_range = np.array([0, 30000])        # e-folding distance (radiogenic heat decay length scale); m
                elif Run_Moho and (Run_LAB == False):
                    hr_range = np.array([0, Moho_z])       # e-folding distance (radiogenic heat decay length scale); m
                
                z = np.arange(0, z_obs[-1] + 1000, 1000)   # Depth array based on Moho depth; m

                # Make empty arrays to store results later
                T_sim = np.zeros((Iteration, len(z)))     # For temperature as a function of depth simulation results
                fit_array = np.zeros((Iteration))         # For NRMSE coefficient simulation results
                hr_array = np.zeros((Iteration))          # For e-folding distance results
                k_array = np.zeros((Iteration))           # For thermal conductivity results
                H0_array = np.zeros((Iteration))          # For radiogenic heat production rate at surface
                qm_array = np.zeros((Iteration))          # For mantle heat flux; W/m^2
                qs_array = np.zeros((Iteration))          # For surface heat flux; W/m^2
                Hr_total_array = np.zeros((Iteration))    # For heat flow from heat production; W/m^2
            
                # Run the fitting model now
                # Loop n times depending on the 'Iteration'
                for i in range(Iteration):       
                
                    # Make random thermal properties
                    T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                    k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                    hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                    H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                    if Run_LAB == False:
                        qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                    # Set basal condition at LAB depth at 1300 °C
                    elif Run_LAB:
                        # Mantle heat flow; W/m^2
                        qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)

                    # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                    Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))      # Calculate crustal heat production as a function of depth assuming exponential decay

                    qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model)

                    qs_array[i] = qs_model                       # Append modeled surface heat flow

                    qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation

                    # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                    if qs_delta <= 10:
                        # Calculate conductive steady-state temperature as a function of depth
                        T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr))))

                        if Uncertainty_box_fitting:
                            # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                            if len(z_obs) == 2:
                                T_obs_sim_base = T[np.where((z >= (z_obs[-1] - z_obs_uncer[-1])))]
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if len(T_obs_sim_base_check[0]) > 0:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break
                            
                            elif len(z_obs) == 3:
                                # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                T_obs_sim_mid = T[np.where((z >= (z_obs[1] - z_obs_uncer[1])) & (z <= (z_obs[1] + z_obs_uncer[1])))]
                                T_obs_sim_base = T[np.where(z >= (z_obs[-1] - z_obs_uncer[-1]))]
                                T_obs_sim_mid_check = np.where((T_obs_sim_mid >= (T_obs[1] - T_obs_uncer[1])) & 
                                                              (T_obs_sim_mid <= (T_obs[1] + T_obs_uncer[1])))
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if (len(T_obs_sim_mid_check[0]) > 0) and (len(T_obs_sim_base_check[0]) > 0):
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth

                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                            elif len(z_obs) == 4:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_mid_1 = T[np.where((z >= (z_obs[1] - z_obs_uncer[1])) & (z <= (z_obs[1] + z_obs_uncer[1])))]
                                    T_obs_sim_mid_2 = T[np.where((z >= (z_obs[2] - z_obs_uncer[2])) & (z <= (z_obs[2] + z_obs_uncer[2])))]
                                    T_obs_sim_base = T[np.where(z >= (z_obs[-1] - z_obs_uncer[-1]))]
                                    T_obs_sim_mid_1_check = np.where((T_obs_sim_mid_1 >= (T_obs[1] - T_obs_uncer[1])) & 
                                                                     (T_obs_sim_mid_1 <= (T_obs[1] + T_obs_uncer[1])))
                                    T_obs_sim_mid_2_check = np.where((T_obs_sim_mid_2 >= (T_obs[2] - T_obs_uncer[2])) & 
                                                                     (T_obs_sim_mid_2 <= (T_obs[2] + T_obs_uncer[2])))
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                                   (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))
                        
                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if (len(T_obs_sim_mid_1_check[0]) > 0) and \
                                       (len(T_obs_sim_mid_2_check[0]) > 0) and (len(T_obs_sim_base_check[0]) > 0):
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs)):     
                                            T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                        elif Uncertainty_box_fitting == False:
                            # Make an empty list to find the corresponding temperature at all depth constraints
                            T_obs_sim = np.zeros((len(z_obs)))

                            # Loop through the length of the empty list
                            for ii in range(len(z_obs)):     
                                T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
    
                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                            evaluator = RegressionMetric(T_obs, T_obs_sim)
                            GOF2 = evaluator.normalized_root_mean_square_error()
    
                            # Append the simulation results to empty arrays
                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                            hr_array[i] = hr                                   # Append the e-folding distance
                            T_sim[i,:] = T                                     # Append the temperature simulation 
                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                            k_array[i] = k                                     # Append thermal conductivity
                            qm_array[i] = qm                                   # Append mantle heat flux
                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
    
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
    
                            # ***If the amount of good fit is more user setting, break the loop***
                            if ind_good_len >= Max_goodfit_profile:
                                break

                # If there's simulation result -> then find the good fit results
                if ind_good_len > 0:
            
                    # Find the good fit simulation results
                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                    good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                    good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                    good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                    good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                    good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                    good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                    good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                    good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production

                    # Find the moderate fit simulation results
                    ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))         # Find the index of moderate fit
                    moderate_T = T_sim[ind_mod[0], :]                                        # Find moderate fit thermal profiles

                    # Depends on how many good fit thermal profiles --> export mean good fit; calculate their R2 and crustal linear geotherms
                    # If there's more than 1 good fit thermal profiles
                    # Find the mean good fit temperature at constraint depths
                    T_pred = []
                    for depth in z_obs:
                        T_pred.append(good_T_mean[int(depth / 1000)])
                    np.array(T_pred)
                    
                    SSR = np.sum((T_pred - T_obs) ** 2)                  # Calculate sum squared regression
                    TSS = np.sum(((T_obs - np.mean(T_obs)) ** 2))        # Calculate total sum of squares
                    R_square = 1 - (SSR / TSS)                           # Calculate R2
                    R2[i_y,i_x] = R_square                               # Append R2 to corresponding location

                    # Calculate linear geothermal gradient
                    Base_z_sim = z[-1] / 1000                         # Get Moho depth
                    geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim  # Calculate linear geothermal gradient
                    dTdz[i_y,i_x] = geotherm                          # Append crustal-scale geothermal gradient to corresponding location

                    # Calculate peak geothermal gradient from 0 to 250 °C
                    index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                    T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                    z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                    if len(T_250C) > 1:                                 # If there's result
                        dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz

                    # Get temperature at 20km depth
                    if len(good_T_mean) >= 21:
                        T_20km[i_y][i_x] = good_T_mean[20]
                    else:
                        T_20km[i_y][i_x] = np.nan

                    # Save mean good fit thermal profiles at each bin in a 3d array
                    Crustal_T[i_y][i_x] = good_T_mean

                    # Save constraint mask (1 is with all constraints, 2 is with all constraints except one, 3 is with all constraints except two
                    #                       4 is with only one constrain
                    if len(z_obs) == 4:
                        With_or_withoutD95_mask[i_y][i_x] = 2
                    elif len(z_obs) == 3:
                        With_or_withoutD95_mask[i_y][i_x] = 3
                    elif len(z_obs) == 2:
                        With_or_withoutD95_mask[i_y][i_x] = 4

                    # Save modeled heat flow results
                    Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                    Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                    Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                    Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000)

                # If output has no result -> run the model without mid-crustal constraint
                elif ind_good_len == 0:
                    if len(z_obs) == 2:
                        # Save result as NaN at each bin
                        Crustal_T[i_y][i_x] = np.array([np.nan])
                        R2[i_y][i_x] = np.nan
                        dTdz[i_y][i_x] = np.nan 
                        dTdz_250_max[i_y][i_x] = np.nan 
                        T_20km[i_y][i_x] = np.nan
                        Qs_model[i_y][i_x] = np.nan
                        Qc_model[i_y][i_x] = np.nan
                        Qm_model[i_y][i_x] = np.nan
                        Qc_Qs[i_y][i_x] = np.nan
           
                        # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                        #                       3 is with all constraints except two
                        #                       4 is with only one constrain
                        With_or_withoutD95_mask[i_y][i_x] = np.nan
                        
                    elif len(z_obs) == 3:
                        # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                        index = [0, -1]
                        T_obs_v2 = np.array(T_obs)[index]
                        z_obs_v2 = np.array(z_obs)[index]
                        T_obs_uncer_v2 = np.array(T_obs_uncer)[index]
                        z_obs_uncer_v2 = np.array(z_obs_uncer)[index]

                        # Run the fitting model now
                        # Loop n times depending on the 'Iteration' parameter
                        for i in range(Iteration):       
                    
                            # Make random thermal properties
                            T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                            k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                            hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                            H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                            if Run_LAB == False:
                                qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                            # Set basal condition at LAB depth at 1300 °C
                            elif Run_LAB:
                                # Mantle heat flow; W/m^2
                                qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
        
                            # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                            Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))      # Calculate crustal heat production as a function of depth assuming exponential decay
        
                            qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model) 
    
                            qs_array[i] = qs_model                             # Append modeled surface heat flow
    
                            qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
    
                            # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                            if qs_delta <= 10:
                                # Calculate conductive steady-state temperature as a function of depth
                                T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                                if Uncertainty_box_fitting:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_base = T[np.where(z >= (z_obs_v2[-1] - z_obs_uncer_v2[-1]))]
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v2[-1] - T_obs_uncer_v2[-1])) & 
                                                                    (T_obs_sim_base <= (T_obs_v2[-1] + T_obs_uncer_v2[-1])))

                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if len(T_obs_sim_base_check[0]) > 0:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v2)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v2)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                                elif Uncertainty_box_fitting == False:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs_v2)))
                            
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs_v2)):
                                        T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
        
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
        
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                # Find the amount of good fit
        
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        # If there's simulation result -> then find the good fit results
                        if ind_good_len > 0:
                
                            # Find the good fit simulation results
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                            good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                            good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                            good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                            good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                            good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                            good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                            good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                            good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
    
                            # Find the moderate fit simulation results
                            ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                            moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
    
                            # Export mean good fit; calculate their R2 and linear geotherms
                            # Calculate R2
                            T_pred = []
                            for depth in z_obs_v2:
                                T_pred.append(good_T_mean[int(depth / 1000)])
                            np.array(T_pred)
                            SSR = np.sum((T_pred - T_obs_v2) ** 2)                  # Calculate sum squared regression
                            TSS = np.sum(((T_obs_v2 - np.mean(T_obs_v2)) ** 2))        # Calculate total sum of squares
                            R_square = 1 - (SSR / TSS)                           # Calculate R2
                            R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
    
                            # Calculate linear geothermal gradient
                            Base_z_sim = z[-1] / 1000                          # Get Moho depth
                            geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                            dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
    
                            # Calculate peak geothermal gradient from 0 to 250C
                            index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                            T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                            z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                            if len(T_250C) > 1:                                 # If there's result
                                dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
    
                            # Get temperature at 20km depth
                            if len(good_T_mean) >= 21:
                                T_20km[i_y][i_x] = good_T_mean[20]
                            else:
                                T_20km[i_y][i_x] = np.nan

                            # Save mean good fit thermal profiles at each bin in a 3d array
                            Crustal_T[i_y][i_x] = good_T_mean
          
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = 4

                            # Save modeled heat flow results
                            Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                            Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                            Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                            Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
    
                        # If no result
                        elif ind_good_len == 0:
                            # Save result as NaN at each bin
                            Crustal_T[i_y][i_x] = np.array([np.nan])
                            R2[i_y][i_x] = np.nan
                            dTdz[i_y][i_x] = np.nan 
                            dTdz_250_max[i_y][i_x] = np.nan 
                            T_20km[i_y][i_x] = np.nan
                            Qs_model[i_y][i_x] = np.nan
                            Qc_model[i_y][i_x] = np.nan
                            Qm_model[i_y][i_x] = np.nan
                            Qc_Qs[i_y][i_x] = np.nan
               
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = np.nan

                    elif len(z_obs) == 4:
                        # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                        index = [0, 2, -1]
                        T_obs_v2 = np.array(T_obs)[index]
                        z_obs_v2 = np.array(z_obs)[index]
                        T_obs_uncer_v2 = np.array(T_obs_uncer)[index]
                        z_obs_uncer_v2 = np.array(z_obs_uncer)[index]

                        # Run the fitting model now
                        # Loop n times depending on the 'Iteration' parameter
                        for i in range(Iteration):       
                    
                            # Make random thermal properties
                            T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                            k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                            hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                            H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                            if Run_LAB == False:
                                qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                            # Set basal condition at LAB depth at 1300 °C
                            elif Run_LAB:
                                # Mantle heat flow; W/m^2
                                qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
        
                            # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                            Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))      # Calculate crustal heat production as a function of depth assuming exponential decay
        
                            qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model) 
    
                            qs_array[i] = qs_model                             # Append modeled surface heat flow
    
                            qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
    
                            # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                            if qs_delta <= 10:
                                # Calculate conductive steady-state temperature as a function of depth
                                T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                                if Uncertainty_box_fitting:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_mid = T[np.where(z >= (z_obs_v2[1] - z_obs_uncer_v2[1]))]
                                    T_obs_sim_base = T[np.where(z >= (z_obs_v2[-1] - z_obs_uncer_v2[-1]))]
                                    T_obs_sim_mid_check = np.where((T_obs_sim_mid >= (T_obs_v2[1] - T_obs_uncer_v2[1])) & 
                                                                    (T_obs_sim_mid <= (T_obs_v2[1] + T_obs_uncer_v2[1])))
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v2[-1] - T_obs_uncer_v2[-1])) & 
                                                                    (T_obs_sim_base <= (T_obs_v2[-1] + T_obs_uncer_v2[-1])))

                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if (len(T_obs_sim_mid_check[0]) > 0) and len(T_obs_sim_base_check[0]) > 0:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v2)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v2)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                                elif Uncertainty_box_fitting == False:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs_v2)))
                            
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs_v2)):
                                        T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
        
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
        
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                # Find the amount of good fit
        
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        # If there's simulation result -> then find the good fit results
                        if ind_good_len > 0:
                
                            # Find the good fit simulation results
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                            good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                            good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                            good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                            good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                            good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                            good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                            good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                            good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
    
                            # Find the moderate fit simulation results
                            ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                            moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
    
                            # Export mean good fit; calculate their R2 and linear geotherms
                            # Calculate R2
                            T_pred = []
                            for depth in z_obs_v2:
                                T_pred.append(good_T_mean[int(depth / 1000)])
                            np.array(T_pred)
                            SSR = np.sum((T_pred - T_obs_v2) ** 2)                  # Calculate sum squared regression
                            TSS = np.sum(((T_obs_v2 - np.mean(T_obs_v2)) ** 2))        # Calculate total sum of squares
                            R_square = 1 - (SSR / TSS)                           # Calculate R2
                            R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
    
                            # Calculate linear geothermal gradient
                            Base_z_sim = z[-1] / 1000                          # Get Moho depth
                            geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                            dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
    
                            # Calculate peak geothermal gradient from 0 to 250C
                            index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                            T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                            z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                            if len(T_250C) > 1:                                 # If there's result
                                dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
    
                            # Get temperature at 20km depth
                            if len(good_T_mean) >= 21:
                                T_20km[i_y][i_x] = good_T_mean[20]
                            else:
                                T_20km[i_y][i_x] = np.nan

                            # Save mean good fit thermal profiles at each bin in a 3d array
                            Crustal_T[i_y][i_x] = good_T_mean
    
                                            
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = 3

                            # Save modeled heat flow results
                            Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                            Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                            Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                            Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
    
                        # If no result
                        elif ind_good_len == 0:
                            # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                            index = [0, -1]
                            T_obs_v3 = np.array(T_obs_v2)[index]
                            z_obs_v3 = np.array(z_obs_v2)[index]
                            T_obs_uncer_v3 = np.array(T_obs_uncer_v2)[index]
                            z_obs_uncer_v3 = np.array(z_obs_uncer_v2)[index]

                            # Run the fitting model now
                            # Loop n times depending on the 'Iteration' parameter
                            for i in range(Iteration):       
                        
                                # Make random thermal properties
                                T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                                k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                                hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                                H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                                if Run_LAB == False:
                                    qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                                # Set basal condition at LAB depth at 1300 °C
                                elif Run_LAB:
                                    # Mantle heat flow; W/m^2
                                    qm = (LAB_temp - T0 - ((H0 * (hr ** 2)) / k) * ((1 - np.exp(-LAB_depth/hr)) ** 2)) / (LAB_depth / k)
            
                                # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                                Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))  # Calculate crustal heat production as a function of depth assuming exponential decay
            
                                qs_model = qm + Hc                        # Calculate modeled surface heat flow (qs_model) 
        
                                qs_array[i] = qs_model                             # Append modeled surface heat flow
        
                                qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
        
                                # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                                if qs_delta <= 10:
                                    # Calculate conductive steady-state temperature as a function of depth
                                    T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 
    
                                    if Uncertainty_box_fitting:
                                        # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                        T_obs_sim_base = T[np.where(z >= (z_obs_v3[-1] - z_obs_uncer_v3[-1]))]
                                        T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v3[-1] - T_obs_uncer_v3[-1])) & 
                                                                        (T_obs_sim_base <= (T_obs_v3[-1] + T_obs_uncer_v3[-1])))
    
                                        # If the modeled geotherm is within the temperature-depth uncertainty box
                                        if len(T_obs_sim_base_check[0]) > 0:
                                            # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                            T_obs_sim = np.zeros((len(z_obs_v3)))
                                        
                                            # Loop through the length of the empty list
                                            for ii in range(len(z_obs_v3)):
                                                T_obs_sim[ii] = T[np.where(z == z_obs_v3[ii])] # Find the temperature from the simulation at the observed depth
                    
                                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                            evaluator = RegressionMetric(T_obs_v3, T_obs_sim)
                                            GOF2 = evaluator.normalized_root_mean_square_error()
                    
                                            # Append the simulation results to empty arrays
                                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                            hr_array[i] = hr                                   # Append the e-folding distance
                                            T_sim[i,:] = T                                     # Append the temperature simulation 
                                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                                            k_array[i] = k                                     # Append thermal conductivity
                                            qm_array[i] = qm                                   # Append mantle heat flux
                                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
                    
                                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                    
                                            # ***If the amount of good fit is more user setting, break the loop***
                                            if ind_good_len >= Max_goodfit_profile:
                                                break
    
                                    elif Uncertainty_box_fitting == False:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v3)))
                                
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v3)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v3[ii])]   # Find the temperature from the simulation at the observed depth
            
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v3, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
            
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                # Find the amount of good fit
            
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break
    
                            # If there's simulation result -> then find the good fit results
                            if ind_good_len > 0:
                    
                                # Find the good fit simulation results
                                ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                                good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                                good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                                good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                                good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                                good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                                good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                                good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                                good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
        
                                # Find the moderate fit simulation results
                                ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                                moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
        
                                # Export mean good fit; calculate their R2 and linear geotherms
                                # Calculate R2
                                T_pred = []
                                for depth in z_obs_v3:
                                    T_pred.append(good_T_mean[int(depth / 1000)])
                                np.array(T_pred)
                                SSR = np.sum((T_pred - T_obs_v3) ** 2)               # Calculate sum squared regression
                                TSS = np.sum(((T_obs_v3 - np.mean(T_obs_v3)) ** 2))  # Calculate total sum of squares
                                R_square = 1 - (SSR / TSS)                           # Calculate R2
                                R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
        
                                # Calculate linear geothermal gradient
                                Base_z_sim = z[-1] / 1000                          # Get Moho depth
                                geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                                dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
        
                                # Calculate peak geothermal gradient from 0 to 250C
                                index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                                T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                                z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                                if len(T_250C) > 1:                                 # If there's result
                                    dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
        
                                # Get temperature at 20km depth
                                if len(good_T_mean) >= 21:
                                    T_20km[i_y][i_x] = good_T_mean[20]
                                else:
                                    T_20km[i_y][i_x] = np.nan

                                # Save mean good fit thermal profiles at each bin in a 3d array
                                Crustal_T[i_y][i_x] = good_T_mean
        
                                            
                                # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                                #                       3 is with all constraints except two
                                #                       4 is with only one constrain
                                With_or_withoutD95_mask[i_y][i_x] = 4
    
                                # Save modeled heat flow results
                                Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                                Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                                Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                                Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
        
                            # If no result
                            elif ind_good_len == 0:
                                # Save result as NaN at each bin
                                Crustal_T[i_y][i_x] = np.array([np.nan])
                                R2[i_y][i_x] = np.nan
                                dTdz[i_y][i_x] = np.nan 
                                dTdz_250_max[i_y][i_x] = np.nan 
                                T_20km[i_y][i_x] = np.nan
                                Qs_model[i_y][i_x] = np.nan
                                Qc_model[i_y][i_x] = np.nan
                                Qm_model[i_y][i_x] = np.nan
                                Qc_Qs[i_y][i_x] = np.nan
        
                                # Save constraint mask (1 is with all constraints, 2 is with all constraints except one, 3 is with all constraints except two
                                #                       4 is with only one constrain
                                With_or_withoutD95_mask[i_y][i_x] = np.nan
            
            # When basal boundary condition is at Moho
            elif (Run_LAB == False) and (Run_Moho) and ('Qs' in condition_indices_v2) and \
                 ('Moho_z' in condition_indices_v2) and ('Moho_T' in condition_indices_v2) and \
                 (('D95' in condition_indices_v2) or ('Curie_depth' in condition_indices_v2)):
                
                # Set two arrays for temperature and depth observations based on active datasets
                T_obs = [10]  # Surface temperature = 10 °C
                z_obs = [0]   # Surface depth = 0 m
                T_obs_uncer = [0]
                z_obs_uncer = [0]

                # Append D95 depth and temperature, and their uncertainties to the observed list
                if 'D95' in condition_indices_v2:
                    T_obs.append(D95_temp)
                    z_obs.append(D95_depth)
                    T_obs_uncer.append(D95_temp_uncer)
                    z_obs_uncer.append(D95_depth_uncer)

                # Append Curie depth and temperature, and their uncertainties to the observed list
                if 'Curie_depth' in condition_indices_v2:
                    T_obs.append(Curie_temp)
                    z_obs.append(Curie_depth)
                    T_obs_uncer.append(Curie_temp_uncer)
                    z_obs_uncer.append(Curie_depth_uncer)
                
                # Append Moho constraint to the observed list
                T_obs.append(Moho_T)
                z_obs.append(Moho_z) 
                T_obs_uncer.append(Moho_T_uncer)
                z_obs_uncer.append(Moho_z_uncer)

                T_obs = np.array(T_obs)
                z_obs = np.array(z_obs)
                T_obs_uncer = np.array(T_obs_uncer)
                z_obs_uncer = np.array(z_obs_uncer)

                # Define ranges of values for unknown variables (Based on Moho depth defined here)
                hr_range = np.array([0, Moho_z])                  # e-folding distance (radiogenic heat decay length scale); m
                    
                z = np.arange(0, z_obs[-1] + 1000, 1000)   # Depth array based on Moho depth; m

                # Make empty arrays to store results later
                T_sim = np.zeros((Iteration, len(z)))     # For temperature as a function of depth simulation results
                fit_array = np.zeros((Iteration))         # For NRMSE coefficient simulation results
                hr_array = np.zeros((Iteration))          # For e-folding distance results
                k_array = np.zeros((Iteration))           # For thermal conductivity results
                H0_array = np.zeros((Iteration))          # For radiogenic heat production rate at surface
                qm_array = np.zeros((Iteration))      # For mantle heat flux; W/m^2
                qs_array = np.zeros((Iteration))          # For surface heat flux; W/m^2
                Hr_total_array = np.zeros((Iteration))    # For heat flow from heat production; W/m^2

                # Run the fitting model now
                # Loop n times depending on the 'Iteration'
                for i in range(Iteration):       
                
                    # Make random thermal properties
                    T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                    k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                    hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                    H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                    qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                    # Calculate crustal heat production as a function of depth assuming exponential decay
                    Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))

                    qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model)

                    qs_array[i] = qs_model                       # Append modeled surface heat flow

                    qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation

                    # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                    if qs_delta <= 10:
                        # Calculate conductive steady-state temperature as a function of depth
                        T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                        if Uncertainty_box_fitting:
                            # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                            if len(z_obs) == 2:
                                T_obs_sim_base = T[np.where((z >= (z_obs[-1] - z_obs_uncer[-1])))]
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if len(T_obs_sim_base_check[0]) > 0:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break
                            
                            elif len(z_obs) == 3:
                                # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                T_obs_sim_mid = T[np.where((z >= (z_obs[1] - z_obs_uncer[1])) & (z <= (z_obs[1] + z_obs_uncer[1])))]
                                T_obs_sim_base = T[np.where(z >= (z_obs[-1] - z_obs_uncer[-1]))]
                                T_obs_sim_mid_check = np.where((T_obs_sim_mid >= (T_obs[1] - T_obs_uncer[1])) & 
                                                              (T_obs_sim_mid <= (T_obs[1] + T_obs_uncer[1])))
                                T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                               (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                                # If the modeled geotherm is within the temperature-depth uncertainty box
                                if (len(T_obs_sim_mid_check[0]) > 0) and (len(T_obs_sim_base_check[0]) > 0):
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs)))
                                
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs)):     
                                        T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
            
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
            
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
            
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
            
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        elif Uncertainty_box_fitting == False:
                            # Make an empty list to find the corresponding temperature at all depth constraints
                            T_obs_sim = np.zeros((len(z_obs)))
                        
                            # Loop through the length of the empty list
                            for ii in range(len(z_obs)):     
                                T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
    
                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                            evaluator = RegressionMetric(T_obs, T_obs_sim)
                            GOF2 = evaluator.normalized_root_mean_square_error()
    
                            # Append the simulation results to empty arrays
                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                            hr_array[i] = hr                                   # Append the e-folding distance
                            T_sim[i,:] = T                                     # Append the temperature simulation 
                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                            k_array[i] = k                                     # Append thermal conductivity
                            qm_array[i] = qm                                   # Append mantle heat flux
                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
    
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
    
                            # ***If the amount of good fit is more user setting, break the loop***
                            if ind_good_len >= Max_goodfit_profile:
                                break

                # If there's simulation result -> then find the good fit results
                if ind_good_len > 0:
            
                    # Find the good fit simulation results
                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                    good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                    good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                    good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                    good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                    good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                    good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                    good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                    good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production

                    # Find the moderate fit simulation results
                    ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))         # Find the index of moderate fit
                    moderate_T = T_sim[ind_mod[0], :]                                        # Find moderate fit thermal profiles

                    # Depends on how many good fit thermal profiles --> export mean good fit; calculate their R2 and crustal linear geotherms
                    # If there's more than 1 good fit thermal profiles
                    # Find the mean good fit temperature at constraint depths
                    T_pred = []
                    for depth in z_obs:
                        T_pred.append(good_T_mean[int(depth / 1000)])
                    np.array(T_pred)
                    
                    SSR = np.sum((T_pred - T_obs) ** 2)                  # Calculate sum squared regression
                    TSS = np.sum(((T_obs - np.mean(T_obs)) ** 2))        # Calculate total sum of squares
                    R_square = 1 - (SSR / TSS)                           # Calculate R2
                    R2[i_y,i_x] = R_square                               # Append R2 to corresponding location

                    # Calculate linear geothermal gradient
                    Base_z_sim = z[-1] / 1000                         # Get Moho depth
                    geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim  # Calculate linear geothermal gradient
                    dTdz[i_y,i_x] = geotherm                          # Append crustal-scale geothermal gradient to corresponding location

                    # Calculate peak geothermal gradient from 0 to 250 °C
                    index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                    T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                    z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                    if len(T_250C) > 1:                                 # If there's result
                        dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz

                    # Get temperature at 20km depth
                    if len(good_T_mean) >= 21:
                        T_20km[i_y][i_x] = good_T_mean[20]
                    else:
                        T_20km[i_y][i_x] = np.nan

                    # Save mean good fit thermal profiles at each bin in a 3d array
                    Crustal_T[i_y][i_x] = good_T_mean
                                            
                    # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                    #                       3 is with all constraints except two
                    #                       4 is with only one constrain
                    With_or_withoutD95_mask[i_y][i_x] = 1

                    # Save modeled heat flow results
                    Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                    Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                    Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                    Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000)

                # If output has no result -> run the model without mid-crustal constraint
                elif ind_good_len == 0:
                    if len(z_obs) == 2:
                        # Save result as NaN at each bin
                        Crustal_T[i_y][i_x] = np.array([np.nan])
                        R2[i_y][i_x] = np.nan
                        dTdz[i_y][i_x] = np.nan 
                        dTdz_250_max[i_y][i_x] = np.nan 
                        T_20km[i_y][i_x] = np.nan
                        Qs_model[i_y][i_x] = np.nan
                        Qc_model[i_y][i_x] = np.nan
                        Qm_model[i_y][i_x] = np.nan
                        Qc_Qs[i_y][i_x] = np.nan
           
                        # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                        #                       3 is with all constraints except two
                        #                       4 is with only one constrain
                        With_or_withoutD95_mask[i_y][i_x] = np.nan

                    elif len(z_obs) == 3:
                        # Set new arrays for temperature and depth observations for modeling that deduces 1 constraint
                        index = [0, -1]
                        T_obs_v2 = np.array(T_obs)[index]
                        z_obs_v2 = np.array(z_obs)[index]
                        T_obs_uncer_v2 = np.array(T_obs_uncer)[index]
                        z_obs_uncer_v2 = np.array(z_obs_uncer)[index]

                        # Run the fitting model now
                        # Loop n times depending on the 'Iteration' parameter
                        for i in range(Iteration):       

                            # Make random thermal properties
                            T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                            k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                            hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                            H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                            qm = np.random.uniform(qm_range[0], qm_range[1], 1)  # Mantle heat flow; W/m^2

                            # Calculate Geothermal gradient; Turcotte and Schubert, 2014
                            Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))     # Calculate crustal heat production as a function of depth assuming exponential decay

                            qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model) 

                            qs_array[i] = qs_model                       # Append modeled surface heat flow
    
                            qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation
    
                            # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                            if qs_delta <= 10:
                                # Calculate conductive steady-state temperature as a function of depth
                                T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                                if Uncertainty_box_fitting:
                                    # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                                    T_obs_sim_base = T[np.where(z >= (z_obs_v2[-1] - z_obs_uncer_v2[-1]))]
                                    T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs_v2[-1] - T_obs_uncer_v2[-1])) & 
                                                                    (T_obs_sim_base <= (T_obs_v2[-1] + T_obs_uncer_v2[-1])))

                                    # If the modeled geotherm is within the temperature-depth uncertainty box
                                    if len(T_obs_sim_base_check[0]) > 0:
                                        # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                        T_obs_sim = np.zeros((len(z_obs_v2)))
                                    
                                        # Loop through the length of the empty list
                                        for ii in range(len(z_obs_v2)):
                                            T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
                
                                        # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                        evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                        GOF2 = evaluator.normalized_root_mean_square_error()
                
                                        # Append the simulation results to empty arrays
                                        fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                        hr_array[i] = hr                                   # Append the e-folding distance
                                        T_sim[i,:] = T                                     # Append the temperature simulation 
                                        H0_array[i] = H0                                   # Append radiogenic heat production rate
                                        k_array[i] = k                                     # Append thermal conductivity
                                        qm_array[i] = qm                                   # Append mantle heat flux
                                        Hr_total_array[i] = Hc                             # Append heat flow from heat production
                
                                        ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                        ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
                
                                        # ***If the amount of good fit is more user setting, break the loop***
                                        if ind_good_len >= Max_goodfit_profile:
                                            break

                                elif Uncertainty_box_fitting == False:
                                    # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                    T_obs_sim = np.zeros((len(z_obs_v2)))
                            
                                    # Loop through the length of the empty list
                                    for ii in range(len(z_obs_v2)):
                                        T_obs_sim[ii] = T[np.where(z == z_obs_v2[ii])]   # Find the temperature from the simulation at the observed depth
        
                                    # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                    evaluator = RegressionMetric(T_obs_v2, T_obs_sim)
                                    GOF2 = evaluator.normalized_root_mean_square_error()
        
                                    # Append the simulation results to empty arrays
                                    fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                    hr_array[i] = hr                                   # Append the e-folding distance
                                    T_sim[i,:] = T                                     # Append the temperature simulation 
                                    H0_array[i] = H0                                   # Append radiogenic heat production rate
                                    k_array[i] = k                                     # Append thermal conductivity
                                    qm_array[i] = qm                                   # Append mantle heat flux
                                    Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  # Find the index of good fit
                                    ind_good_len = len(ind_good[0])                                # Find the amount of good fit
        
                                    # ***If the amount of good fit is more user setting, break the loop***
                                    if ind_good_len >= Max_goodfit_profile:
                                        break

                        # If there's simulation result -> then find the good fit results
                        if ind_good_len > 0:
                
                            # Find the good fit simulation results
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                            good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                            good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                            good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                            good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                            good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                            good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                            good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                            good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production
    
                            # Find the moderate fit simulation results
                            ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))     # Find the index of moderate fit
                            moderate_T = T_sim[ind_mod[0], :]                                    # Find moderate fit thermal profiles
    
                            # Export mean good fit; calculate their R2 and linear geotherms
                            # Calculate R2
                            T_pred = []
                            for depth in z_obs_v2:
                                T_pred.append(good_T_mean[int(depth / 1000)])
                            np.array(T_pred)
                            SSR = np.sum((T_pred - T_obs) ** 2)                  # Calculate sum squared regression
                            TSS = np.sum(((T_obs - np.mean(T_obs)) ** 2))        # Calculate total sum of squares
                            R_square = 1 - (SSR / TSS)                           # Calculate R2
                            R2[i_y][i_x] = R_square                              # Append R2 to corresponding location
    
                            # Calculate linear geothermal gradient
                            Base_z_sim = z[-1] / 1000                          # Get Moho depth
                            geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim   # Calculate linear geothermal gradient
                            dTdz[i_y][i_x] = geotherm                          # Append geothermal gradient to corresponding location
    
                            # Calculate peak geothermal gradient from 0 to 250C
                            index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                            T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                            z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                            if len(T_250C) > 1:                                 # If there's result
                                dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz
    
                            # Get temperature at 20km depth
                            if len(good_T_mean) >= 21:
                                T_20km[i_y][i_x] = good_T_mean[20]
                            else:
                                T_20km[i_y][i_x] = np.nan

                            # Save mean good fit thermal profiles at each bin in a 3d array
                            Crustal_T[i_y][i_x] = good_T_mean
               
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = 4

                            # Save modeled heat flow results
                            Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                            Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                            Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                            Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000) 
    
                        # If no result
                        elif ind_good_len == 0:
                            # Save result as NaN at each bin
                            Crustal_T[i_y][i_x] = np.array([np.nan])
                            R2[i_y][i_x] = np.nan
                            dTdz[i_y][i_x] = np.nan 
                            dTdz_250_max[i_y][i_x] = np.nan 
                            T_20km[i_y][i_x] = np.nan
                            Qs_model[i_y][i_x] = np.nan
                            Qc_model[i_y][i_x] = np.nan
                            Qm_model[i_y][i_x] = np.nan
                            Qc_Qs[i_y][i_x] = np.nan
               
                            # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                            #                       3 is with all constraints except two
                            #                       4 is with only one constrain
                            With_or_withoutD95_mask[i_y][i_x] = np.nan

            # When basal boundary condition is at Moho
            elif (Run_LAB == False) and (Run_Moho) and ('Qs' in condition_indices_v2) and \
                 ('Moho_z' in condition_indices_v2) and ('Moho_T' in condition_indices_v2):
                
                # Set two arrays for temperature and depth observations based on active datasets
                T_obs = [10]  # Surface temperature = 10 °C
                z_obs = [0]   # Surface depth = 0 m
                T_obs_uncer = [0]
                z_obs_uncer = [0]
                
                # Append Moho constraint to the observed list
                T_obs.append(Moho_T)
                z_obs.append(Moho_z) 
                T_obs_uncer.append(Moho_T_uncer)
                z_obs_uncer.append(Moho_z_uncer)

                T_obs = np.array(T_obs)
                z_obs = np.array(z_obs)
                T_obs_uncer = np.array(T_obs_uncer)
                z_obs_uncer = np.array(z_obs_uncer)

                # Define ranges of values for unknown variables (Based on Moho depth defined here)
                hr_range = np.array([0, Moho_z])           # e-folding distance (radiogenic heat decay length scale); m
                    
                z = np.arange(0, z_obs[-1] + 1000, 1000)   # Depth array based on Moho depth; m

                # Make empty arrays to store results later
                T_sim = np.zeros((Iteration, len(z)))     # For temperature as a function of depth simulation results
                fit_array = np.zeros((Iteration))         # For NRMSE coefficient simulation results
                hr_array = np.zeros((Iteration))          # For e-folding distance results
                k_array = np.zeros((Iteration))           # For thermal conductivity results
                H0_array = np.zeros((Iteration))          # For radiogenic heat production rate at surface
                qm_array = np.zeros((Iteration))          # For mantle heat flux; W/m^2
                qs_array = np.zeros((Iteration))          # For surface heat flux; W/m^2
                Hr_total_array = np.zeros((Iteration))    # For heat flow from heat production; W/m^2

                # Run the fitting model now
                # Loop n times depending on the 'Iteration'
                for i in range(Iteration):       
                
                    # Make random thermal properties
                    T0 = np.random.uniform(T0_range[0], T0_range[1], 1)      # Surface temperature; °C  
                    k = np.random.uniform(k_range[0], k_range[1], 1)         # Thermal conductivity; W/m/°C
                    hr = np.random.uniform(hr_range[0], hr_range[1], 1)      # e-folding distance (radiogenic heat decay length scale); m
                    H0 = np.random.uniform(H0_range[0], H0_range[1], 1)      # Radiogenic heat production rate at surface; W/m^3
                    qm = np.random.uniform(qm_range[0], qm_range[1], 1)      # Mantle heat flow; W/m^2

                    # Calculate crustal heat production as a function of depth assuming exponential decay
                    Hc = H0 * hr * (1 - np.exp(-z[-1] / hr))

                    qs_model = qm + Hc                           # Calculate modeled surface heat flow (qs_model)

                    qs_array[i] = qs_model                       # Append modeled surface heat flow

                    qs_delta = np.abs((qs_model * 1000) - (Qs * 1000)) # Calculate the heat flow difference between model and observation

                    # If the heat flow difference is less or equal to 10 mW m-2, then go forward to do the calculation
                    if qs_delta <= 10:
                        # Calculate conductive steady-state temperature as a function of depth
                        T = T0 + ((qm * z) / (k)) + ((((qs_model - qm) * hr) / (k)) * (1 - np.exp(- (z / hr)))) 

                        if Uncertainty_box_fitting:
                            # Make sure the modeled geotherms if within the uncertainty temperature-depth box of constraint data
                            T_obs_sim_base = T[np.where((z >= (z_obs[-1] - z_obs_uncer[-1])))]
                            T_obs_sim_base_check = np.where((T_obs_sim_base >= (T_obs[-1] - T_obs_uncer[-1])) & \
                                                           (T_obs_sim_base <= (T_obs[-1] + T_obs_uncer[-1])))

                            # If the modeled geotherm is within the temperature-depth uncertainty box
                            if len(T_obs_sim_base_check[0]) > 0:
                                # Make an empty list to find the corresponding temperature at the surface, D95, and Moho depth
                                T_obs_sim = np.zeros((len(z_obs)))
                            
                                # Loop through the length of the empty list
                                for ii in range(len(z_obs)):     
                                    T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
        
                                # Get the goodness of fit; Compare between observed temperature and simulation temperature
                                evaluator = RegressionMetric(T_obs, T_obs_sim)
                                GOF2 = evaluator.normalized_root_mean_square_error()
        
                                # Append the simulation results to empty arrays
                                fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                                hr_array[i] = hr                                   # Append the e-folding distance
                                T_sim[i,:] = T                                     # Append the temperature simulation 
                                H0_array[i] = H0                                   # Append radiogenic heat production rate
                                k_array[i] = k                                     # Append thermal conductivity
                                qm_array[i] = qm                                   # Append mantle heat flux
                                Hr_total_array[i] = Hc                             # Append heat flow from heat production
        
                                ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                                ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
        
                                # ***If the amount of good fit is more user setting, break the loop***
                                if ind_good_len >= Max_goodfit_profile:
                                    break

                        elif Uncertainty_box_fitting == False:
                            # Make an empty list to find the corresponding temperature at all depth constraints
                            T_obs_sim = np.zeros((len(z_obs)))

                            # Loop through the length of the empty list
                            for ii in range(len(z_obs)):     
                                T_obs_sim[ii] = T[np.where(z == z_obs[ii])]   # Find the temperature from the simulation at the observed depth
    
                            # Get the goodness of fit; Compare between observed temperature and simulation temperature
                            evaluator = RegressionMetric(T_obs, T_obs_sim)
                            GOF2 = evaluator.normalized_root_mean_square_error()
    
                            # Append the simulation results to empty arrays
                            fit_array[i] = GOF2                                # Append the goodness of fit of each iteration
                            hr_array[i] = hr                                   # Append the e-folding distance
                            T_sim[i,:] = T                                     # Append the temperature simulation 
                            H0_array[i] = H0                                   # Append radiogenic heat production rate
                            k_array[i] = k                                     # Append thermal conductivity
                            qm_array[i] = qm                                   # Append mantle heat flux
                            Hr_total_array[i] = Hc                             # Append heat flow from heat production
    
                            ind_good = np.where((fit_array < goodfit) & (fit_array != 0))     # Find the index of good fit
                            ind_good_len = len(ind_good[0])                                   # Find the amount of good fit
    
                            # ***If the amount of good fit is more user setting, break the loop***
                            if ind_good_len >= Max_goodfit_profile:
                                break

                # If there's simulation result -> then find the good fit results
                if ind_good_len > 0:
            
                    # Find the good fit simulation results
                    ind_good = np.where((fit_array < goodfit) & (fit_array != 0))  
                    good_T = T_sim[ind_good[0], :]                   # Find good fit thermal profiles
                    good_T_mean = np.mean(good_T, axis = 0)          # Get the mean of all good fit thermal profiles
                    good_hr = hr_array[ind_good[0]]                  # Find good-fit e-folding distance
                    good_k = k_array[ind_good[0]]                    # Find good-fit thermal conductivity
                    good_qm = qm_array[ind_good[0]]                  # Find good-fit mantle heat flux
                    good_H0 = H0_array[ind_good[0]]                  # Find good-fit radiogenic heat production rate
                    good_qs_model = qs_array[ind_good[0]]            # Find good-fit modeled surface heat flow
                    good_Hr_total = Hr_total_array[ind_good[0]]      # Find good-fit heat flow from heat production

                    # Find the moderate fit simulation results
                    ind_mod = np.where((fit_array < moderatefit) & (fit_array != 0))         # Find the index of moderate fit
                    moderate_T = T_sim[ind_mod[0], :]                                        # Find moderate fit thermal profiles

                    # Depends on how many good fit thermal profiles --> export mean good fit; calculate their R2 and crustal linear geotherms
                    # If there's more than 1 good fit thermal profiles
                    # Find the mean good fit temperature at constraint depths
                    T_pred = []
                    for depth in z_obs:
                        T_pred.append(good_T_mean[int(depth / 1000)])
                    np.array(T_pred)
                    
                    SSR = np.sum((T_pred - T_obs) ** 2)                  # Calculate sum squared regression
                    TSS = np.sum(((T_obs - np.mean(T_obs)) ** 2))        # Calculate total sum of squares
                    R_square = 1 - (SSR / TSS)                           # Calculate R2
                    R2[i_y,i_x] = R_square                               # Append R2 to corresponding location

                    # Calculate linear geothermal gradient
                    Base_z_sim = z[-1] / 1000                         # Get Moho depth
                    geotherm = (T_pred[-1] - T_pred[0]) / Base_z_sim  # Calculate linear geothermal gradient
                    dTdz[i_y,i_x] = geotherm                          # Append crustal-scale geothermal gradient to corresponding location

                    # Calculate peak geothermal gradient from 0 to 250 °C
                    index = np.argmin(np.abs(good_T_mean - 250))        # Find the index position closest to 250 °C
                    T_250C = good_T_mean[0:index]                       # Get the temperature from surface to 250 °C
                    z_250C = np.arange(0, index, 1)                     # Get the depth at 250 °C
                    if len(T_250C) > 1:                                 # If there's result
                        dTdz_250_max[i_y][i_x] = np.max(np.gradient(T_250C, z_250C))     # Calculate the maximum dTdz

                    # Get temperature at 20km depth
                    if len(good_T_mean) >= 21:
                        T_20km[i_y][i_x] = good_T_mean[20]
                    else:
                        T_20km[i_y][i_x] = np.nan

                    # Save mean good fit thermal profiles at each bin in a 3d array
                    Crustal_T[i_y][i_x] = good_T_mean
                                            
                    # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                    #                       3 is with all constraints except two
                    #                       4 is with only one constrain
                    With_or_withoutD95_mask[i_y][i_x] = 1

                    # Save modeled heat flow results
                    Qs_model[i_y][i_x] = np.mean(good_qs_model) * 1000
                    Qc_model[i_y][i_x] = np.mean(good_Hr_total) * 1000
                    Qm_model[i_y][i_x] = np.mean(good_qm) * 1000
                    Qc_Qs[i_y][i_x] = (np.mean(good_Hr_total) * 1000) / (np.mean(good_qs_model) * 1000)

                # If output has no result -> run the model without mid-crustal constraint
                elif ind_good_len == 0:
                    # Save result as NaN at each bin
                    Crustal_T[i_y][i_x] = np.array([np.nan])
                    R2[i_y][i_x] = np.nan
                    dTdz[i_y][i_x] = np.nan 
                    dTdz_250_max[i_y][i_x] = np.nan 
                    T_20km[i_y][i_x] = np.nan
                    Qs_model[i_y][i_x] = np.nan
                    Qc_model[i_y][i_x] = np.nan
                    Qm_model[i_y][i_x] = np.nan
                    Qc_Qs[i_y][i_x] = np.nan
       
                    # Save constraint mask (1 is with all constraints, 2 is with all constraints except one,
                    #                       3 is with all constraints except two
                    #                       4 is with only one constrain
                    With_or_withoutD95_mask[i_y][i_x] = np.nan

    # Replace 0 with NaN
    R2 = np.where(R2 == 0, np.nan, R2)
    dTdz = np.where(dTdz == 0, np.nan, dTdz)
    dTdz_250_max = np.where(dTdz_250_max == 0, np.nan, dTdz_250_max)
    T_20km = np.where(T_20km == 0, np.nan, T_20km)
    Qs_model = np.where(Qs_model == 0, np.nan, Qs_model)
    Qc_model = np.where(Qc_model == 0, np.nan, Qc_model)
    Qm_model = np.where(Qm_model == 0, np.nan, Qm_model)
    Qc_Qs = np.where(Qc_Qs == 0, np.nan, Qc_Qs)

    ## Save 3D thermal model results as NetCDF file
    # Create empty folder to store modeling results
    if (Run_D95) and (quadtree):        # If quadtree setting is on
        # Define folder name
        crustal_T_folder = f'Best_T_mean_{minimum_pixel_size}_min{min_EQ_cutoff}_best{goodfit}_quad_{Iteration}_{start_time}'

    elif (Run_D95) and (quadtree == False):     # If quadtree setting is off
        # Define folder name
        crustal_T_folder = f'Best_T_mean_{minimum_pixel_size}_min{min_EQ_cutoff}_best{goodfit}_fixed_{Iteration}_{start_time}'
    elif Run_D95 == False:     # If D95 is off
        # Define folder name
        crustal_T_folder = f'Best_T_mean_{minimum_pixel_size}_best{goodfit}_{Iteration}_{start_time}'

    # Create empty folders 
    os.makedirs(crustal_T_folder, exist_ok = True)

    # Save 3D thermal model results as NetCDF file
    Crustal_T_shape = np.shape(Crustal_T)                                           # Get shape of thermal model result
    lon_dim = Crustal_T_shape[1]                                                    # Get shape in x direction
    lat_dim = Crustal_T_shape[0]                                                    # Get shape in y direction
    max_z = max(len(profile) for row in Crustal_T for profile in row if profile is not None)   # Maximum length of depth direction

    # Create a new 3D array to store model result
    padded_Crustal_T = np.full((lat_dim, lon_dim, max_z), np.nan, dtype = 'f4')     # Create an empty array of the model size and filled with NaN
    for i_y, row in enumerate(Crustal_T):                                           # Loop through thermal model result
        for i_x, profile in enumerate(row):
            if profile is not None:
                padded_Crustal_T[i_y, i_x, :len(profile)] = profile                     # Fill data for each profile

    # If the heat diffusion option is on
    if Diffusion_2D:
        print('Start 2D lateral heat diffusion on the output results.')
        print('')
        padded_Crustal_T_diffuse = padded_Crustal_T.copy()
        dx = minimum_pixel_size_deg * 111 * 1000                          # degree to meter; dx = dy
        kappa = 10 ** -6                                                  # Thermal diffusivity; m^2 s^-1
        t_total = (Diffusion_time * (10 ** 6)) * 365.25 * 24 * 60 * 60    # Total diffusion time; Myr to second

        dt = (dx ** 2) / (4 * kappa)                                      # Time step; change of time
        nt = int(t_total / dt)                                            # Total number of time step

        # Loop through the depth
        for depth in tqdm(range(max_z)):
            valid_mask = ~np.isnan(padded_Crustal_T_diffuse[:, :, depth])         # Get a mask where pixel is not NaN
            for t_step in range(nt):                                              # Loop through the amount of diffusion finite time step
                for x in range(1, padded_Crustal_T_diffuse.shape[0] - 1):         # Loop through x dimension
                    for y in range(1, padded_Crustal_T_diffuse.shape[1] - 1):     # Loop through y dimension
                        # Only compute for valid points
                        if valid_mask[x, y]:
                            # Find the valid neighbors
                            neighbors = []                                                    # Make an empty list to find neighboring temperature
                            if valid_mask[x + 1, y] == True:                                  # Check if right temperature pixel is valid
                                neighbors.append(padded_Crustal_T_diffuse[x + 1, y, depth])   # Get the temperature
                            if valid_mask[x - 1, y] == True:                                  # Check if left temperature pixel is valid
                                neighbors.append(padded_Crustal_T_diffuse[x - 1, y, depth])   # Get the temperature
                            if valid_mask[x, y + 1] == True:                                  # Check if above temperature pixel is valid
                                neighbors.append(padded_Crustal_T_diffuse[x, y + 1, depth])   # Get the temperature
                            if valid_mask[x, y - 1] == True:                                  # Check if the below temperature pixel is valid
                                neighbors.append(padded_Crustal_T_diffuse[x, y - 1, depth])   # Get the temperature

                            # If there are valid neighbors
                            if len(neighbors) == 4:
                                # Update the existing depth slice and update it with the diffused temperature at the next time step 
                                # Numerically model the 2D heat equation using finite central difference approximation
                                # See supplementary document for details
                                padded_Crustal_T_diffuse[x, y, depth] = padded_Crustal_T_diffuse[x, y, depth] + kappa * dt * \
                                                                        ((sum(neighbors) - (4 * padded_Crustal_T_diffuse[x, y, depth])) / (dx ** 2))

        T_20km_diffuse = padded_Crustal_T_diffuse[:, :, 20]

    # Make an array to store the latitude and longitude of the center of each pixel
    lat_arr_center = np.linspace((lat_max - (minimum_pixel_size_deg / 2)), 
                                ((lat_max - (minimum_pixel_size_deg * lat_dim)) + (minimum_pixel_size_deg / 2)), lat_dim)
    lon_arr_center = np.linspace((lon_min + (minimum_pixel_size_deg / 2)),
                                ((lon_min + (minimum_pixel_size_deg * lon_dim)) - (minimum_pixel_size_deg / 2)), lon_dim)
    
    # If quadtree setting is on
    if (Run_D95) and (quadtree):
        # Create NetCDF file
        nc_file = Dataset(f'{crustal_T_folder}/best_T_mean_{minimum_pixel_size}_min{min_EQ_cutoff}_best35_quad_{Iteration}_{start_time}.nc',
                    'w', format = 'NETCDF4')
    # If quadtree setting is off
    elif (Run_D95) and (quadtree == False):
        # Create NetCDF file
        nc_file = Dataset(f'{crustal_T_folder}/best_T_mean_{minimum_pixel_size}_min{min_EQ_cutoff}_best35_fixed_{Iteration}_{start_time}.nc',
                    'w', format = 'NETCDF4')
    elif Run_D95 == False:
        # Create NetCDF file
        nc_file = Dataset(f'{crustal_T_folder}/best_T_mean_{minimum_pixel_size}_best35_{Iteration}_{start_time}.nc',
                    'w', format = 'NETCDF4')

    # Define file dimensions
    nc_file.createDimension('longitude', lon_dim)                 # Define longitude dimension (x direction)
    nc_file.createDimension('latitude', lat_dim)                  # Define latitude dimension (y direction)
    nc_file.createDimension('Depth', max_z)                       # Define depth dimension (z direction)

    # Create variables
    Lon_var = nc_file.createVariable('longitude', 'f4', ('longitude',))
    Lat_var = nc_file.createVariable('latitude', 'f4', ('latitude',))
    Temperature_var = nc_file.createVariable('Temperature', 'f4', ('latitude', 'longitude', 'Depth'))
    if Diffusion_2D:
        Diffused_Temperature_var = nc_file.createVariable('Diffused_Temperature', 'f4', ('latitude', 'longitude', 'Depth'))
        T_20km_diffuse_var = nc_file.createVariable('Temperature_at_20km_diffuse', 'f4', ('latitude', 'longitude'))
    Qs_var = nc_file.createVariable('Surface_heat_flow', 'f4', ('latitude', 'longitude'))
    if Run_D95:
        D95_var = nc_file.createVariable('D95', 'f4', ('latitude', 'longitude'))
        D95_uncer_var = nc_file.createVariable('D95_uncertainty', 'f4', ('latitude', 'longitude'))
    if Run_Curie:
       Curie_var = nc_file.createVariable('Curie_depth', 'f4', ('latitude', 'longitude'))    
    if Run_Moho:
        Mohoz_var = nc_file.createVariable('Crustal_thickness', 'f4', ('latitude', 'longitude'))
        MohoT_var = nc_file.createVariable('Moho_temperature', 'f4', ('latitude', 'longitude'))
    if Run_LAB:
        LAB_var = nc_file.createVariable('LAB_depth', 'f4', ('latitude', 'longitude'))
    dTdz_var = nc_file.createVariable('Crustal_dTdz', 'f4', ('latitude', 'longitude'))
    dTdz_250_max_var = nc_file.createVariable('dTdz_250_max', 'f4', ('latitude', 'longitude'))
    T_20km_var = nc_file.createVariable('Temperature_at_20km', 'f4', ('latitude', 'longitude'))
    R2_var = nc_file.createVariable('R2', 'f4', ('latitude', 'longitude'))
    With_or_withoutD95_mask_var = nc_file.createVariable('D95_mask', 'f4', ('latitude', 'longitude'))
    Qs_model_var = nc_file.createVariable('Modeled_surface_heat_flow', 'f4', ('latitude', 'longitude'))
    Qc_model_var = nc_file.createVariable('Modeled_radiogenic_heat_flow', 'f4', ('latitude', 'longitude'))
    Qm_model_var = nc_file.createVariable('Modeled_mantle_heat_flow', 'f4', ('latitude', 'longitude'))
    Qc_Qs_var = nc_file.createVariable('Qc_Qs', 'f4', ('latitude', 'longitude'))

    # Create the CRS variable
    crs_var = nc_file.createVariable('crs', np.int8, ())
    crs_var.standard_name = 'crs'
    crs_var.grid_mapping_name = 'latitude_longitude'
    crs_var.crs_wkt = ("GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',"
                      "SPHEROID['WGS_1984',6378137.0,298.257223563]],"
                      "PRIMEM['Greenwich',0.0],"
                      "UNIT['Degree',0.0174532925199433]]")

    # Assign data to NetCDF file varaibles
    Lat_var[:] = lat_arr_center                                       # Latitude (center of pixel)
    Lon_var[:] = lon_arr_center                                       # Longitude (center of pixel)
    if Diffusion_2D:
        Diffused_Temperature_var[:, :, :] = padded_Crustal_T_diffuse  # 3D thermal model results with static heat diffusion
        T_20km_diffuse_var[:, :] = T_20km_diffuse                     # Temperature at 20km depth (diffused)
    Temperature_var[:, :, :] = padded_Crustal_T                       # 3D thermal model results
    Qs_var[:, :] = hf                                                 # Surface heat flow
    if Run_D95:
        D95_var[:, :] = D_depth                                       # D95
        D95_uncer_var[:, :] = D95_uncer                               # D95 uncertainty
    if Run_Curie:
        Curie_var[:, :] = Curie_depth_arr                             # Curie depth
    if Run_Moho:
        Mohoz_var[:, :] = zm_arr                                      # Crustal thickness
        MohoT_var[:, :] = Moho_T_arr                                  # Moho temperature
    if Run_LAB:
        LAB_var[:, :] = LAB_arr                                       # LAB depth
    dTdz_var[:, :] = dTdz                                             # Crustal scale geothermal gradient
    dTdz_250_max_var[:, :] = dTdz_250_max                             # Maximum geothermal gradient from 0 to 250 °C
    T_20km_var[:, :] = T_20km                                         # Temperature at 20km depth
    R2_var[:, :] = R2                                                 # Coefficient of determination
    With_or_withoutD95_mask_var[:, :] = With_or_withoutD95_mask       # Mask to determine if thermal profile is constrained with D95 or not
    Qs_model_var[:, :] = Qs_model                                     # Modeled surface heat flow
    Qc_model_var[:, :] = Qc_model                                     # Modeled radiogenic heat flow
    Qm_model_var[:, :] = Qm_model                                     # Modeled mantle heat flow
    Qc_Qs_var[:, :] = Qc_Qs                                           # Qc / Qs
    
    # Add metadata to each variable layer
    Lat_var.units  = 'degrees_north'
    Lon_var.units  = 'degrees_east'

    Temperature_var.long_name = 'Crustal temperature'
    Temperature_var.units  = '°C'
    Temperature_var.grid_mapping = 'crs'

    if Diffusion_2D: 
        Diffused_Temperature_var.long_name = 'Crustal temperature with static heat diffusion'
        Diffused_Temperature_var.units  = '°C'
        Diffused_Temperature_var.grid_mapping = 'crs'

        T_20km_diffuse_var.long_name = 'Temperature at 20km depth (statically diffused)'
        T_20km_diffuse_var.units  = '°C'
        T_20km_diffuse_var.grid_mapping = 'crs'

    Qs_var.long_name = 'Surface heat flow'
    Qs_var.units  = 'mW m-2'
    Qs_var.grid_mapping = 'crs'

    if Run_D95:
        D95_var.long_name = 'Seismogenic thickness'
        D95_var.units  = 'km'
        D95_var.grid_mapping = 'crs'

        D95_uncer_var.long_name = 'Seismogenic thickness uncertainty'
        D95_uncer_var.units  = 'km'
        D95_uncer_var.grid_mapping = 'crs'

    if Run_Curie:
        Curie_var.long_name = 'Curie depth'
        Curie_var.units = 'km'
        Curie_var.grid_mapping = 'crs'

    if Run_Moho:
        Mohoz_var.long_name = 'Crustal thickness'
        Mohoz_var.units  = 'km'
        Mohoz_var.grid_mapping = 'crs'
    
        MohoT_var.long_name = 'Moho temperature'
        MohoT_var.units  = '°C'
        MohoT_var.grid_mapping = 'crs'

    if Run_LAB:
        LAB_var.long_name = 'Lithosphere-asthenosphere boundary depth'
        LAB_var.units  = 'km'
        LAB_var.grid_mapping = 'crs'
        
    dTdz_var.long_name = 'Crustal scale geothermal gradient'
    dTdz_var.units  = '°C km-1'
    dTdz_var.grid_mapping = 'crs'

    dTdz_250_max_var.long_name = 'Maximum geothermal gradient from surface to depth at 250°C'
    dTdz_250_max_var.units  = '°C km-1'
    dTdz_250_max_var.grid_mapping = 'crs'

    T_20km_var.long_name = 'Temperature at 20km depth'
    T_20km_var.units  = '°C'
    T_20km_var.grid_mapping = 'crs'

    R2_var.long_name = 'Coefficient of determination'
    R2_var.units  = 'unitless'
    R2_var.grid_mapping = 'crs'

    With_or_withoutD95_mask_var.long_name = 'Mask of number of constraints being used'
    With_or_withoutD95_mask_var.details = '(1 = all constraints; 2 = all constraints except one; 3 = all constraints except two;4 is with only one constrain'
    With_or_withoutD95_mask_var.units  = 'unitless'
    With_or_withoutD95_mask_var.grid_mapping = 'crs'

    Qs_model_var.long_name = 'Modeled surface heat flow'
    Qs_model_var.units = 'mW m-2'
    Qs_model_var.grid_mapping = 'crs'

    Qc_model_var.long_name = 'Modeled radiogenic heat flow'
    Qc_model_var.units = 'mW m-2'
    Qc_model_var.grid_mapping = 'crs'

    Qm_model_var.long_name = 'Modeled mantle heat flow'
    Qm_model_var.units = 'mW m-2'
    Qm_model_var.grid_mapping = 'crs'

    Qc_Qs_var.long_name = 'Ratio of radiogenic heat flow and surface heat flow'
    Qc_Qs_var.units = 'unitless'
    Qc_Qs_var.grid_mapping = 'crs'

    # Add model boundary metadata
    nc_file.geospatial_lat_min = lat_min
    nc_file.geospatial_lat_max = lat_max
    nc_file.geospatial_lon_min = lon_min
    nc_file.geospatial_lon_max = lon_max
    nc_file.geospatial_lat_resolution = minimum_pixel_size_deg
    nc_file.geospatial_lon_resolution = minimum_pixel_size_deg
    nc_file.geospatial_lon_units = 'degree'
    nc_file.geospatial_lat_units = 'degree'
    
    # Make a model boundary polygon in WKT format
    nc_file.geospatial_bounds = f'POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))' 

    # Add model metadata
    nc_file.Authors = """Lee, T.; Affiliation: Nevada Bureau of Mines and Geology, UNR; Department of Geological Sciences and Engineering, UNR
    Zuza, A. V.; Affiliation: Nevada Bureau of Mines and Geology, UNR
    Trugman, T. D.; Affiliation: Nevada Seismological Laboratory, UNR
    Vlaha, D. R.; Affiliation: Nevada Bureau of Mines and Geology, UNR; Department of Geological Sciences and Engineering, UNR
    Cao, W.; Affiliation: Department of Geological Sciences and Engineering, UNR"""

    nc_file.Corr_Author = 'Terry Lee'
    nc_file.Corr_author_contact = 'terrywaihol@unr.edu'                               
    nc_file.Date_created = start_time                                                    
    nc_file.Project = 'Continental Crust Thermal Model (CCTM)'                 
    nc_file.description = 'This dataset contains 3D volume of continental geotherms, D95 result, and relevant temperature constraints.'
    nc_file.references = 'Lee et al., 2025. Continental thermal model: Implications to crustal rheology and beyond.'
    nc_file.Time_of_Creation = f'{start_time}'

    nc_file.close()

    ## Creating an output config file
    config_output = pd.DataFrame()
    config_output['Start_time'] = [start_time]
    config_output['lat_max'] = [lat_max]
    config_output['lat_min'] = [lat_min]
    config_output['lon_max'] = [lon_max]
    config_output['lon_min'] = [lon_min]
    config_output['Run_D95'] = [Run_D95]
    config_output['Run_Curie'] = [Run_Curie]
    config_output['Run_Moho'] = [Run_Moho]
    config_output['Run_LAB'] = [Run_LAB]
    config_output['maximum_pixel_size'] = [maximum_pixel_size]
    if Run_D95 and quadtree:
        config_output['min_EQ_cutoff'] = [min_EQ_cutoff]
        config_output['quadtree'] = [quadtree]
        config_output['fixed_pixel_size'] = [fixed_pixel_size]
        config_output['D95_T_config'] = [D95_T_config]
    if Run_Curie:
        config_output['Curie_depth_T'] = [Curie_depth_T]
        config_output['Curie_T_uncer'] = [Curie_T_uncer]
    if Run_LAB:
        config_output['LAB_T_config'] = [LAB_T_config]
    config_output['Iteration'] = [Iteration]
    config_output['T0_range_min'] = [T0_range[0]]
    config_output['T0_range_max'] = [T0_range[1]]
    config_output['k_range_min'] = [k_range[0]]
    config_output['k_range_max'] = [k_range[1]]
    config_output['H0_range_min'] = [H0_range[0]]
    config_output['H0_range_max'] = [H0_range[1]]
    config_output['qm_range_min'] = [qm_range[0]]
    config_output['qm_range_max'] = [qm_range[1]]
    config_output['goodfit'] = [goodfit]
    config_output['moderatefit'] = [moderatefit]      
    config_output['Max_goodfit_profile'] = [Max_goodfit_profile]
    config_output['Uncertainty_box_fitting'] = [Uncertainty_box_fitting]
    config_output['Diffusion_2D'] = [Diffusion_2D]
    config_output['Diffusion_time'] = [Diffusion_time]

    config_output.to_csv(f'{crustal_T_folder}/output_config.csv')

    end_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')   # Get current time
    print('Model end time:', end_time)