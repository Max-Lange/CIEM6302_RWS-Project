import numpy as np
import pandas as pd
import geopandas as gpd

def filter_out(df, gdf):

    '''
    Filter the Incidents data according to the following criteria:

    1. Geographical data filtering. It filters out accidents with no road segment in the NWD Road Network Data.  
    2. Fiter out incidents with no duration.
    3. Filter out incidents that last longer than a day.
 
    Input: Incidents DataFrame (before filtering), Road Network GeoDataFrame.
    Output: Incidents DataFrame (after filtering)

    It adds an additional column to the filtered DataFrame: 'Duration [min]'

    It does not reset the DataFrame index
    '''
    
    #Define global CRS projector
    global_crs = 'EPSG:4326'
    
    # 1. Geographical Data Filtering

    # 1.1 Create deep copy of original df and gdf
    filtered_df = df.copy(deep = True)
    gdf2 = gdf.copy(deep = True)

    # 1.2 Create an incidents gdf
    filtered_gdf = gpd.GeoDataFrame(filtered_df, 
                                 geometry=gpd.points_from_xy(filtered_df['primaire_locatie_lengtegraad'], filtered_df['primaire_locatie_breedtegraad']),
                                 crs = global_crs)

    # 1.3 Create a buffer of 500 meters to each side of each linestring in the mapped roads
    gdf2['geometry'] = gdf2['geometry'].buffer(50)
   
    # 1.4 Convert to EPSG:4326
    gdf2.to_crs(global_crs, inplace= True)

    # 1.5 Initialize filtering with the first segment
    # --- This will return boolean values for each point (1 = point is in the buffer of segment, 0 otherwise)
    points_filter = pd.DataFrame(filtered_gdf.within(gdf2.loc[0, 'geometry']), columns = ['Filter'])

    # 1.6 Update filtering boolean values for all segments
    for i in range(1, len(gdf2)):
        points_filter['Filter'] += filtered_gdf.within(gdf2.loc[i, 'geometry'])

    # 1.7 Apply filter to incidents df (renamed due to the deep copy)
    filtered_df = filtered_df[points_filter['Filter']]

    # 2. Filter incidents by duration
    # 2.1 Calculate duration of each incident
    filtered_df['Duration [min]'] = (filtered_df['endtime_new'] - filtered_df['starttime_new']) / np.timedelta64(1, 'm')

    # 2.2 Remove values corresponding to durations equal to zero or larger than 1 day
    filtered_df = filtered_df[filtered_df['Duration [min]'].between(0,1440, inclusive='right')]

    # 3 Return fitered DataFrame
    return filtered_df