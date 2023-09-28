import numpy as np

def filter_out(df, gdf):

    '''
    Filter the Incidents data according to the following criteria:
    1. Filter out incidents that have missing or incomplete data.   
    2. Fiter out incidents with no duration.
    3. Filter out incidents that last longer than a day.
    4. Filter out incidents that are not located in the Netherlands.
    5. Filter out incidents that occurred in roads that are not in the NWD Road Network Data.

    Input: Incidents DataFrame (before filtering), Road Network GeoDataFrame.
    Output: Incidents DataFrame (after filtering)

    It adds an additional column to the filtered DataFrame: 'Duration [min]'
    It does not reset the DataFrame index
    '''
    # Create deep copy of original df
    filtered_df = df.copy(deep = True)

    # Calculate duration of each incident
    filtered_df['Duration [min]'] = (filtered_df['endtime_new'] - filtered_df['starttime_new']) / np.timedelta64(1, 'm')

    # Drop rows that contain missing values
    filtered_df.dropna(axis=0, inplace=True)

    # Remove values corresponding to durations equal to zero or larger than 1 day
    filtered_df = filtered_df[filtered_df['Duration [min]'].between(0,1440, inclusive='right')]

    # Get the road numbers that are mapped in the GeoDataFrame
    mapped_roads = gdf['WEGNR_HMP'].unique()

    # Remove entries from the DataFrame that are not in the mapped road numbers
    filtered_df = filtered_df[filtered_df['vild_primair_wegnummer'].isin(mapped_roads)]

    # This is a very specific case of a road number that has to be removed
    filtered_df = filtered_df[filtered_df['vild_primair_wegnummer'] != 'N46']

    # There are a couple of points located in Germany, these are also removed.
    filtered_df = filtered_df[filtered_df['primaire_locatie_lengtegraad'] <= 7.22777777778]

    # Return fitered DataFrame
    return filtered_df