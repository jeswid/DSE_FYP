import numpy as np
import geopandas as gpd
from pyprojroot import here

"""""
This function loads all required data, particularly the matrix M, all the grid
pts x, and the province and district level dengue cases final dataset
"""""

def load_data(data_path = here() / "data" / "processed"):
    """
    Load all necessary data for the model
    
    Returns:
        Dictionary containing loaded data
    """
    # Load grid points coordinates x
    x = np.load(data_path / "lat_lon_x_jkt.npy")
    
    # Load the vectors to be assigned as matrices M-lo and M-hi
    pol_pts_lo = np.load(data_path / "pol_pts_jkt_lo.npy")
    pol_pts_hi = np.load(data_path / "pol_pts_jkt_hi.npy")
    
    # Load shapefiles at district and province level
    df_lo = gpd.read_file(data_path / "jkt_prov.shp")
    df_hi = gpd.read_file(data_path / "jkt_dist.shp")
    
    # Create the M matrices for aggregation
    # This function would need to be implemented based on how the original notebook creates these matrices
    # M_lo and M_hi are matrices showing whether point j is in polygon i
    
    return {
        "x": x,
        "pol_pts_lo": pol_pts_lo,
        "pol_pts_hi": pol_pts_hi,
        "df_lo": df_lo,
        "df_hi": df_hi,
        # Add other necessary data here
    } 