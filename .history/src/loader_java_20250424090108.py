import numpy as np
import geopandas as gpd
from pyprojroot import here

def load_data(data_path = here() / "data"):
    """
    Load all necessary data for the model
    
    Returns:
        Dictionary containing loaded data
    """
    # Load grid points coordinates
    x = np.load(data_path / "lat_lon_x_java.npy")
    
    # Load regional data
    pol_pts_lo = np.load(data_path / "pol_pts_java_lo.npy")
    pol_pts_hi = np.load(data_path / "pol_pts_java_hi.npy")
    
    # Load shapefiles
    df_lo = gpd.read_file(data_path / "java_prov.shp")
    df_hi = gpd.read_file(data_path / "java_dist.shp")
    
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