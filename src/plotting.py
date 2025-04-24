import numpy as np
import plotly.express as px

def plot_process(gp_draws):
    """
    Plot multiple GP draws as line plots
    
    Args:
        gp_draws: Array of GP draws with shape (num_samples, num_regions)
    """
    p = px.line()
    for i in range(gp_draws.shape[1]):
        p.add_scatter(
            x=np.arange(gp_draws.shape[2]), 
            y=gp_draws[0,i, :],
            line_color='rgb(31, 119, 180)',  # A nice blue color
            opacity=0.3  # Add transparency
        )

    p.update_layout(
        template="plotly_white",
        xaxis_title="region", 
        yaxis_title="num cases",
        showlegend=False
    )
    return p  # Return the figure instead of showing it directly

def plot_incidence_map(geodf,plot_col="incidence", title="Incidence", ax=None, vmin=0.001, vmax=0.008, cmap="viridis"):
    """
    Plot incidence data on a map with value annotations.
    
    Parameters:
    -----------
    geodf : geopandas.GeoDataFrame
        GeoDataFrame containing the incidence data and geometry
    plot_col : str
        Column name to plot
    title : str
        Title for the plot
    ax : matplotlib.axes.Axes
        The axes to plot on
    vmin, vmax : float
        Minimum and maximum values for the color scale
    cmap : str
        Colormap to use for the plot
        
    Returns:
    --------
    None
    """
    # Plot the map
    geodf.plot(
        column=plot_col,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        ax=ax,
    )
    
    # Add text labels with incidence values
    for idx, row in geodf.iterrows():
        # Get centroid coordinates of each polygon
        centroid = row.geometry.centroid
        # Format incidence value as percentage with 2 decimal places
        value = f"{row[plot_col]*100:.3f}%"
        # Add text with white background for visibility
        ax.annotate(
            value,
            xy=(centroid.x, centroid.y),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            color='white',
            bbox=dict(
                facecolor='black',
                alpha=0.7,
                edgecolor='none',
                pad=1
            )
        )
    
    ax.set_title(title)