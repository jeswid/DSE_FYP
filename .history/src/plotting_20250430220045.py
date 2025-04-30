import numpy as np
import plotly.express as px

def plot_process(gp_draws):
    """
    Plot multiple GP draws as line plots
    
    Args:
        gp_draws: Array of GP draws with shape (num_samples, num_regions)
    
    Returns:
        A Plotly figure object with overlaid GP sample lines.
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
    Plot a choropleth map of incidence values with annotated labels.
    
    Args:
        geodf: GeoDataFrame containing geometries and a column of incidence values
        plot_col: Name of the column in geodf to visualize
        title: Title of the plot
        ax: Matplotlib Axes object to plot on
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        cmap: Colormap to apply (e.g., 'viridis', 'plasma')
    
    Returns:
        None (modifies the passed-in axes with the plotted map)
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