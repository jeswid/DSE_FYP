import jax.numpy as jnp

def dist_euclid(x, z):
    """
    Computes the Euclidean distance between two sets of points (e.g., spatial coordinates).
    
    This is typically used as part of a kernel function for Gaussian Processes
    to compute pairwise distances between input locations.

    Args:
        x: Array of shape (n_x, d) or (n_x,) representing coordinates (e.g., lat/lon)
        z: Array of shape (n_z, d) or (n_z,) representing coordinates

    Returns:
        A (n_x, n_z) matrix of pairwise Euclidean distances
    """
    x = jnp.array(x) 
    z = jnp.array(z) 

    # Reshape 1D inputs into column vectors
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1) 
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1) 
    n_x, m = x.shape 
    n_z, m_z = z.shape 
    assert m == m_z # Ensure dimensionality matches

    delta = jnp.zeros((n_x,n_z))  # Initialize distance matrix

    # Compute squared distance component-wise
    for d in jnp.arange(m):
        x_d = x[:,d] # Extract d-th dimension of x (shape: [n_x])
        z_d = z[:,d] # Extract d-th dimension of z (shape: [n_z])
        delta += (x_d[:,jnp.newaxis] - z_d)**2 # Broadcasting to compute pairwise (x_i - z_j)^2
    return jnp.sqrt(delta) # Return Euclidean distance matr

def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-4):
    """
    Computes the squared exponential (RBF) kernel matrix for a Gaussian Process.
    
    This kernel expresses the assumption that nearby points in input space have similar values.

    Args:
        x: Input locations of shape (n_x, d)
        z: Input locations of shape (n_z, d)
        var: Output variance (kernel amplitude)
        length: Length scale controlling smoothness
        noise: Observation noise term
        jitter: Small constant added to diagonal for numerical stability

    Returns:
        Kernel matrix of shape (n_x, n_z)
    """
    dist = dist_euclid(x, z)  # Pairwise Euclidean distances
    deltaXsq = jnp.power(dist/ length, 2.0) # Scale distances by length scale
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k 

def M_g(M, g):
    '''
    Matrix multiplication for aggregating GP draws over polygons
    
    Args:
        M: Matrix with binary entries m_ij, showing whether point j is in polygon i
        g: Vector of GP draws over grid
    
    Returns:
        Vector of sums over each polygon
    '''
    M = jnp.array(M)
    g = jnp.array(g).T
    return(jnp.matmul(M, g)) 