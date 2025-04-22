import jax.numpy as jnp

def dist_euclid(x, z):
    """
    Computes Eucledian Distance Between Regions. This function is used by
    exp_sq_kernel function (kernel function for gaussian processes)
    """
    x = jnp.array(x) # (ngrid_pts, lat/lon) <- i.e (7304,2)
    z = jnp.array(z) # (ngrid_pts, lat/lon) <- i.e (7304,2)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1) #(2618,) -> (7304,1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1) #(2618,) -> (7304,1)
    n_x, m = x.shape # 7304 , 2
    n_z, m_z = z.shape # 7304 , 2
    assert m == m_z
    delta = jnp.zeros((n_x,n_z)) #(ngrid_pts,ngrid_pts) <- i.e (7304,7304)
    for d in jnp.arange(m):
        x_d = x[:,d] #(ngrid_pts-lat/lon,) <- (7304,)
        z_d = z[:,d] #(ngrid_pts-lat/lon,) <- (7304,)
        delta += (x_d[:,jnp.newaxis] - z_d)**2 # (7304,7304)

    return jnp.sqrt(delta) #(7304,7304)

def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-4):
    """
    Exponential squared kernel (RBF kernel) for Gaussian processes
    
    Args:
        x: First set of points (n_x, d)
        z: Second set of points (n_z, d)
        var: Kernel variance
        length: Length scale
        noise: Noise term
        jitter: Small constant for numerical stability
    
    Returns:
        Kernel matrix of shape (n_x, n_z)
    """
    dist = dist_euclid(x, z) #(7304, 7304)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k # (ngrid_pts, ngrid_pts) <- (7304,7304)

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