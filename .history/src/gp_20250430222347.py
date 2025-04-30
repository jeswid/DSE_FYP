import jax.numpy as jnp
import jax.nn as nn
import numpyro
import numpyro.distributions as dist
from pyprojroot import here
import sys
from numpyro.handlers import condition
# Add code src directory to sys.path
sys.path.append(str(here() / "src"))
from kernels import M_g, exp_sq_kernel

def gp_aggr(config=None):
    """
    Gaussian Process aggregation model for spatial data
    
    Args:
        config: Dictionary containing model configuration:
            x: Spatial grid points (num_grid_points, 2)
            gp_kernel: Gaussian Process kernel function (default: exp_sq_kernel)
            noise: Noise parameter (default: 1e-2)
            jitter: Jitter parameter for numerical stability (default: 1e-2)
            M_lo: Low resolution aggregation matrix (1, 100)
            M_hi: High resolution aggregation matrix (5, 100)
            kernel_length: Prior distribution for kernel length (default/initial: InverseGamma(4,1))
            kernel_var: Prior distribution for kernel variance (default/initial: LogNormal(0,0.1))
    
    Returns:
        Sampled GP values and their aggregations
    """
    if config is None:
        config = {}
        
    # Set default parameters --> section 5.2.4
    x = config.get('x', None)
    gp_kernel = config.get('gp_kernel', exp_sq_kernel)
    noise = config.get('noise', 1e-4)
    jitter = config.get('jitter', 1e-4)
    M_lo = config.get('M_lo', None) # Low-res aggregation matrix M (e.g., provinces)
    M_hi = config.get('M_hi', None) # High-res aggregation matrix M (e.g., districts)
    kernel_length_prior = config.get('kernel_length', dist.InverseGamma(4, 1)) # Section: Prior for kernel length (initial)
    kernel_var_prior = config.get('kernel_var', dist.LogNormal(0, 0.1)) # Section: Prior for kernel variance (initial)

    # GP hyperparameters --> section 5.2.4
    kernel_length = numpyro.sample("kernel_length", kernel_length_prior)
    kernel_var = numpyro.sample("kernel_var", kernel_var_prior)

    # section 
    log_mean = numpyro.sample("log_mean", dist.Normal(jnp.log(20), 0.1))
    
    # Create kernel with smaller variance --> section 5.2.3, 5.2.4
    k = gp_kernel(x, x, kernel_var, kernel_length, noise, jitter)
    
    # Sample GP on log scale with appropriate mean --> section 5.2.3
    log_f = numpyro.sample("log_f", dist.MultivariateNormal(
        loc=jnp.ones(x.shape[0]) * log_mean,
        covariance_matrix=k))
    
    # Transform back to original scale --> section 4.2.3
    f = jnp.exp(log_f)  
    
    # section 5.2.5
    gp_aggr_lo = numpyro.deterministic("gp_aggr_lo", M_g(M_lo, f))
    gp_aggr_hi = numpyro.deterministic("gp_aggr_hi", M_g(M_hi, f))

    # section 5.2.6
    gp_aggr = numpyro.deterministic("gp_aggr", jnp.concatenate([gp_aggr_lo, gp_aggr_hi]))

    return gp_aggr

def gp_aggr_count(config=None):
    """
    Gaussian Process aggregation model for modelling spatial count data
    
    Args:
        config: Dictionary containing model configuration:
            x: Spatial grid points (num_grid_points, 2)
            gp_kernel: Gaussian Process kernel function (default: exp_sq_kernel)
            noise: Noise parameter (default: 1e-4)
            jitter: Jitter parameter for numerical stability (default: 1e-4)
            M_lo: Low resolution aggregation matrix (num_regions, num_grid_points)
            M_hi: High resolution aggregation matrix (num_regions, num_grid_points)
            kernel_length: Prior distribution for kernel length (default: InverseGamma(4,1))
            kernel_var: Prior distribution for kernel variance (default: LogNormal(0,0.1))
            total_population: Total population data (num_regions,)
            pop_density: Population density data (num_regions,)
            urban_frac: Urban fraction data (num_regions,)
            hdi_index: HDI index data (num_regions,)
            count: Count data (num_regions,)
            prior_pred: True if prior predictive, False if posterior predictive
    """
    if config is None:
        config = {}
        
    # Set defaults
    x = config.get('x', None)
    gp_kernel = config.get('gp_kernel', exp_sq_kernel)
    noise = config.get('noise', 1e-4)
    jitter = config.get('jitter', 1e-4)
    M_lo = config.get('M_lo', None)
    M_hi = config.get('M_hi', None)
    kernel_length_prior = config.get('kernel_length', dist.InverseGamma(4, 1))
    kernel_var_prior = config.get('kernel_var', dist.LogNormal(0, 0.1))
    pop_density = config.get('pop_density', None)
    urban_frac = config.get('urban_frac', None)
    hdi_index = config.get('hdi_index', None)
    total_population = config.get('total_population', None)
    count_data = config.get('count', None)
    is_prior_pred = config.get('prior_pred', False)
    # GP
    config_gp = config.copy()
    config_gp['x'] = x
    config_gp['gp_kernel'] = gp_kernel
    config_gp['noise'] = noise
    config_gp['jitter'] = jitter
    config_gp['M_lo'] = M_lo
    config_gp['M_hi'] = M_hi
    config_gp['kernel_length'] = kernel_length_prior
    config_gp['kernel_var'] = kernel_var_prior
    # Call gp_aggr
    aggr_gp = numpyro.deterministic("aggr_gp", gp_aggr(config_gp))

    # Fixed effects
    b0 = numpyro.sample("b0", dist.Normal(100, 10))  # Intercept
    b_pop_density = numpyro.sample("b_pop_density", dist.Normal(0, 1))  # Effect of population density
    b_hdi = numpyro.sample("b_hdi", dist.Normal(0, 1))  # Effect of HDI
    b_urban = numpyro.sample("b_urban", dist.Normal(0, 1))  # Effect of urbanicity

    # Linear predictor
    # lp = numpyro.deterministic("lp", (b0 + 
    #                                   aggr_gp + 
    #                                   b_pop_density * pop_density + 
    #                                   b_hdi * hdi_index + 
    #                                   b_urban * urban_frac))  #
    lp = numpyro.deterministic("lp", nn.softplus(b0 + 
                                      aggr_gp + 
                                      b_pop_density * pop_density + 
                                      b_hdi * hdi_index + 
                                      b_urban * urban_frac)) 
    
   
    sigma = numpyro.sample("sigma", dist.HalfNormal(50))
    pred_cases = numpyro.sample(
        "pred_cases",
        dist.Normal(lp, sigma),
        obs=None if is_prior_pred else count_data
    )
    return pred_cases