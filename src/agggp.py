# -*- coding: utf-8 -*-
"""aggGP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10uPICQYygbbgllcXfOZgaqRP7ta5zCxv
"""

!pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

!pip install numpyro dill

!pip install arviz

import os
import numpy as np
import pandas as pd
import geopandas as gpd

import time

import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer import SVI, Trace_ELBO, Predictive
import numpyro.diagnostics

from termcolor import colored

import dill
import pickle
import arviz as az

import jax
print(jax.devices())

numpyro.set_host_device_count(4)

# Check the host device count using jax.local_device_count()
host_device_count = jax.local_device_count()
print(f"Host device count: {host_device_count}")

"""#GP Kernel Function"""

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
    dist = dist_euclid(x, z) #(7304, 7304)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k # (ngrid_pts, ngrid_pts) <- (7304,7304)

"""#Aggregation Functions"""

def M_g(M, g):
    '''
    - $M$ is a matrix with binary entries $m_{ij},$ showing whether point $j$ is in polygon $i$
    - $g$ is a vector of GP draws over grid
    - $maltmul(M, g)$ gives a vector of sums over each polygon
    '''
    M = jnp.array(M)
    g = jnp.array(g).T
    return(jnp.matmul(M, g))

"""#Aggregated Prevalence Model - must edit this to include HDI, population density


"""

def prev_model_gp_aggr(args):
    """Dengue prevalence model with a Gaussian Process"""

    x = args["x"]  # Spatial grid points: (num_grid_points, 2)
    gp_kernel = args["gp_kernel"]  # Gaussian Process kernel
    noise = args["noise"]
    jitter = args["jitter"]

    pop_density = args["pop_density"]  # (num_districts,)
    hdi = args["hdi"]  # (num_districts,)
    M = args["M"]  # (num_districts, num_grid_points) aggregation matrix
    total_cases = args["total_cases"]
    total_population = args["total_population"]

    # GP hyperparameters
    kernel_length = numpyro.sample("kernel_length", args["kernel_length"])
    kernel_var = numpyro.sample("kernel_var", args["kernel_var"])

    # GP Kernel and Sample
    k = gp_kernel(x, x, kernel_var, kernel_length, noise, jitter)
    f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))  # (num_grid_points,)

    # Aggregate GP values to district level
    gp_aggr = numpyro.deterministic("gp_aggr", M @ f)  # (num_districts,)

    # Fixed effects
    b0 = numpyro.sample("b0", dist.Normal(0, 1))  # Intercept
    b_pop_density = numpyro.sample("b_pop_density", dist.Normal(0, 1))  # Effect of population density
    b_hdi = numpyro.sample("b_hdi", dist.Normal(0, 1))  # Effect of HDI

    # Linear predictor
    lp = b0 + gp_aggr + b_pop_density * pop_density + b_hdi * hdi  # (num_districts,)

    # Prevalence probability
    theta = numpyro.deterministic("theta", jax.nn.sigmoid(lp))  # (num_districts,)

    # Binomial likelihood
    observed_cases = numpyro.sample(
        "observed_cases",
        dist.Binomial(total_count=total_population, probs=theta),
        obs=total_cases
    )

    return observed_cases

"""#Load Data"""

# Lat/Lon Values of artificial grid
x = np.load("lat_lon_x_all.npy")

# combined regional data
pol_pts_all = np.load("pol_pts_all.npy")
pt_which_pol_all = np.load("pt_which_pol_all.npy")

#combine the dataframes
df_combined = gpd.read_file("final_combined_divisions.shp")

#check columns
df_combined.head()

#rename Pop_densit to Pop_density
df_combined = df_combined.rename(columns={'Pop_densit': 'Pop_density'})

#check head again
df_combined.head()

"""#Vars needed to be changed (change according to the agg prevalence model parameters)"""

M = jnp.array(pol_pts_all)
pop_density = jnp.array(df_combined["Pop_density"])
hdi = jnp.array(df_combined["HDI"])
test_cases = jnp.array(df_combined["Population"])
cases = jnp.array(df_combined["Cases"])

#print the shape of all the vars above
print(M.shape)
print(pop_density.shape)
print(hdi.shape)
print(test_cases.shape)
print(cases.shape)
print(x.shape)
print(pt_which_pol_all.shape)

"""#Agg GP Model"""

args = {
        "x" : jnp.array(x), # Lat/lon vals of grid points # Shape (num_districts, 2)
        "gp_kernel" : exp_sq_kernel,
        "jitter" : 1e-4,
        "noise" : 1e-4,
        "M" : jnp.array(pol_pts_all), # Aggregation matrix # Shape (num_districts, num_districts)
        # GP Kernel Hyperparams
        "kernel_length" : dist.InverseGamma(3,3), #(,)
        "kernel_var" : dist.HalfNormal(0.05),
        "pop_density": jnp.array(df_combined["Pop_density"]), # Shape (num_districts,)
        "hdi": jnp.array(df_combined["HDI"]), # Shape (num_districts, 2)
        "total_cases" : jnp.array(df_combined["Cases"]),
        "total_population" : jnp.array(df_combined["Population"])
    }

"""#Run MCMC"""

# 🔹 Random keys
run_key, predict_key = random.split(random.PRNGKey(3))

# 🔹 MCMC settings
n_warm = 1000
n_samples = 2000
n_chains = 4

# 🔹 Directory for saving
save_dir = "model_weights"
os.makedirs(save_dir, exist_ok=True)

"""#Save Model"""

# 🔹 Run MCMC for each chain separately to prevent total loss on crash
for chain_id in range(n_chains):
    print(f"\nRunning Chain {chain_id + 1}/{n_chains}...")

    # Generate a separate key for each chain
    chain_run_key = random.split(run_key, n_chains)[chain_id]

    # Initialize MCMC with controlled step size
    mcmc = MCMC(NUTS(prev_model_gp_aggr),
        num_warmup=n_warm,
        num_samples=n_samples,
        num_chains=1)
    # Run the chain
    start = time.time()
    mcmc.run(chain_run_key, args)
    end = time.time()
    t_elapsed_min = round((end - start) / 60)

    # 🔹 Save after each chain completes
    f_path = f"{save_dir}/aggGP_chain{chain_id}_nsamples_{n_samples}_tt{t_elapsed_min}min.pkl"
    with open(f_path, "wb") as file:
        dill.dump(mcmc, file)

    print(f"✅ Saved Chain {chain_id + 1} to {f_path}")
    print(f"⏳ Time taken: {t_elapsed_min} min\n")

#chain 4 crashed before finished training, so run it separately in another machine
# Generate a separate key for each chain
chain_run_key = random.split(run_key, n_chains)[3]

# Initialize MCMC
mcmc = MCMC(NUTS(prev_model_gp_aggr),
        num_warmup=n_warm,
        num_samples=n_samples,
        num_chains=1)


# Run the chain
start = time.time()
mcmc.run(chain_run_key, args)
end = time.time()
t_elapsed_min = round((end - start) / 60)

# 🔹 Save after each chain completes
f_path = f"{save_dir}/aggGP_chain{chain_id}_nsamples_{n_samples}_tt{t_elapsed_min}min.pkl"
with open(f_path, "wb") as file:
    dill.dump(mcmc, file)

print(f"✅ Saved Chain {chain_id + 1} to {f_path}")
print(f"⏳ Time taken: {t_elapsed_min} min\n")

# 🔹 Print total elapsed time
total_end = time.time()
print("\nMCMC Total elapsed time:", round(total_end), "s")
print("MCMC Total elapsed time:", round(total_end / 60), "min")
print("MCMC Total elapsed time:", round(total_end / (60 * 60)), "h")

"""##Combine the MCMC saved files tgt"""

#load the files
save_dir = "model_weights"
n_samples = 2000  # Adjust based on your settings

# Load MCMC objects
mcmc_list = []
for chain_id in range(3):  # Since you only have chains 0 to 2
    f_path = f"{save_dir}/aggGP_chain{chain_id}_nsamples_{n_samples}_tt*.pkl"

    # Find the correct file (in case time differs)
    matching_files = [f for f in os.listdir(save_dir) if f.startswith(f"aggGP_chain{chain_id}_nsamples_{n_samples}_tt")]

    if matching_files:
        with open(os.path.join(save_dir, matching_files[0]), "rb") as file:
            mcmc_list.append(dill.load(file))
    else:
        print(f"Missing Chain {chain_id} file!")

# Extract samples from each MCMC object
samples_list = [mcmc.get_samples() for mcmc in mcmc_list]

# Convert to NumPy array and add a new chain dimension
combined_samples = {k: np.stack([s[k] for s in samples_list], axis=0) for k in samples_list[0].keys()}

# Convert to ArviZ InferenceData format
idata = az.from_dict(posterior=combined_samples)

"""##Check diagnostics"""

#trace plot (check mixing)
az.plot_trace(idata)

#rank plot (ensure good mixing across chains)
az.plot_rank(idata)

#rhat (check diagnostics)
print(az.rhat(idata))

#effective sample size (ESS, >1000 ideally)
print(az.ess(idata))

# Extract posterior samples (already combined)
pos_samples = idata.posterior

# Print MCMC summary
print(az.summary(idata, var_names=["gp_aggr", "kernel_length", "kernel_var"]))

# Compute ESS and R-hat diagnostics
ss = numpyro.diagnostics.summary(combined_samples)

# Compute and print diagnostics
r = np.mean(ss["gp_aggr"]["n_eff"])
print(f"Average ESS for all aggGP effects : {round(r)}")
print(f"Max r_hat for all aggGP effects : {round(np.max(ss['gp_aggr']['r_hat']),2)}")
print(f"kernel_length R-hat : {round(ss['kernel_length']['r_hat'], 2)}")
print(f"kernel_var R-hat : {round(ss['kernel_var']['r_hat'],2)}")