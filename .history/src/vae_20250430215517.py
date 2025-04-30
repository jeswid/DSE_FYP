import jax
import jax.numpy as jnp
from jax import lax
from jax.example_libraries import stax
import numpyro
import numpyro.distributions as dist

"""

"""

def vae_encoder(hidden_dim=50, z_dim=40):
    """
    VAE encoder network architecture
    
    Args:
        hidden_dim: Number of hidden dimensions
        z_dim: Number of latent dimensions
    
    Returns:
        Stax neural network for encoding
    """
    return stax.serial(
        #(num_samples, num_regions) -> (num_samples, hidden_dims)
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        stax.FanOut(2),
        stax.parallel(
            # mean : (num_samples, hidden_dim) -> (num_samples, z_dim)
            stax.Dense(z_dim, W_init=stax.randn()), #(5,50)
            #std : (num_samples, hidden_dim) -> (num_samples, z_dim)
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp)
        )
    )

def vae_decoder(hidden_dim, out_dim):
    """
    VAE decoder network architecture
    
    Args:
        hidden_dim: Number of hidden dimensions
        out_dim: Output dimensions (number of regions)
    
    Returns:
        Stax neural network for decoding
    """
    return stax.serial(
        # (num_samples, z_dim) -> (num_samples, hidden_dim)
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        # (num_samples, hidden_dim) -> (num_samples, num_regions)
        stax.Dense(out_dim, W_init=stax.randn())
    )

def vae_model(batch, hidden_dim, z_dim, vae_var):
    """
    VAE model (decoder portion)
    
    Args:
        batch: Input data batch
        hidden_dim: Number of hidden dimensions
        z_dim: Number of latent dimensions
        vae_var: VAE variance parameter
    
    Returns:
        Sampled observations
    """
    batch = jnp.reshape(batch, (batch.shape[0], -1)) # (num_samples, num_regions)
    batch_dim, out_dim = jnp.shape(batch)

    # vae-decoder in numpyro module
    decode = numpyro.module(
        name="decoder",
        nn=vae_decoder(hidden_dim=hidden_dim, out_dim=out_dim),
        input_shape=(batch_dim, z_dim) #(5,40)
    )

    # Sample a univariate normal
    z = numpyro.sample(
        "z",
        dist.Normal(
            jnp.zeros((batch_dim,z_dim)),
            jnp.ones((batch_dim,z_dim))
        )
    )
    # Forward pass from decoder
    gen_loc = decode(z) #(num_regions,)
    obs = numpyro.sample(
        "obs",
        dist.Normal(gen_loc, vae_var),
        obs=batch
    ) #(num_samples, num_regions)
    return obs

def vae_guide(batch, hidden_dim, z_dim):
    """
    VAE guide (encoder portion)
    
    Args:
        batch: Input data batch
        hidden_dim: Number of hidden dimensions
        z_dim: Number of latent dimensions
    
    Returns:
        Sampled latent variables
    """
    batch = jnp.reshape(batch, (batch.shape[0], -1)) #(num_samples, num_regions)
    batch_dim, input_dim = jnp.shape(batch)# num_samples , num_regions

    # vae-encoder in numpyro module
    encode = numpyro.module(
        name="encoder",
        nn=vae_encoder(hidden_dim=hidden_dim,z_dim=z_dim),
        input_shape=(batch_dim, input_dim) #(5,58)
    ) #(num_samples, num_regions) -> (num_samples, hidden_dims)

    # Sampling mu, sigma - Pretty much the forward pass
    z_loc, z_std = encode(batch) #mu : (num_samples, z_dim), sigma2 : (num_samples, z_dim)
    # Sample a value z based on mu and sigma
    z = numpyro.sample("z", dist.Normal(z_loc, z_std)) #(num_sample, z_dim)
    return z
