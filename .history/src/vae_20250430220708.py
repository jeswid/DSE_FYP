import jax
import jax.numpy as jnp
from jax import lax
from jax.example_libraries import stax
import numpyro
import numpyro.distributions as dist

"""
This script implements the VAE encoder, decoder, and full VAE architecture 
following the approach outlined by Semenova et al. (2023). The model is 
based on a probabilistic VAE framework using NumPyro and Stax for neural 
network components. The original implementation can be found on Semenova 
et al.’s GitHub repository, as referenced in their publication.

The architecture consists of:
- An encoder network that maps high-dimensional input to a latent space
- A decoder network that reconstructs the input from the latent space
- A NumPyro probabilistic model and guide function for variational inference

"""

def vae_encoder(hidden_dim=50, z_dim=40):
    """
    Constructs the encoder neural network for the VAE using Stax.
    Encodes input features into latent variables (mean and standard deviation).

    Args:
        hidden_dim: Number of hidden units in the encoder
        z_dim: Dimensionality of the latent variable z

    Returns:
        A stax model representing the encoder, which outputs (z_mu, z_sigma)
    """
    return stax.serial(
        # Transform input: (num_samples, num_regions) -> (num_samples, hidden_dim)
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        stax.FanOut(2), # Split into two parallel branches
        stax.parallel(
            # Output latent mean: (num_samples, hidden_dim) -> (num_samples, z_dim)
            stax.Dense(z_dim, W_init=stax.randn()), #(5,50)
            # Output latent stddev: same as above, with positive constraint via Exp
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp)
        )
    )

def vae_decoder(hidden_dim, out_dim):
    """
    Constructs the decoder neural network for the VAE using Stax.
    Reconstructs the observed data from latent representations.

    Args:
        hidden_dim: Number of hidden units in the decoder
        out_dim: Dimensionality of the output (number of regions)

    Returns:
        A stax model representing the decoder
    """
    return stax.serial(
        # Latent input: (num_samples, z_dim) -> (num_samples, hidden_dim)
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        # Output reconstruction: (num_samples, hidden_dim) -> (num_samples, out_dim)
        stax.Dense(out_dim, W_init=stax.randn())
    )

def vae_model(batch, hidden_dim, z_dim, vae_var):
    """
     VAE probabilistic model. Defines the generative process of the data 
    using the decoder. Assumes standard Normal prior over latent variables.

    Args:
        batch: Observed data batch (num_samples x num_regions)
        hidden_dim: Number of hidden units in decoder
        z_dim: Latent space dimensionality
        vae_var: Observation noise variance (assumed constant)

    Returns:
        Sampled observations from the decoder
    """
    batch = jnp.reshape(batch, (batch.shape[0], -1)) # Flatten to 2D
    batch_dim, out_dim = jnp.shape(batch)

    # Load decoder neural network
    decode = numpyro.module(
        name="decoder",
        nn=vae_decoder(hidden_dim=hidden_dim, out_dim=out_dim),
        input_shape=(batch_dim, z_dim) #(5,40)
    )

    # Sample latent variable z ~ N(0, I)
    z = numpyro.sample(
        "z",
        dist.Normal(
            jnp.zeros((batch_dim,z_dim)),
            jnp.ones((batch_dim,z_dim))
        )
    )
     # Decode to generate observation means
    gen_loc = decode(z) #(num_regions,)
    obs = numpyro.sample(
        "obs",
        dist.Normal(gen_loc, vae_var),
        obs=batch
    ) #(num_samples, num_regions)
    return obs

def vae_guide(batch, hidden_dim, z_dim):
    """
     VAE guide function (inference network). Encodes data into latent space
    and defines the variational distribution over z.

    Args:
        batch: Observed data batch (num_samples x num_regions)
        hidden_dim: Number of hidden units in encoder
        z_dim: Latent space dimensionality

    Returns:
        Sampled latent variable z ~ q(z|x)
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
