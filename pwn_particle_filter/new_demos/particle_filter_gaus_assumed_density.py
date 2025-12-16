#!/usr/bin/env python
import numpy as np

from datetime import datetime
from datetime import timedelta
start_time = datetime.now().replace(microsecond=0)

# %%

np.random.seed(1991)

# %%
# Initialise Stone Soup ground-truth and transition models.
from scipy.special import logsumexp
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
	ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
														  ConstantVelocity(0.05)])
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

# %%
# Create the truth path
for k in range(1, 21):
	timesteps.append(start_time+timedelta(seconds=k))
	truth.append(GroundTruthState(
		transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
		timestamp=timesteps[k]))

# %%
# Plot the ground truth.

from stonesoup.plotter import AnimatedPlotterly, AnimationPlotter

plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
#plotter = AnimationPlotter(legend_kwargs=dict(loc='upper left'))
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig


# %%
# Initialise the bearing, range sensor using the appropriate measurement model.
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.detection import Detection

sensor_x = 50
sensor_y = 0

measurement_model = CartesianToBearingRange(
	ndim_state=4,
	mapping=(0, 2),
	noise_covar=np.diag([np.radians(0.2), 1]),
	translation_offset=np.array([[sensor_x], [sensor_y]])
)

# %%
# Populate the measurement array
measurements = []
for state in truth:
	measurement = measurement_model.function(state, noise=True)
	measurements.append(Detection(measurement, timestamp=state.timestamp,
								  measurement_model=measurement_model))

# %%
# Plot those measurements

plotter.plot_measurements(measurements, [0, 2])
plotter.fig


# %%
# Set up the particle filter

from stonesoup.predictor.particle import ParticlePredictor
predictor = ParticlePredictor(transition_model)
from stonesoup.resampler.particle import ESSResampler
from stonesoup.resampler.base import Resampler
from stonesoup.base import Property


LVol = 1000
from pwn_particle_filter.resampler import GausADResampler
resampler = GausADResampler(LVol)
from stonesoup.updater.particle import ParticleUpdater
updater = ParticleUpdater(measurement_model, resampler)

# %%
# Initialise a prior
# ^^^^^^^^^^^^^^^^^^
# To start we create a prior estimate. This is a :class:`~.ParticleState` which describes
# the state as a distribution of particles using :class:`~.StateVectors` and weights.
# This is sampled from the Gaussian distribution (using the same parameters we
# had in the previous examples).

from scipy.stats import multivariate_normal

from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors


from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
# Sample from the prior Gaussian distribution
samples = sample_gaussian_fibonacci(np.array([0, 1, 0, 1]),
								  np.diag([1.5, 0.5, 1.5, 0.5]),
								  LVol,
								  type='Fibonacci')

number_particles = samples.shape[0]

# Create prior particle state.
prior = ParticleState(state_vector=StateVectors(samples.T),
					  weight=np.array([Probability(1/number_particles)]*number_particles),
					  timestamp=start_time)
# %%
# Run the tracker
# ^^^^^^^^^^^^^^^
# We now run the predict and update steps, propagating the collection of particles and resampling
# when told to (at every step).
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
	prediction = predictor.predict(prior, timestamp=measurement.timestamp)
	hypothesis = SingleHypothesis(prediction, measurement)
	post = updater.update(hypothesis)
	track.append(post)
	prior = track[-1]
plotter.plot_tracks(track, [0, 2], particle=True, plot_history=False)
plotter.fig

# %%
# open browser for non interactive view
try:
	plotter.fig.write_html("particle_filter.html", auto_open=True)
except Exception:
	print("could not write html")