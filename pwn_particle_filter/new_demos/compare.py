# compare bearing only for gaussian assumed density resampler and kalman filter

from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
from datetime import timedelta

# Load Stone Soup materials
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing

# Load the filter components
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.particle import ParticlePredictor
from pwn_particle_filter.resampler import GausADResampler
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import SingleTargetTracker
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.resampler.particle import ESSResampler

# load compare stuff
from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.measures import Euclidean
from stonesoup.plotter import MetricPlotter

# set a random seed and start of the simulation
np.random.seed(2001)
start_time = datetime.now()

# %%
# 1) Create the moving platform and the Bearing-Only radar
# --------------------------------------------------------
# Firstly, we create the initial state of the platform, including the origin point and the
# cartesian (x, y) movement direction. Then, we create a transition model (in 2D cartesian coordinates)
# of the platform.
# At this point, we can set up the Radar which receives only the bearing measurements from the targets using the
# :class:`~.RadarBearing` sensor.

# Import the platform to place the sensor
from stonesoup.platform.base import MovingPlatform

# Define the platform location, place it in the origin, and define its Cartesian movements.
# In addition, specify the position and velocity mapping. This is done in 2D Cartesian coordinates.

platform_state_vector = StateVector([[0], [-0.3], [0], [0]])
position_mapping = (0, 2)
velocity_mapping = (1, 3)

# Create the initial state (position and time)
platform_state = State(platform_state_vector, start_time)

# Create a platform transition model, let's assume it is moving with constant velocity
platform_transition_model = CombinedLinearGaussianTransitionModel([
	ConstantVelocity(0.1), ConstantVelocity(0)])

# We can instantiate the platform's initial state, position and velocity mapping, and 
# the transition model using the  :class:`~.MovingPlatform` platform class.
platform = MovingPlatform(states=platform_state,
						  position_mapping=position_mapping,
						  velocity_mapping=velocity_mapping,
						  transition_model=platform_transition_model)

# At this stage, we need to create the sensor, let's import the RadarBearing. 
# This sensor only provides the bearing measurements from the target detections, 
# the range is not specified.
from stonesoup.sensor.radar.radar import RadarBearing

# Configure the radar noise, since we are using just a single dimension we need to specify only the
# noise associated with the bearing dimension, we assume a bearing accuracy of +/- 0.025 degrees for 
# each measurement
noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(0.025) ** 2])))

# This radar needs to be informed of the x and y mapping of the target space.
radar_mapping = (0, 2)

# Instantiate the radar
radar = RadarBearing(ndim_state=4,
					 position_mapping=radar_mapping,
					 noise_covar=noise_covar)

# As presented in the other examples we have to place the sensor on the platform.
platform.add_sensor(radar)
# At this point we can also check the offset rotation or the mounting of the radar in respect to the
# platform as shown in other tutorials.

# %%
# 2) Generate the ground truth target movements
# --------------------------------------------------
# We now build a ground truth simulator of a single target with a transition model
# and a known initial state.

# Load the single target ground truth simulator
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors

# Instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([
	ConstantVelocity(5.0), ConstantVelocity(5.0)])

# make prior
# Target starts near the path
prior_mu = np.array([80, -8, 12, -1])
prior_cov = np.diag([60**2, 4**2, 60**2, 4**2])
sample_size = 1000

# Sample from the prior Gaussian distribution around the true initial state
samples = sample_gaussian_fibonacci(prior_mu,
								  prior_cov,
								  sample_size,
								  type='Fibonacci')


# Create prior particle and kalman state.
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
g_prior = ParticleState(state_vector=StateVectors(samples.T),
					  weight=np.array([Probability(1/sample_size)]*sample_size),
					  timestamp=start_time)
k_prior = GaussianState(
	state_vector=StateVector(prior_mu.reshape(-1,1)),
	covar=CovarianceMatrix(prior_cov),
	timestamp=start_time)






# Set up the ground truth simulation
initial_truth = GroundTruthState(prior_mu, timestamp=start_time)

groundtruth_simulation = SingleTargetGroundTruthSimulator(
	transition_model=transition_model,
	initial_state=initial_truth,
	timestep=timedelta(seconds=2),
	number_steps=60
)

# %%
# 3) Set up the detection simulation that generates the bearing measurements
# --------------------------------------------------------------------------
# After defining the measurement model and simulation, we will use these components to run our example.
# The measurement model is the :class:`~.Cartesian2DToBearing`.

# Define the measurement model using a Cartesian to bearing
meas_model = Cartesian2DToBearing(
	ndim_state=4,
	mapping=(0, 2),
	noise_covar=noise_covar)

# Import the PlatformDetectionSimulator
from stonesoup.simulator.platform import PlatformDetectionSimulator

sim = PlatformDetectionSimulator(groundtruth=groundtruth_simulation,
								 platforms=[platform])

# %%
# 4) Set up the tracker
# ---------------------
# Instantiate the filter components
g_predictor = ParticlePredictor(transition_model)

g_updater = ParticleUpdater(measurement_model=meas_model,
						 resampler=GausADResampler(n_samples=sample_size))

iid_predictor = ParticlePredictor(transition_model)
iid_updater = ParticleUpdater(measurement_model=meas_model,
							resampler=ESSResampler())
							

k_predictor = ExtendedKalmanPredictor(transition_model)
k_updater = ExtendedKalmanUpdater(measurement_model=None)




from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

times = []
truth_path = GroundTruthPath()
platform_path = GroundTruthPath()
saved_detections = []  # list[(time, detections_set)]

for time, detections in sim:
	times.append(time)
	saved_detections.append((time, detections))

	gt = next(iter(groundtruth_simulation.current[1]))  # GroundTruthPath
	truth_path.append(gt[-1])                           # GroundTruthState at this time
	platform_path.append(GroundTruthState(platform.state_vector, timestamp=time))

g_track = Track()
k_track = Track()
iid_track = Track()
g_state = g_prior
iid_state = g_prior

for time, detections in saved_detections:
	g_prediction = g_predictor.predict(g_track[-1] if g_track else g_prior, timestamp=time)
	k_prediction = k_predictor.predict(k_track[-1] if k_track else k_prior, timestamp=time)
	iid_prediction = iid_predictor.predict(iid_track[-1] if iid_track else iid_state, timestamp=time)

	if detections:
		det = next(iter(detections))
		g_updater.measurement_model = det.measurement_model
		g_state = g_updater.update(SingleHypothesis(g_prediction, det))
		iid_updater.measurement_model = det.measurement_model
		iid_state = iid_updater.update(SingleHypothesis(iid_prediction, det))
		k_updater.measurement_model = det.measurement_model
		k_state = k_updater.update(SingleHypothesis(k_prediction, det))
	else:
		g_state = g_prediction
		k_state = k_prediction
		iid_state = iid_prediction

	g_track.append(g_state)
	k_track.append(k_state)
	iid_track.append(iid_state)



# Now compare them both

def get_mean_from_particle_track(track):
	mean_track = Track()
	
	for state in track:
		weights = state.weight
		state_vectors = state.state_vector.data  # shape (num_particles, dim)
		
		w = weights / np.sum(weights)
		mean_vector = np.sum(state_vectors * w, axis=1, keepdims=True)

		mean_track.append(State(
			state_vector=StateVector(mean_vector),
			timestamp=state.timestamp
		))
	return mean_track

def gaussian_track_to_mean_state_track(track): # when kalman fails cov can have complex numbers
    out = Track()
    for s in track:
        out.append(State(state_vector=StateVector(np.asarray(s.state_vector)),
                         timestamp=s.timestamp))
    return out

k_mean_track = gaussian_track_to_mean_state_track(k_track)
g_mean_track = get_mean_from_particle_track(g_track)
iid_mean_track = get_mean_from_particle_track(iid_track)



# OSPA parameters
c = 1000   # cutoff / cardinality penalty scale
p = 1
pos_measure = Euclidean((0, 2))  # OSPA in x/y only

ospa_pf_fib = OSPAMetric(c=c, p=p, measure=pos_measure,
						   generator_name="OSPA_PF_FIB",
						   tracks_key="PF_tracks", truths_key="truths")

ospa_ekf = OSPAMetric(c=c, p=p, measure=pos_measure,
							generator_name="OSPA_EKF",
							tracks_key="EKF_tracks", truths_key="truths")

ospa_iid = OSPAMetric(c=c, p=p, measure=pos_measure,
							generator_name="OSPA_PF_IID",
							tracks_key="PF_IID_tracks", truths_key="truths")

metricmanager = MultiManager([ospa_pf_fib, ospa_ekf, ospa_iid])
metricmanager.add_data(
	{
		"PF_tracks": g_mean_track,
		"EKF_tracks": k_mean_track,
		"PF_IID_tracks": iid_mean_track,
		"truths": truth_path
	}
)
metrics = metricmanager.generate_metrics()

from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(times, tail_length=0.3)
plotter.plot_ground_truths(truth_path, [0, 2])
plotter.plot_ground_truths(platform_path, [0, 2], label="Sensor platform")
plotter.plot_tracks(g_track, [0, 2], particle=True, plot_history=False, label="Particle Filter with Deterministic Samples")
plotter.plot_tracks(k_mean_track, [0, 2], particle=False, plot_history=False, label="Extended Kalman Filter", line=dict(color='green'), marker=dict(color='green'))
plotter.plot_tracks(iid_track, [0, 2], particle=True, plot_history=False, label="Particle Filter with Random Samples", line=dict(color='red'), marker=dict(color='red'))
plotter.fig

# # %%
# open browser for non interactive view
if __name__ == "__main__":
	try:
		plotter.fig.write_html("particle_filter.html", auto_open=True)

		graph = MetricPlotter()
		graph.plot_metrics(metrics, generator_names=["OSPA_PF_FIB", "OSPA_EKF", "OSPA_PF_IID"])
		plt.show()

	except Exception as e:
		print("could not write html")
		raise e
