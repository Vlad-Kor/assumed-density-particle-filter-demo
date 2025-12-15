from stonesoup.types.state import State
from stonesoup.types.array import CovarianceMatrix
from stonesoup.base import Property
from stonesoup.types.prediction import Prediction
from stonesoup.predictor._utils import predict_lru_cache

from stonesoup.resampler.base import Resampler
import numpy as np
from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
from stonesoup.proposal.base import Proposal


class GausADResampler(Resampler):
	n_samples: int = Property(default=1000,
	                         doc="Number of samples to draw from the distribution.")
	def resample(self, particles):
		mu_pred = np.mean(particles, axis=0) # shape (3,)
		C_pred = np.cov(particles, rowvar=False, bias=True)

		new_particles = sample_gaussian_fibonacci(mu_pred, C_pred, self.n_samples)
		return new_particles
	
class GausADProposal(Proposal):
	n_samples: int = Property(default=1000,
	                         doc="Number of samples to draw from the distribution.")
	def rvs(self, state, *args, **kwargs):
		mu_pred = np.mean(state, axis=0) # shape (3,)
		C_pred = np.cov(state, rowvar=False, bias=True)

		samples = sample_gaussian_fibonacci(mu_pred, C_pred, self.n_samples)
		return samples


		





















# class PWNState(State):
# 	"""Gaussian State type

# 	This is a simple PWN state object, which, as the name suggests,
# 	is described by a Partially Wrapped Normal distribution.
	
# 	The PWN disribution should have the first dimension be wrapped,
# 	with the rest being unwrapped.
# 	"""
# 	covar: CovarianceMatrix = Property(doc='Covariance matrix of state.')

# 	def __init__(self, state_vector, covar, *args, **kwargs):
# 		# Don't cast away subtype of covar if not necessary
# 		if not isinstance(covar, CovarianceMatrix):
# 			covar = CovarianceMatrix(covar)
# 		super().__init__(state_vector, covar, *args, **kwargs)
# 		if self.state_vector.shape[0] != self.covar.shape[0]:
# 			raise ValueError(
# 				"state vector and covariance should have same dimensions")

# 	@property
# 	def mean(self):
# 		"""The state mean, equivalent to state vector"""
# 		return self.state_vector

# class PWNStatePrediction(Prediction, PWNState):
# 	  """ PWNStatePrediction type

# 	This is a simple GauPWNssian state prediction object, which, as the name
# 	suggests, is described by a Partially Wrapped Normal distribution.
# 	"""
