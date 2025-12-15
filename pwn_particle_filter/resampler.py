from stonesoup.base import Property


from stonesoup.resampler.base import Resampler
import numpy as np
from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors

class GausADResampler(Resampler):
	n_samples: int = Property(default=1000,
							 doc="Number of samples to draw from the distribution.")
	def resample(self, particles):
		LVol = self.n_samples
		

		# StoneSoup stores particles as (ndim, N)
		X = np.asarray(particles.state_vector).T  # -> (N, ndim)

		# get normalised weights
		w = np.asarray(particles.weight, dtype=float)
		w = w / w.sum()

		# weighted mean/cov
		mu = (w[:, None] * X).sum(axis=0)
		Xm = X - mu
		C = (Xm.T * w) @ Xm

		# deterministic redraw
		X_new = sample_gaussian_fibonacci(mu, C, LVol, type='Fibonacci')  # (N, ndim)
		N = X_new.shape[0]

		# return a NEW ParticleState with uniform weights
		new = ParticleState(state_vector=StateVectors(X_new.T),
							timestamp=particles.timestamp)
		new.weight = np.full(N, 1.0 / N)
		return new
		





















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
