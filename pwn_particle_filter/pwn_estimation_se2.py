import numpy as np
from scipy.stats import norm

def generate_pwn_sample_random(mu, cov, num_samples):
	samples = np.random.multivariate_normal(mu, cov, num_samples)
	samples[:,0] = samples[:,0] % (2 * np.pi)
	return samples


def estimate_pwn_parameters_d3(samples):
	# see https://isas.iar.kit.edu/pdf/MFI2014_Kurz-PWN.pdf
	# calculate hybrid moments

	s = samples

	print(s.shape) # shape (N,3)

	s_tilde = np.column_stack([
		np.cos(s[:,0]),
		np.sin(s[:,0]),
		s[:,1],
		s[:,2],
	]) # shape (N,4)
	print(s_tilde.shape)

	mu_tilde = np.mean(s_tilde, axis=0) # shape (4,)
	C_tilde = np.cov(s_tilde, rowvar=False, bias=True) # shape (4,4)

	mu = np.array([np.arctan2(mu_tilde[1], mu_tilde[0]), mu_tilde[2], mu_tilde[3]])

	# calculate covariance
	c_11 = -2 * np.log(np.sqrt(mu_tilde[0]**2 + mu_tilde[1]**2))
	

	p = - C_tilde[0,2] * np.exp(c_11 / 2)
	q = C_tilde[1,2] * np.exp(c_11 / 2)

	c_12 = p * np.sin(mu[0]) + q * np.cos(mu[0])
	

	p_tilde = - C_tilde[0,3] * np.exp(c_11 / 2)
	q_tilde = C_tilde[1,3] * np.exp(c_11 / 2)
	c_13 = p_tilde * np.sin(mu[0]) + q_tilde * np.cos(mu[0])

	C = np.zeros((3,3))
	C[0,0] = c_11

	C[0,1] = c_12
	C[1,0] = c_12

	C[0,2] = c_13
	C[2,0] = c_13

	C[1:,1:] = C_tilde[2:,2:]
	return mu, C


def _test_pwn_estimation_d3():
	mean = np.array([1*np.pi, 1*np.pi, 1*np.pi])
	Cov = np.array([
    [1.0, 0.3, -0.2],
    [0.3, 2.0, 0.4],
    [-0.2, 0.4, 0.5],
	])
	samples = generate_pwn_sample_random(mean, Cov, 100000)
	mu_est, C_est = estimate_pwn_parameters_d3(samples)
	np.set_printoptions(suppress=True) # disable scientific notation
	print(f"mu_est: {mu_est} vs mu: {mean}")
	print(f"C_est:\n{C_est}\nvs C:\n{Cov}")


if __name__ == "__main__":
	_test_pwn_estimation_d3()