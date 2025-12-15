import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def get_fib_grid(dim, sample_count):
	match dim:
		case 2:
			indices = np.arange(0, sample_count)
			indices_p1 = np.arange(0, sample_count + 1)
			gol = (1+5**0.5)/2
			
			# centered rank-1 lattice generator
			equidistant_generator = (2 * indices + 1) / (2 * sample_count)
			
			t = equidistant_generator
			p = (indices_p1 / gol) % 1
			p = p[1:]


			fib_grid = np.column_stack((t , p))
			return fib_grid

		
		case _:
			raise NotImplementedError()
		
def transform_grid_gaussian(grid, mu, cov):
		eps = 1e-9
		grid = np.clip(grid, eps, 1 - eps) # avoid inf in ppf

		gaus = norm.ppf(grid)

		var = np.mean(gaus**2, axis=0)

		gaus = gaus / np.sqrt(var)

		# scale with eigen decomposition
		ew, V = np.linalg.eig(cov)

		D = np.diag(np.sqrt(ew))	

		gaus = gaus.T	# (2,L)

		gaus = V @ D @ gaus # (2,2) @ (2,2) @ (2,L) -> (2,L)

		gaus = gaus.T # (L,2)

		gaus[:,0] += mu[0]
		gaus[:,1] += mu[1]

		return gaus

def generate_pwn_samples(mu, cov, num_samples, dim=2):
	grid = get_fib_grid(dim, num_samples)

	gaus_grid = transform_grid_gaussian(grid, mu, cov)

	# wrapp
	gaus_grid[:,0] = gaus_grid[:,0] % (2 * np.pi)

	return gaus_grid



def estimate_pwn_parameters_d2(samples):
	# see https://isas.iar.kit.edu/pdf/MFI2014_Kurz-PWN.pdf
	# calculate hybrid moments

	s = samples

	print(s.shape) # shape (N,2)

	s_tilde = np.array([
		np.cos(s[:,0]),
		np.sin(s[:,0]),
		s[:,1]
	]) # shape (3,N)
	s_tilde = s_tilde.T # shape (N,3)
	print(s_tilde.shape)

	mu_tilde = np.mean(s_tilde, axis=0) # shape (3,)
	C_tilde = np.cov(s_tilde, rowvar=False, bias=True) # shape (3,3)

	mu = np.array([np.arctan2(mu_tilde[1], mu_tilde[0]), mu_tilde[2]])

	# calculate covariance
	c_11 = -2 * np.log(np.sqrt(mu_tilde[0]**2 + mu_tilde[1]**2))
	

	p = - C_tilde[0,2] * np.exp(c_11 / 2)
	q = C_tilde[1,2] * np.exp(c_11 / 2)

	c_12 = p * np.sin(mu[0]) + q * np.cos(mu[0])

	c_22 = C_tilde[2,2]

	C = np.array([
		[c_11, c_12],
		[c_12, c_22]
	])
	#print("C_est:", C)
	return mu, C




# test functions
def plot_basic_pwn_samples():
	# test
	mean_x = 1*np.pi
	mean_y = 1*np.pi
	sigma_x = 2
	sigma_y = 1
	correlation = 0.5

	Cov = np.array([
		[sigma_x**2, correlation * sigma_x * sigma_y],
		[correlation * sigma_x * sigma_y, sigma_y**2]
	])

	mean = np.array([mean_x, mean_y])

	samples = generate_pwn_samples(mean, Cov, 10000)

	plt.scatter(samples[:,0], samples[:,1], s=5)
	plt.show()


def test_pwn_estimation_d2():
	# generate samples
	mean_x = 1*np.pi
	mean_y = 1*np.pi
	sigma_x = 2
	sigma_y = 1
	correlation = 0.5

	Cov = np.array([
		[sigma_x**2, correlation * sigma_x * sigma_y],
		[correlation * sigma_x * sigma_y, sigma_y**2]
	])

	mean = np.array([mean_x, mean_y])

	samples = generate_pwn_samples(mean, Cov, 10000)

	# estimate parameters
	mu_est, C_est = estimate_pwn_parameters_d2(samples)
	np.set_printoptions(suppress=True) # disable scientific notation
	print(f"mu_est: {mu_est} vs mu: {mean}")
	print(f"C_est:\n{C_est}\nvs C:\n{Cov}")




if __name__ == "__main__":
	#plot_basic_pwn_samples()
	test_pwn_estimation_d2()
