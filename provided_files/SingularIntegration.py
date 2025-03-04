import numpy as np

def singularIntegration(x, f, x0):
	''' 
	This function computes the following integral: J = int f(x)/(x0 - x) dx using singularity subtraction. 
	'''

	n = len(x)
	f0 = np.interp(x0, x, f)

	integrand_num = np.zeros(n)

	for i in range(n):
		if x[i] - x0 != 0:
			integrand_num[i] = f[i]/(x0 - x[i]) - f0/(x0 - x[i])

	J = np.trapz(integrand_num, x=x)

	J += f0*(-np.log(x[-1] - x0) + np.log(x0 - x[0]))

	return J
