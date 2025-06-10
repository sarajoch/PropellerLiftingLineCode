import numpy as np

# You can define the formula you will use here with appropriate inputs and returns
def computeUtUa(r,V,n,gamma) :
    Ut = gamma / (2*np.pi*r)
    Ua = -V + np.sqrt(V**2 + 2 * Ut * ( 2* np.pi * r * n - Ut/2 ))
    return Ut, Ua

def computeV0n(nu,RN,J_values,DP,c07) :
    n = np.array([(RN * nu) / (c07 * DP * np.sqrt(J**2 + (0.7 * np.pi)**2)) for J in J_values])
    V = J * n * DP
    return V, n

def computeBetaiVinf(r,V0,n,Ut,Ua) :
    beta_i = np.arctan((V0 + Ua * 0.5)/(2*np.pi*r*n - 0.5 * Ut))
    V_inf = np.sqrt((V0 + 0.5 * Ua)**2 + (2*np.pi*r*n - 0.5 * Ut)**2)
    return beta_i, V_inf
    
def computePitchAngle(P, r) :
    return np.arctan(P/(2*np.pi*r))

def computeAngleAttack(phi, beta_i) :
    return phi - beta_i

def interpolateCL() :
    a = 1
    
def computeGammaTot() :
    a = 1

rho = 1025.0
nu = 1.08e-6
RN = 9.78e7
DP = 4.65
c07 = 1.6182
NPB = 3

rR = np.array([0.167, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8 , 0.9, 1.0]) # Array of non-dimensional radii of each blade section
r = rR * DP/2. # dimensional radius
cD = np.array([0.257, 0.27, 0.306, 0.321, 0.333, 0.343, 0.35, 0.354, 0.355, 0.348, 0.32, 0.257, 0.0]) # Chord over diameter ratio for each blade section
chord = cD * DP # dimensional chord
PD = np.array([0.88, 0.904, 0.976, 1.021, 1.045, 1.028, 1.014, 1.046, 1.1, 1.167, 1.177, 1.148, 1.1]) # Pitch over diameter ratio for each blade section
pitch = PD * DP # dimensional pitch
J = np.array([ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 ]) # Advance ratios to calculate at

num_J = len(J)
num_rR = len(rR)

num_AoA = len(np.arange(-8,8,0.5)) # The number of angles of attack tested in xfoil
# The database of xfoil results you have collected, from xfoil results at many angles of attack for each foil section
alpha_database = np.array((num_AoA,num_rR)) 
CL_database = np.array((num_AoA,num_rR)) 
## NOTE - if you have issues with convergence, you can try interpolating this database so you have more values of rR or update xfoil results with more angles of attack

V0, n = computeV0n(nu,RN,J,DP,c07) # compute the propeller advance velocity (called V0 here) and propeller rotational speed for every J (uses vectorized calculation)

# Create matrices to store all the converged values, you will use this data to calculate sectional forces once the circulation is converged
Ut_all = np.zeros((num_rR, num_J))
Ua_all = np.zeros((num_rR, num_J))
beta_all = np.zeros((num_rR, num_J)) # You can experiment with starting from another value of beta, such ones or a solution without induced velocities
Vinf_all = np.zeros((num_rR, num_J))
alpha_all = np.zeros((num_rR, num_J))
CL_all = np.zeros((num_rR, num_J))
# Setup initial guess for gamma
gamma_all = 0.1 * np.ones((num_rR, num_J))
gamma_all[0,:] = 0.0 # gamma at root and tip equals zero
gamma_all[-1,:] = 0.0

# Loop over each speed
for i in range(num_J):
    # Parameters for iteration
    max_iter = 1000
    tol = 1e-6
    relax = 0.1  # under-relaxation factor, 0.1 should work well but you can adjust if needed
    k = 0
    rel_diff = 1.

    gamma = gamma_all[:,i]
    gamma_new = gamma_all[:,i]
    # Iteration to converge gamma distribution
    while (k < max_iter) and (rel_diff > tol):  
        # Loop over each blade section (within while loop, because we want to converge all values of gamma)
        for j in range(num_rR):
            # The calculations to repeat every iteration
            # Note that many of the inputs are scalar, since we have two for-loops (one for speeds, one for blade sections)
            Ut, Ua = computeUtUa(r,V0[i],n[i],gamma) # Replace this with the correct induced velocities for each question 
            beta, Vinf = computeBetaiVinf(r,V0[i],n[i],Ut,Ua) 
            phi = computePitchAngle(pitch[j],r[j])
            alpha = phi - beta
            CL = interpolateCL(alpha_database,CL_database,alpha)
            gamma_new[j] = computeGammaTot(chord[j],CL,Vinf)
        
        # Check difference in gamma (excluding root and tip), reset gamma as needed
        rel_diff = np.linalg.norm(gamma_new[1:-1,i] - gamma[1:-1,i]) / np.mean(gamma[1:-1,i])
        if rel_diff >= tol:
            gamma += relax*(gamma_new - gamma) # Here we use an under-relaxation term to avoid 'jumping' between circulation distributions.
            k += 1
        else :
            gamma = gamma_new

        # Store data after convergence to use for calculating forces
        Ut_all[i,j] = Ut
        Ua_all[i,j] = Ua
        beta_all[i,j] = beta
        Vinf_all[i,j] = Vinf
        alpha_all[i,j] = alpha
        CL_all[i,j] = CL
        gamma_all[i,j] = gamma

## Now loop through the converged values you found in the previous loop, and calculate forces on the blades
## Integrating these forces along the blade and summing up for the number of blades gets you total forces at each speed
## Now you can find KT, KQ, and eta to plot on the open-water diagram
