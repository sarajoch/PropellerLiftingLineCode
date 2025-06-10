import numpy as np
import matplotlib.pyplot as plt
import readData as data
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import InductionFactors as induction
import SingularIntegration as singular

Rn = 9.78 * 10**7
D = 4.65
R = D/2
z = 4
rho = 1025
nu = 1.08 * 10**(-6)
P_vapor = 2160
P_atm = 101325
PD07 = 1.1
EAR = 0.65

J_values = np.array([ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 ])
Cl_values = np.array([0.2263, 0.2257,0.204,0.205,0.2237,0.2539,0.2808,0.293,0.2904, 0.2476, 0.184, 0.1267])
rR, cD, tD, PD = data.geometryData()
J_ow, Ktexp, Kqexp, etaexp = data.OpenwaterData()
directory = "Data" 
cl_database = data.create_cl_database(directory)
r, c, t, P = rR * R, cD * D, tD * D, PD * D
num_J = len(J_values)
num_r = len(r)
index_07 = np.where(rR == 0.7)[0][0]  # Find index where rR is 0.7
c07 = c[index_07]
n = np.array([(Rn * nu) / (c07 * D * np.sqrt(J**2 + (0.7 * np.pi)**2)) for J in J_values])
V = J_values * n * D

def plotCl(rR, Cl_values):
    plt.plot(rR[:-1], Cl_values)
    plt.xlabel('r/R')
    plt.ylabel('Cl')
    plt.title('2D Lift Coefficient Distribution')
    plt.grid()
    plt.show()

def plotOpenWaterData(J, KT, KQ, eta, Jexp , Ktexp , Kqexp , etaexp):
    plt.figure(figsize=(7,5))
    plt.plot(J, KT, label=r'$K_T$')
    plt.plot(Jexp, Ktexp, '--', label=r'$K_T$ open water test')
    plt.plot(J, KQ, label=r'$K_Q$')
    plt.plot(Jexp, Kqexp, '--', label=r'$K_Q$ open water test')
    plt.plot(J, eta, label=r'$\eta$')
    plt.plot(Jexp, etaexp, '--', label=r'$\eta$ open water test')
    plt.xlabel('J')
    plt.ylabel(r'$K_T$, $K_Q$ & $\eta$')
    plt.grid()
    plt.legend()
    plt.show()

def computeUtUa(r,V,n,gamma) :
    Ut = (z * gamma) / (2*np.pi*r)
    Ua = -V + np.sqrt(V**2 + 2 * Ut * ( 2* np.pi * r * n - Ut/2 ))
    return Ut, Ua

def computeV0n(nu,RN,J_values,DP,c07) :
    n = np.array([(RN * nu) / (c07 * DP * np.sqrt(J**2 + (0.7 * np.pi)**2)) for J in J_values])
    V = J_values * n * DP
    return V, n

def computeBetaiVinfAlpha(r,V0,n,Ut,Ua, P) :
    beta_i = np.arctan((V0 + Ua * 0.5)/(2*np.pi*r*n - 0.5 * Ut))
    V_inf = np.sqrt((V0 + 0.5 * Ua)**2 + (2*np.pi*r*n - 0.5 * Ut)**2)
    phi = np.arctan(P/(2*np.pi*r))
    alpha = phi - beta_i
    return beta_i, V_inf, alpha

def interpolate_cl(database, rR_query, alpha_query):
    """Interpolates CL value for a given alpha at a specific r/R."""
    available_rR = sorted(database.keys())  # Sorted r/R values
    cl_values_at_rR = []

    for rR in available_rR:
        data = database[rR]
        alphas, CLs = data[:, 0], data[:, 1]

        cl_interp = interp1d(alphas, CLs, kind='linear', fill_value='extrapolate')
        cl_values_at_rR.append(cl_interp(alpha_query))

    cl_rR_interp = interp1d(available_rR, cl_values_at_rR, kind='linear', fill_value='extrapolate')
    return cl_rR_interp(rR_query)

def interpolateSpan(N, r, c, t, P):
    r_new = np.linspace(r[0], r[-1], N)
    interp_c = interp1d(r, c, kind='cubic', fill_value="extrapolate")
    interp_t = interp1d(r, t, kind='cubic', fill_value="extrapolate")
    interp_P = interp1d(r, P, kind='cubic', fill_value="extrapolate")
    c_n = interp_c(r_new)
    t_n = interp_t(r_new)
    P_n = interp_P(r_new)

    return r_new, c_n, t_n, P_n

def llcWithoutInducedVelocity():
    KT = []
    KQ = []
    eta = []
    beta_i = np.zeros((num_J, num_r))
    gamma0 = np.zeros((num_J, num_r))
    for j in range(len(J_values)):
        dT = []
        dQ = []
        n = (Rn * nu) / (c07 * D * np.sqrt(J_values[j]**2 + (0.7 * np.pi)**2))
        V = J_values[j] * n * D
        for i in range(len(r)):
            beta = np.arctan(V/(2*np.pi*r[i]*n))
            beta_i[j][i] = beta
            phi = np.arctan(P[i]/(2*np.pi*r[i]))
            alpha = phi - beta
            #CL = Cl_values[i] + 2 * np.pi * alpha
            CL = interpolate_cl(cl_database, r[i]/R, alpha)
            Vinf = np.sqrt(V**2 + (2*np.pi*r[i]*n)**2)
            gamma = ( CL * Vinf * c[i] )/2
            if (i ==0) or i == (len(r) - 1):
                gamma = 0
            gamma0[j][i] = gamma
            Rn_inf = (Vinf * c[i])/nu
            Cf = 0.075/(np.log10(Rn_inf)-2)**2 if Rn_inf > 0 else 0
            Cd = 2 * Cf * (1 + 2 * (t[i]/c[i]) + 60 * (t[i]/c[i])**4) * ( 1 + (CL**2)/8) if c[i] > 0 else 0
            dD = (rho/2) * Vinf**2 * c[i] * Cd if c[i] > 0 else 0
            dT.append(rho * gamma * (2 * np.pi * r[i] * n) - dD * np.sin(beta))
            dQ.append((rho * gamma * V + dD * np.cos(beta)) * r[i])

        T = z * np.trapz(np.array(dT), r, None)
        Q = z * np.trapz(np.array(dQ), r, None)
        thrustCoeff = T/(rho * n**2 * D**4)
        torqueCoeff = Q/(rho * n**2 * D**5)
        KT.append(thrustCoeff)
        KQ.append(torqueCoeff * 10)
        eta.append((J_values[j]*thrustCoeff)/(2*np.pi*torqueCoeff))

    return KT, KQ, eta, beta_i

def simpleInducedVel(tol, max_iter, r):
    gamma = np.zeros((num_J, num_r))

    KT = []
    KQ = []
    eta = []
    beta_i = np.zeros((num_J, num_r))

    for j in range(num_J):
        n = (Rn * nu) / (c07 * D * np.sqrt(J_values[j]**2 + (0.7 * np.pi)**2))
        V = J_values[j] * n * D
    
        dT = []
        dQ = []

        Ut_all = np.zeros(num_r)
        Ua_all = np.zeros(num_r)
        beta_all = np.zeros(num_r) 
        Vinf_all = np.zeros(num_r)
        alpha_all = np.zeros(num_r)
        CL_all = np.zeros(num_r)

        diff = 1.0
        iter = 0

        gamma_J = np.ones(num_r) * 0.1
        gamma_new = np.zeros(num_r)

        while diff > tol and iter < max_iter:

            for i in range(num_r):
                Ut, Ua = computeUtUa(r[i], V, n, gamma_J[i])
                beta, Vinf, alpha = computeBetaiVinfAlpha(r[i], V, n, Ut, Ua, P[i])
                phi = np.arctan(P[i]/(2*np.pi*r[i]))
                alpha = phi - beta
                CL = interpolate_cl(cl_database, r[i]/R, alpha)
                gamma_new[i] = Vinf * c[i] * CL / 2 if i != 0 or i != (len(r) - 1) else 0
                gamma_new[i] = 0 if i == 0 or i == (len(r) - 1) else gamma_new[i]
                Ut_all[i] = Ut
                Ua_all[i] = Ua
                beta_all[i] = beta
                Vinf_all[i] = Vinf
                alpha_all[i] = alpha
                CL_all[i] = CL
                beta_i[j][i] = beta

            # Check difference in gamma (excluding root and tip), reset gamma as needed
            diff = np.linalg.norm(gamma_new[1:-1] - gamma_J[1:-1]) / np.mean(gamma_J[1:-1])
            if diff >= tol:
                gamma_J += 0.1*(gamma_new - gamma_J)
                iter += 1
            else :
                gamma[j,:] = gamma_new
                beta_i[j] = beta_all

        for k in range(num_r):
            Rn_inf = (Vinf_all[k] * c[k])/nu
            Cf = 0.075/(np.log10(Rn_inf)-2)**2 if Rn_inf > 0 else 0
            Cd = 2 * Cf * (1 + 2 * (t[k]/c[k]) + 60 * (t[k]/c[k])**4) * ( 1 + ((CL_all[k]**2)/8)) if c[k] > 0 else 0
            dD = (rho/2) * Vinf_all[k]**2 * c[k] * Cd if c[k] > 0 else 0
            dT.append(rho * gamma_new[k] * (2 * np.pi * r[k] * n - 0.5 * Ut_all[k]) - dD * np.sin(beta_all[k]))
            dQ.append((rho * gamma_new[k] * (V + Ua_all[k]) + dD * np.cos(beta_all[k])) * r[k])

        T = z * trapezoid(np.array(dT), r)
        Q = z * trapezoid(np.array(dQ), r)
        thrustCoeff = T/(rho * n**2 * D**4)
        torqueCoeff = Q/(rho * n**2 * D**5)
        KT.append(thrustCoeff)
        KQ.append(torqueCoeff * 10)
        eta.append((J_values[j] / (2*np.pi) ) * (thrustCoeff / torqueCoeff))
    
    return KT, KQ, eta, beta_i, gamma

    
def llcWithInductionFactor(beta_i, tol, max_iter, r_int, c_int, t_int, P_int, r):
    g = np.zeros((num_J, len(r_int)))

    KT = []
    KQ = []
    eta = []

    for j in range(len(J_values)):
        dT = []
        dQ = []

        interp_beta = interp1d(r, beta_i[j], kind='cubic', fill_value="extrapolate")
        beta_i_j = interp_beta(r_int)
        n = (Rn * nu) / (c07 * D * np.sqrt(J_values[j]**2 + (0.7 * np.pi)**2))
        V = J_values[j] * n * D
        i_a = np.zeros(len(r_int))
        i_t = np.zeros(len(r_int))
        Ua = np.zeros(len(r_int))
        Ut = np.zeros(len(r_int))
        beta_all = np.zeros(len(r_int))
        Vinf_all = np.zeros(len(r_int))
        alpha_all = np.zeros(len(r_int))
        CL_all = np.zeros(len(r_int))
        gamma_J = np.ones(len(r_int)) * 10
        gamma_J[0], gamma_J[-1] = 0, 0
        gamma_new = np.zeros(len(r_int))

        diff = 1.0
        iter = 0
        while diff > tol and iter < max_iter:

            dgamma = np.gradient(gamma_J, r_int)

            for i in range(1, len(r_int)-1):
                for k in range(len(r_int)):
                    i_a[k], i_t[k] = induction.inductionFactors(r_int[k], r_int[i], np.abs(beta_i_j[k]), z)

                Ua[i] = ( 1 / (2 * np.pi) ) * singular.singularIntegration(r_int, i_a * dgamma, r_int[i])
                Ut[i] = ( 1 / (2 * np.pi) ) * singular.singularIntegration(r_int, i_t * dgamma, r_int[i])
            
            for l in range(len(r_int)):
                beta_i_j[l], Vinf_all[l], alpha_all[l] = computeBetaiVinfAlpha(r_int[l], V, n, Ut[l], Ua[l], P_int[l])
                CL = interpolate_cl(cl_database, r_int[l]/R, alpha_all[l])
                CL_all[l] = CL
                gamma_new[l] = Vinf_all[l] * c_int[l] * CL / 2 if l != 0 or l != (len(r_int) - 1) else 0
                gamma_new[l] = 0 if l == 0 or l == (len(r_int) - 1) else gamma_new[l]

            diff = np.linalg.norm(gamma_new[1:-1] - gamma_J[1:-1]) / np.mean(gamma_J[1:-1])
            if diff >= tol:
                gamma_J += 0.05*(gamma_new - gamma_J) 
                iter += 1
            else:
                g[j,:] = gamma_new

        for m in range(len(r_int)):
            Rn_inf = (Vinf_all[m] * c_int[m])/nu
            Cf = 0.075/(np.log10(Rn_inf)-2)**2 if Rn_inf > 0 else 0
            Cd = 2 * Cf * (1 + 2 * (t_int[m]/c_int[m]) + 60 * (t_int[m]/c_int[m])**4) * ( 1 + ((CL_all[m]**2)/8)) if c_int[m] > 0 else 0
            #Cd = 2 * Cf * (1 + 2 * (t_int[m]/c_int[m])) if c_int[m] > 0 else 0
            dD = (rho/2) * Vinf_all[m]**2 * c_int[m] * Cd if c_int[m] > 0 else 0
            dT.append(rho * gamma_new[m] * (2 * np.pi * r_int[m] * n - 0.5 * Ut[m]) - dD * np.sin(beta_i_j[m]))
            dQ.append((rho * gamma_new[m] * ( V + 0.5 * Ua[m] ) + dD * np.cos(beta_i_j[m])) * r_int[m])

        T = z * np.trapz(np.array(dT), r_int, None)
        Q = z * np.trapz(np.array(dQ), r_int, None)
        thrustCoeff = T/(rho * n**2 * D**4)
        torqueCoeff = Q/(rho * n**2 * D**5)
        KT.append(thrustCoeff)
        KQ.append(torqueCoeff * 10)
        eta.append((J_values[j]*thrustCoeff)/(2*np.pi*torqueCoeff))

    return KT, KQ, eta



r_N, c_N, t_N, P_N = interpolateSpan(37, r, c, t, P)


# TASK 3.2 LLC without induced velocity
KT_temp, KQ_temp, eta_temp, beta_temp = llcWithoutInducedVelocity()
plotOpenWaterData(J_values, KT_temp, KQ_temp, eta_temp, J_ow, Ktexp, Kqexp, etaexp)


#TASK 3.3 LLC with a simple model of induced velocities
KT3, KQ3, eta3, beta3, gamma3 = simpleInducedVel(10e-2, 1000, r)
plotOpenWaterData(J_values, KT3, KQ3, eta3, J_ow, Ktexp, Kqexp, etaexp)

#Task 3.4 LLC with a more accurate model of induced velocities
KT, KQ, eta = llcWithInductionFactor(beta_temp, 10e-5, 1000, r_N, c_N, t_N, P_N, r)
plotOpenWaterData(J_values, KT, KQ, eta, J_ow, Ktexp, Kqexp, etaexp)




            

