import numpy as np
import matplotlib.pyplot as plt
import readData as data
from scipy.integrate import trapezoid

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
r, c, t, P = rR * R, cD * D, tD * D, PD * D
r_values = r[:-1]
index_07 = np.where(rR == 0.7)[0][0]  # Find index where rR is 0.7
c07 = c[index_07]
n = np.array([(Rn * nu) / (c07 * D * np.sqrt(J**2 + (0.7 * np.pi)**2)) for J in J_values])
V = J_values * n * D
CLi = np.array([])

def plotCl():
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
    Ut = (4 * gamma) / (2*np.pi*r)
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

def llcWithoutInducedVelocity():
    KT = []
    KQ = []
    eta = []
    for j in range(len(J_values)):
        dT = []
        dQ = []
        n = (Rn * nu) / (c07 * D * np.sqrt(J_values[j]**2 + (0.7 * np.pi)**2))
        V = J_values[j] * n * D
        for i in range(len(r_values)):
            beta = np.arctan(V/(2*np.pi*r_values[i]*n))
            phi = np.arctan(P[i]/(2*np.pi*r_values[i]))
            alpha = phi - beta
            CL = Cl_values[i] + 2 * np.pi * alpha
            Vinf = np.sqrt(V**2 + (2*np.pi*r_values[i]*n)**2)
            gamma = ( CL * Vinf * c[i] )/2
            if (i ==0) or i == (len(r_values) - 1):
                gamma = 0
            Rn_inf = (Vinf * c[i])/nu
            Cf = 0.075/(np.log10(Rn_inf)-2)**2
            Cd = 2 * Cf * (1 + 2 * (t[i]/c[i]) + 60 * (t[i]/c[i])**4) * ( 1 + (CL**2)/8)
            dD = (rho/2) * Vinf**2 * c[i] * Cd 
            dT.append(rho * gamma * 2 * np.pi * r_values[i] * n - dD * np.sin(beta))
            dQ.append((rho * gamma * V + dD * np.cos(beta)) * r_values[i])

        T = z * np.trapz(np.array(dT), r_values, None)
        Q = z * np.trapz(np.array(dQ), r_values, None)
        thrustCoeff = T/(rho * n**2 * D**4)
        torqueCoeff = Q/(rho * n**2 * D**5)
        KT.append(thrustCoeff)
        KQ.append(torqueCoeff * 10)
        eta.append((J_values[j]*thrustCoeff)/(2*np.pi*torqueCoeff))

    return KT, KQ, eta

def simpleInducedVel(tol, max_iter, r):
    num_J = len(J_values)
    num_r = len(r)

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
                CL = Cl_values[i] + 2 * np.pi * alpha
                gamma_new[i] = Vinf * c[i] * CL / 2
                if i == 0 or i == (len(r) - 1):
                    gamma_new[i] = 0

                Ut_all[i] = Ut
                Ua_all[i] = Ua
                beta_all[i] = beta
                Vinf_all[i] = Vinf
                alpha_all[i] = alpha
                CL_all[i] = CL

            # Check difference in gamma (excluding root and tip), reset gamma as needed
            h = gamma_J[1:-1]
            mean = np.mean(gamma_J[1:-1])
            diff = np.linalg.norm(gamma_new[1:-1] - gamma_J[1:-1]) / np.mean(gamma_J[1:-1])
            if diff >= tol:
                gamma_J += 0.1*(gamma_new - gamma_J) # Here we use an under-relaxation term to avoid 'jumping' between circulation distributions.
                iter += 1
            else :
                gamma[j,:] = gamma_new
                beta_i[j] = beta_all

        for k in range(num_r):
            Rn_inf = (Vinf_all[k] * c[k])/nu
            Cf = 0.075/(np.log10(Rn_inf)-2)**2
            #Cd = 2 * Cf * (1 + 2 * (t[k]/c[k]) + 60 * (t[k]/c[k])**4) * ( 1 + (CL_all[k]**2)/8)
            Cd = 2 * Cf * ( 1 + 2 * (t[k]/c[k])) + ((0.15**2 - (EAR/z)**2) * (1.375 + 0.967 * (P[k]/D)**2)) - ((0.15**3 - (EAR/z)**3) * (5.928 + 4.505 * (P[k]/D)**2))
            dD = (rho/2) * Vinf_all[k]**2 * c[k] * Cd 
            dT.append(rho * gamma_new[k] * 2 * np.pi * r[k] * n - dD * np.sin(beta_all[k]))
            dQ.append((rho * gamma_new[k] * V + dD * np.cos(beta_all[k])) * r[k])

        T = z * trapezoid(np.array(dT), r)
        Q = z * trapezoid(np.array(dQ), r)
        thrustCoeff = T/(rho * n**2 * D**4)
        torqueCoeff = Q/(rho * n**2 * D**5)
        KT.append(thrustCoeff)
        KQ.append(torqueCoeff * 10)
        eta.append((J_values[j] / (2*np.pi) ) * (thrustCoeff / torqueCoeff))
    
    return KT, KQ, eta, beta_i

KT, KQ, eta, beta = simpleInducedVel(10e-10, 1000, r_values)
#KT, KQ, eta = llcWithoutInducedVelocity()
plotOpenWaterData(J_values, KT, KQ, eta, J_ow, Ktexp, Kqexp, etaexp)
            


# if __name__ == "__main__":
#     main()
