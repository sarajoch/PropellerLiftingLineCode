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

KT, KQ, eta = llcWithoutInducedVelocity()
plotOpenWaterData(J_values, KT, KQ, eta, J_ow, Ktexp, Kqexp, etaexp)
            


# if __name__ == "__main__":
#     main()
