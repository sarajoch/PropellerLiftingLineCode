import numpy as np

def geometryData():
    data = np.loadtxt("Data/Geometry.txt", skiprows=1, delimiter=',')
    rR = data[:, 0]
    cD = data[:, 1]
    tD = data[:, 2]
    PD = data[:, 3]

    return rR, cD, tD, PD

def OpenwaterData():
    data = np.loadtxt("Data/OpenwaterData.txt", skiprows=2, delimiter=',')
    J = data[:, 0]
    Kt = data[:, 1]
    Kq = data[:, 2] * 10
    eta = data[:, 3]

    return J, Kt, Kq, eta