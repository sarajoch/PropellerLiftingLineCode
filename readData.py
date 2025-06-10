import numpy as np
import os

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

def parse_xfoil_file(filepath):
    """Parses an XFOIL-generated file and extracts alpha and CL values."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract r/R from filename
    filename = os.path.basename(filepath)
    rR = float(filename.replace("results", "").replace(".txt", "")) / 1000  # Convert to correct scale
    
    # Find the line where data starts
    for i, line in enumerate(lines):
        if 'alpha' in line and 'CL' in line:
            data_start = i + 2
            break
    
    # Read alpha and CL values
    data = []
    for line in lines[data_start:]:
        values = line.split()
        if len(values) < 2:
            continue
        alpha, CL = np.radians(float(values[0])), float(values[1])
        data.append((alpha, CL))
    
    return rR, np.array(data)

def create_cl_database(directory):
    """Creates a database of CL values for different radii."""
    database = {}
    
    for filename in os.listdir(directory):
        if filename.startswith("results") and filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            rR, data = parse_xfoil_file(filepath)
            database[rR] = data

        if database:
            reference_rR = next(iter(database))  # Get first r/R to extract alpha values
            alpha_values = database[reference_rR][:, 0]  # Use the same alpha range
            cl_values = np.zeros_like(alpha_values)  # CL = 0 for all alphas
            database[1.0] = np.column_stack((alpha_values, cl_values))
    
    return database