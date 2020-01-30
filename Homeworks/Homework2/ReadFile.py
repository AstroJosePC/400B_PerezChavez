import numpy as np
import astropy.units as u

def Read(filename):
    # Read file in read mode
    file = open(filename, 'r')
    
    # Read time snapshot, split to isolate value, and attach unit
    line1 = file.readline()
    label, value = line1.split()
    time = float(value) * u.Myr
    
    # Read number of particles, split to isolate value
    line2 = file.readline()
    label, value = line2.split()
    total = float(value)
    
    # Close file
    file.close()
    
    # Get data as numpy array with column labels
    data = np.genfromtxt(filename, dtype=None, names=True, skip_header=3)
    
    # return values
    return time, total, data
