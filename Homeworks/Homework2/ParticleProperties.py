from ReadFile import Read
import numpy as np
import astropy.units as u

def ParticleInfo(filename, ptype, pnum):
    """
    Get a particle's distance, velocity, and mass.
    ptype can be integer (1, 2, 3) or string (halo, disk, bulge).
    
    :param filename: file path to data file
    :param ptype: particle type; halo (1), disk (2), or bulge (3) 
    :param pnum: particle number
    :return: distance, velocity, mass
    """
    # If ptype is string, then reassign according to indexing rule
    if isinstance(ptype, str):
        if ptype.lower()=='halo':
            ptype = 1
        elif ptype.lower()=='disk':
            ptype = 2
        elif ptype.lower()=='bulge':
            ptype = 3
        else:
            raise ValueError('Wrong particle type')
    
    # Read data file, and index corresponding particle number/type
    time, total, data = Read(filename)
    particle = data[data['type'] == ptype][pnum]
    
    # Calculate the rounded distance/velocity vector magnitudes, and get mass (w/ Msun units)
    distance = np.round(np.sqrt(particle[2]**2 + particle[3]**2 + particle[4]**2), 3)
    velocity = np.round(np.sqrt(particle[5]**2 + particle[6]**2 + particle[7]**2), 3)
    mass = particle[1] * 1e10 * u.M_sun
    
    # Return calculated values with units
    return distance * u.kpc, velocity * u.km/u.s, mass
