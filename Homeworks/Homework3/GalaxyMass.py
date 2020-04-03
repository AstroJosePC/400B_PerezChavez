# Add path to other HW folders
# Incites modular design / No need to copy ReadFile into Homework3 folder
import sys
sys.path.append("../")

# Import Functions
from Homework2.ReadFile import Read
import numpy as np

def GetParticleType(ptype):
    """
    Function to parse particle type parameter.
    `ptype` can be an integer (1-3) or string (halo, disk, bulge).

    :param ptype: particle type => halo (1), disk (2), or bulge (3) 
    :return: parsed particle type
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
    
    # else assume it is an integer; check it's within range
    else:
        if not (1 <= ptype <= 3):
            raise ValueError('Particle type out of range (1,3)')
    return ptype


def GalaxyMass(input_file, ptype):
    """
    Read in MK data file, and return the total mass of any 
    desired galaxy component: Halo (1), Disk (2), Bulge (3)
    
    `ptype` can be integer or string.

    :param filename: file path to data file (string) OR data array with headers
    :param ptye: particle type; halo (1), disk (2), or bulge (3) 
    :return: Total component mass in 10^12 Msun
    
    """
    # Read data in
    if isinstance(input_file, str):
        data = Read(input_file)[2]
    else:
        data = input_file
    
    # Mask of component mass using parsed ptype
    mask = data['type'] == GetParticleType(ptype)

    # Now get the mass of all particles in component in 10^10 Msun
    masses = data[mask]['m']

    # Sum up masses; Divide by 100 to express in 10^12 Msun.
    total_mass = masses.sum() / 100

    # Return rounded mass value
    return np.round(total_mass, 3)