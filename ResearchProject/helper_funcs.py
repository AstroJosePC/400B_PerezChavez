import numpy as np
import astropy.units as u
from numpy.lib.recfunctions import structured_to_unstructured as rec2arr
from scipy.special import comb
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


snapsfolder = '../../VLowResData'
crange=np.linspace(0, 1, 256)

def NewAlphaZeroColormap(orig_cm, name, extent=25):
    """
    Only works for ListedColormaps
    """
    newcolors = orig_cm(crange)
    init = cm.inferno(0, alpha=0)
    final = newcolors[extent]
    t = crange[:extent][:, None] / crange[extent]
    newcolors[:extent] = t * t * t * np.subtract(final, init) + init
    new_cm = ListedColormap(newcolors, name=name)
    cm.register_cmap(name=name, cmap=new_cm)

# Create new color maps with alpha approaching zero for low values
NewAlphaZeroColormap(cm.viridis, 'cviridis')
NewAlphaZeroColormap(cm.inferno, 'cinferno')
NewAlphaZeroColormap(cm.plasma, 'cplasma')
NewAlphaZeroColormap(cm.magma, 'cmagma')
NewAlphaZeroColormap(cm.cividis, 'ccividis')

# This method works for LinearSegmentedColormaps
segms = {'red': ((0.0, 0.0416, 0.0416), (0.365079, 1.0, 1.0), (1.0, 1.0, 1.0)),
 'green': ((0.0, 0.0, 0.0),
  (0.365079, 0.0, 0.0),
  (0.746032, 1.0, 1.0),
  (1.0, 1.0, 1.0)),
 'blue': ((0.0, 0.0, 0.0), (0.746032, 0.0, 0.0), (1.0, 1.0, 1.0)),
'alpha': [(0.0, 0.04, 0.04), (0.1, 0.75, 1.0), (1.0, 1.0, 1.0)]}
new_hot = matplotlib.colors.LinearSegmentedColormap('chot', segmentdata=segms, N=256)
new_hot._init() # create the _lut array, with rgba values
cm.register_cmap(name='chot', cmap=new_hot)


# a function that will rotate the position and velocity vectors
# so that the disk angular momentum is aligned with z axis. 
def GetRotMatrix(posI, velI):
    # input:  3D array of positions and velocities
    # returns: Rotaiton matrix that can rotate vectors such that j is in z direction

    # compute the angular momentum
    L = np.sum(np.cross(posI.T,velI.T), axis=0)
    # normalize the vector
    L_norm = L/np.sqrt(np.sum(L**2))


    # Set up rotation matrix to map L_norm to z unit vector (disk in xy-plane)
    
    # z unit vector
    z_norm = np.array([0, 0, 1])
    
    # cross product between L and z
    vv = np.cross(L_norm, z_norm)
    s = np.sqrt(np.sum(vv**2))
    
    # dot product between L and z 
    c = np.dot(L_norm, z_norm)
    
    # rotation matrix
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_x = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
    R = I + v_x + np.dot(v_x, v_x)*(1 - c)/s**2
    return R
    
    
def RotateFrame(*vectors, RotMat=None, rotAll=False):
    # input:  Rotation matrix, 3D array of positions and velocities
    # returns: 3D array of rotated positions and velocities such that j is in z direction
    if RotMat is None:
        if len(vectors)%2==0:
            RotMatrices = [GetRotMatrix(pos, vel) for pos,vel in zip(vectors[::2], vectors[1::2])]
            if rotAll:
                # TODO: This one would actually not work...
                return [RotateFrame(vector, RotMat=rotmat) for vector, rotmat in zip(vectors, RotMatrices)]
            else:
                if len(vectors)==2:
                    return RotateFrame(vectors[0], RotMat=RotMatrices[0])
                else:
                    return [RotateFrame(vector, RotMat=rotmat) for vector, rotmat in zip(vectors[::2], RotMatrices)]
        else:
            raise ValueError('The entered vectors should be paired position and velocity vectors')
    else:
        # Rotate coordinate system
        if len(vectors) > 1:
            return [np.dot(RotMat, vector) for vector in vectors]
        else:
            return np.dot(RotMat, vectors[0])

def SurfaceDensityProfile(mass_data, r_vector=None, cyl_vector=None, radii=None):
    # Surface Density Profile Calculations
    
    # calculate the radial distances and azimuthal angles in the cylindrical coordinates
    if r_vector is None and cyl_vector is None:
        raise( ValueError('At last one vector representation must be input') )
    if not cyl_vector:
        cyl_vector = Cyl_Coords(r_vector)
    if not radii:
        radii = np.arange(0.1, 0.95 * cyl_vector[0].max(), 1.0)
    
    # create the mask to select particles for each radius
    # np.newaxis creates a virtual axis to make tmp_r_mag 2 dimensional
    # so that all radii can be compared simultaneously
    enc_mask = cyl_vector[0][:, np.newaxis] < np.asarray(radii).flatten()
    # calculate the enclosed masses within each radius
    # relevant particles will be selected by enc_mask (i.e., *1)
    # outer particles will be ignored (i.e., *0)
    m_enc = np.sum(mass_data[:, np.newaxis] * enc_mask, axis=0)

    # use the difference between nearby elements to get mass in each annulus
    # N.B.: we ignored the very central tiny circle and a small portion of outermost particles
    #       feel free to modify it to fit your needs
    m_annuli = np.diff(m_enc) # this array is one element less then m_enc

    # calculate the surface density by dividing the area of the annulus
    sigma = m_annuli / (np.pi * (radii[1:]**2 - radii[:-1]**2))

    # we use the geometric mean of two consecutive elements in "radii" as the radius of each annulus
    # this array have the same amount of elements as self.Sigma, can be used for plotting
    r_annuli = np.sqrt(radii[1:] * radii[:-1])
    return r_annuli, sigma


def Cyl_Coords(*r_vectors, notZ=True):
    """
    r_vector: R-vector in 3, N shape
    :return: cylindrical coords vector 3, N shape; rho, theta, z
    """
    cyl_vectors = []
    for r_vector in r_vectors:
        rho_mag = np.sqrt((r_vector[:2]**2).sum(axis=0))
        theta = np.arctan2(r_vector[1], r_vector[0])
        if notZ:
            vectors = np.asarray([rho_mag, theta])
        else:
            height = vector[2]
            vectors = np.asarray([rho_mag, theta, height])
        
        cyl_vectors.append(vectors)
        
    if len(cyl_vectors) > 1:
        return cyl_vectors
    else:
        return cyl_vectors[0]


def Lerp(init, final, frac=None, smooth=True, N=4):
    """
    linear interpolation b/t two values according to a fractional weight
    if smooth, then we  smooth curve, not a linear fraction
    """
    if frac is None:
        frac = np.linspace(0, 1, 802)
    if smooth:
        t = smoothstep(frac, N=4)
    else:
        t = frac
    return t*(final - init) + init

def smoothstep(x=None, x_min=0, x_max=1, N=4):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    return result

def Read(filename, time=False):
    # Read file in read mode
    
    with open(filename, 'r') as file:
        # Read time snapshot, split to isolate value, and attach unit
        line1 = file.readline()
        label, value = line1.split()
        timeMyr = float(value) * u.Myr

        # Get data as numpy array with column labels
        data = np.genfromtxt(file, dtype=np.float32, names=True, skip_header=2)
    
    if time:
        return timeMyr, data
    else:
        return data
    
    
def get_snapshot(gdata, baryons=False, snap=0):
    if isinstance(gdata, str):
        if len(gdata) <= 3:
            gname = gdata.upper()
            data = Read(f'{snapsfolder}/{gname}_VLowRes/{gname}_{snap:03d}.txt', time=False)
        else:
            data = Read(gdata, time=False)
        if baryons:
            baryon_data = data[data ['type'] != 1]
            r = rec2arr(baryon_data[['x', 'y', 'z']]).T
            v = rec2arr(baryon_data[['vx', 'vy', 'vz']]).T
            return data, baryon_data, r, v
        else:
            r = rec2arr(data[['x', 'y', 'z']]).T
            v = rec2arr(data[['vx', 'vy', 'vz']]).T
            return data, r, v
    else:
        r = rec2arr(gdata[['x', 'y', 'z']]).T
        v = rec2arr(gdata[['vx', 'vy', 'vz']]).T
        return r, v

def get_orbit(orbit_data, snap=0):
    if isinstance(orbit_data, str):
        if len(orbit_data) <= 3:
            gname = orbit_data.upper()
            com = np.genfromtxt(f'../Data/Orbit_{gname}.txt', names=True)
        else:
            com = np.genfromtxt(orbit_data, names=True)
        r = rec2arr(com[snap][['x', 'y', 'z']])[:, None]
        v = rec2arr(com[snap][['vx', 'vy', 'vz']])[:, None]
        return com, r, v
    else:
        r = rec2arr(orbit_data[snap][['x', 'y', 'z']])[:, None]
        v = rec2arr(orbit_data[snap][['vx', 'vy', 'vz']])[:, None]
        return r, v

def RotateTop(gvectors, comr):
    mw_rcom = mw_r - mw_com_r
    mw_rotmat = GetRotMatrix(mw_rcom, mw_vcom)
    mw_top_rcom, = RotateFrame(mw_rotmat, mw_rcom)

    
def getMag(*vectors, axis=0):
    if len(vectors) > 1:
        return [np.sqrt((vector**2).sum(axis=axis)) for vector in vectors]
    else:
        return np.sqrt((vectors[0]**2).sum(axis=axis))


def JacobiRadius(smallR, bigR, smallData, bigData, Rmax=None, fraction=1/2):
    # Calculate the jacobi radius of the M33-M31 System
    D = np.sqrt(((bigR - smallR)**2).sum())
    bigRmags = getMag(rec2arr(bigData[['x', 'y', 'z']]) - bigR.T, axis=1)
    BigMassEncl = bigData[bigRmags < D]['m'].sum()

    smallRmags = getMag(rec2arr(smallData[['x', 'y', 'z']]) - smallR.T, axis=1)
    if (Rmax is None) or (Rmax > D):
        SmallMassEncl = smallData[smallRmags < D*fraction]['m'].sum()        
    else:
        SmallMassEncl = smallData[smallRmags < Rmax]['m'].sum()
    jac_radius = D * (SmallMassEncl / (2 * BigMassEncl) )**(1/3)
    return jac_radius


def get_Frame(snap=0):
    # Load in galaxy data, separate baryons, and position and velocity vectors
    mw_data, mw_baryons, mw_r, mw_v = get_snapshot('mw', baryons=True, snap=snap)
    m31_data, m31_baryons, m31_r, m31_v = get_snapshot('m31', baryons=True, snap=snap)
    m33_data, m33_baryons, m33_r, m33_v = get_snapshot('m33', baryons=True, snap=snap)

    # Center of Mass info from Orbits
    mw_com, mw_com_r, mw_com_v = get_orbit('mw', snap=snap)
    m31_com, m31_com_r, m31_com_v = get_orbit('m31', snap=snap)
    m33_com, m33_com_r, m33_com_v = get_orbit('m33', snap=snap)

    time = mw_com['t'][snap] * u.Myr

    # Transform particle vectors to CoM per Galaxy; 
    # Shapes: (3, Nparticles)
    mw_rcom = mw_r - mw_com_r
    mw_vcom = mw_v - mw_com_v
    m31_rcom = m31_r - m31_com_r
    m31_vcom = m31_v - m31_com_v

    # Rotate Frame along angular momentun axis; M33 isn't centered just YET
    mw_top_rcom, m31_top_rcom = RotateFrame(mw_rcom, mw_vcom, m31_rcom, m31_vcom)

    # NOW Rotate each galaxy (in original frame) to M33's L-aligned frame; including M33
    # Then rotate M33's center of mass vector to then center M33 in rotated frame
    m33_rotmat = GetRotMatrix(m33_r, m33_v)
    m33_top_r, mw_m33_r, m31_m33_r, m33_top_com_r = RotateFrame(m33_r, mw_r, m31_r, m33_com_r, RotMat=m33_rotmat)
    m33_top_com_r = RotateFrame(m33_com_r, RotMat=m33_rotmat)

    # finally center all rotated frames above to M33's CoM
    m33_top_rcom = m33_top_r - m33_top_com_r
    mw_m33_rcom, m31_m33_rcom = mw_m33_r - m33_top_com_r, m31_m33_r - m33_top_com_r

    # Get R magnitudes of each galaxy particles with respect to its CoM
    mw_rmag, m31_rmag, m33_rmag = getMag(mw_rcom, m31_rcom, m33_top_rcom)

    # Finally create an indexing array removing galaxy "outlier" particles
    mw_rmin, mw_rmax = np.percentile(mw_rmag, [1, 99.8])
    m31_rmin, m31_rmax = np.percentile(m31_rmag, [1, 99.8])
    m33_rmin, m33_rmax = np.percentile(m33_rmag, [1, 99.8])

    mw_pselection, = np.where(mw_rmag < mw_rmax)
    m31_pselection, = np.where(m31_rmag < m31_rmax)
    m33_pselection, = np.where(m33_rmag < m33_rmax)

    # Calculate cylindrical coordinates of each top-view galaxy frame
    mw_cyl, m31_cyl, m33_cyl = Cyl_Coords(mw_top_rcom, m31_top_rcom, m33_top_rcom)
    return mw_top_rcom, m31_top_rcom, m33_top_rcom, mw_m33_rcom, m31_m33_rcom, mw_cyl, m31_cyl, m33_cyl


def updateScatters(*args):
    for scatt, cyl, select in zip(args[::3], args[1::3], args[2::3]):
        scatt.set_offsets(np.c_[cyl[1][select], cyl[0][select]])
        scatt.set_sizes(np.log10(cyl[0][select]**2 + 1.0))
        scatt.set_array(cyl[0][select])

        