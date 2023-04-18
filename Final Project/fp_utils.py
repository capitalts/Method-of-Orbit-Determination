import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.special import lpmn
import pandas as pd
from datetime import datetime
import sympy as sym
import csv

leap_sec = 37 #s
x_p = np.array([20.816, 22.156, 23.439, 24.368, 25.676, 26.952, 28.108])/1000 #arcsec
y_p = np.array([381.008, 382.613, 384.264, 385.509, 386.420, 387.394, 388.997])/1000 #arcsec
del_UT1 = np.array([144.0585, 143.1048, 142.2335, 141.3570, 140.4078, 139.3324, 138.1510])/1000

#Rotation Martices
def R1(theta):
    return np.array([[1, 0, 0], \
                     [0, np.cos(theta), np.sin(theta)], \
                     [0, -np.sin(theta), np.cos(theta)]])
def R2(theta):
    return np.array([[np.cos(theta), 0, -np.sin(theta)],\
                    [0, 1, 0], \
                    [np.sin(theta), 0, np.cos(theta)]])
def R3(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0], \
                     [-np.sin(theta), np.cos(theta), 0], \
                    [0, 0, 1]])

#read in nutation data file
def read_nut80():
    # IAU1980 Theory of Nutation model
    dat_file = "nut80.dat"  

    #nutaton model column names
    column_names = ['ki1', 'ki2', 'ki3', 'ki4', 'ki5', 'Aj', 'Bj', 'Cj', 'Dj', 'j']

    #nutation dataframe
    df = pd.read_csv(dat_file, sep="\s+", names=column_names)
    return df

df = read_nut80()


def gregorian_to_jd(year, month, day, hour, minute, second):
    '''Convert Gregorian calendar date to Julian date time.
    Input
        year : int
        month : int
        day : int
        hour : int
        minute : int
        second : float
    Returns
        jd : float '''
    
    a = int((14 - month)/12)
    y = year + 4800 - a
    m = month + 12*a - 3
    jd = day + int((153*m + 2)/5) + 365*y + int(y/4) - int(y/100) + int(y/400) - 32045
    jd = jd + (hour - 12)/24 + minute/1440 + second/86400
    return jd

def ECI2ECEF(r_ECI, JD_UTC, x_p, y_p, leap_sec, del_UT1):

    '''
    Converts ECI to ECEF using IAU-76/FK5

    Inputs:
    r_ECI: ECI position vector
    JD_UTC: Julian Date in UTC
    x_p: x polar motion in arc seconds
    y_p: y polar motion in arc seconds
    leap_sec: leap seconds
    del_UT1: UT1-UTC in seconds

    returns: ECI position vector
    '''

    # time constants
    JD2000 = 2451545.0

    #T_UT1
    JD_UT1 = JD_UTC + del_UT1/86400
    T_UT1 = (JD_UT1-JD2000)/36525

    #T_TT
    TAI = JD_UTC + leap_sec/86400
    JD_TT = TAI + 32.184/86400
    T_TT = (JD_TT-JD2000)/36525

    #radians conversions
    arc_sec_to_rad = np.pi/(180*3600)
    deg2rad = np.pi/180

    #Earth Rotation Angles
    x_p = x_p*arc_sec_to_rad
    y_p = x_p*arc_sec_to_rad

    # Polar Motion Matrix
    W = np.matmul(R1(y_p), R2(x_p))
    # r_PEF = np.matmul(W, r_ECEF)

    #Greenwich Mean Sidereal Time
    GMST = 67310.54841 + (876600*3600 + 8640184.812866)*T_UT1 + 0.093104*T_UT1**2 - 6.2E-6*T_UT1**3

    #convert GMST to radians
    GMST = GMST/240*deg2rad

    #anamolies
    r = 360
    Mmoon = (134.96298139 + (1325*r + 198.8673981)*T_TT + 0.0086972*T_TT**2 + 1.78E-5*T_TT**3)
    Mdot = (357.52772333 + (99*r + 359.0503400)*T_TT - 0.0001603*T_TT**2 - 3.3E-6*T_TT**3)
    uMoon = (93.27191028 + (1342*r + 82.0175381)*T_TT - 0.0036825*T_TT**2 + 3.1E-6*T_TT**3)
    Ddot = (297.85036306 + (1236*r + 307.1114800)*T_TT - 0.0019142*T_TT**2 + 5.3E-6*T_TT**3)
    lamMoon = (125.04452222 - (5*r + 134.1362608)*T_TT + 0.0020708*T_TT**2 + 2.2E-6*T_TT**3)
    alpha = np.array([Mmoon, Mdot, uMoon, Ddot, lamMoon])*deg2rad

    # # IAU1980 Theory of Nutation model
    # dat_file = "nut80.dat"  

    # #nutaton model column names
    # column_names = ['ki1', 'ki2', 'ki3', 'ki4', 'ki5', 'Aj', 'Bj', 'Cj', 'Dj', 'j']

    # #nutation dataframe
    # df = pd.read_csv(dat_file, sep="\s+", names=column_names)

    #nutation in lam
    del_psi = np.dot((df['Aj']*10**-4 + df['Bj']*10**-4*T_TT)*arc_sec_to_rad, np.sin(np.dot(df[df.columns[0:5]], alpha)))

    #nutation in obliquity
    del_epsilon = np.dot((df['Cj']*10**-4 + df['Dj']*10**-4*T_TT)*arc_sec_to_rad, np.cos(np.dot(df[df.columns[0:5]], alpha)))

    #mean obliquity of the ecliptic
    epsilon_m = 84381.448 - 46.8150*T_TT - 0.00059*T_TT**2 + 0.001813*T_TT**3

    #EOP corrections
    # ddel_psi = -104.524E-3
    # ddel_epsilon = -8.685E-3

    #conversion to radians
    epsilon_m = epsilon_m*arc_sec_to_rad

    #true obliquity of the ecliptic
    epsilon = epsilon_m + del_epsilon

    #equation of the equinoxes
    Eq_eq = del_psi*np.cos(epsilon_m) + 0.000063*arc_sec_to_rad*np.sin(2*alpha[4]) + 0.00264*arc_sec_to_rad*np.sin(alpha[4])

    #greenwich apparent sidereal time
    GAST = GMST + Eq_eq

    #sidereal rotation matrix
    R = R3(-GAST)
    # r_TOD = np.matmul(R, r_PEF)

    #nutation matrix R1, R3, R1
    N = np.matmul(np.matmul(R1(-epsilon_m), R3(del_psi)), R1(epsilon))
    # r_mod = np.matmul(N, r_TOD)

    #precession angles
    C_a = (2306.2181*T_TT + 0.30188*T_TT**2 + 0.017998*T_TT**3)*arc_sec_to_rad
    theta_a = (2004.3109*T_TT - 0.42665*T_TT**2 - 0.041833*T_TT**3)*arc_sec_to_rad
    z_a = (2306.2181*T_TT + 1.09468*T_TT**2 + 0.018203*T_TT**3)*arc_sec_to_rad

    #precession matrix
    P = np.matmul(np.matmul(R3(C_a), R2(-theta_a)), R3(z_a))

    r_ECEF = np.matmul(np.matmul(np.matmul(np.matmul(W.T, R.T), N.T), P.T), r_ECI)
    return r_ECEF


def ECEF2ECI(r_ECEF, JD_UTC, x_p, y_p, leap_sec, del_UT1):

    '''
    Converts ECEF to ECI using IAU-76/FK5

    Inputs:
    r_ECEF: ECEF position vector
    JD_UTC: Julian Date in UTC
    x_p: x polar motion in arc seconds
    y_p: y polar motion in arc seconds
    leap_sec: leap seconds
    del_UT1: UT1-UTC in seconds

    returns: ECI position vector
    '''

    # time constants
    JD2000 = 2451545.0
    del_UT1 /= 1000

    #T_UT1
    JD_UT1 = JD_UTC + del_UT1/86400
    T_UT1 = (JD_UT1-JD2000)/36525

    #T_UT1
    TAI = JD_UTC + leap_sec/86400
    JD_TT = TAI + 32.184/86400
    T_TT = (JD_TT-JD2000)/36525

    #radians conversions
    arc_sec_to_rad = np.pi/(180*3600)
    deg2rad = np.pi/180

    #Earth Rotation Angles
    x_p = x_p*arc_sec_to_rad
    y_p = x_p*arc_sec_to_rad

    # Polar Motion Matrix
    W = np.matmul(R1(y_p), R2(x_p))
    # r_PEF = np.matmul(W, r_ECEF)

    #Greenwich Mean Sidereal Time
    GMST = 67310.54841 + (876600*3600 + 8640184.812866)*T_UT1 + 0.093104*T_UT1**2 - 6.2E-6*T_UT1**3

    #convert GMST to radians
    GMST = GMST/240*deg2rad

    #anamolies
    r = 360
    Mmoon = (134.96298139 + (1325*r + 198.8673981)*T_TT + 0.0086972*T_TT**2 + 1.78E-5*T_TT**3)
    Mdot = (357.52772333 + (99*r + 359.0503400)*T_TT - 0.0001603*T_TT**2 - 3.3E-6*T_TT**3)
    uMoon = (93.27191028 + (1342*r + 82.0175381)*T_TT - 0.0036825*T_TT**2 + 3.1E-6*T_TT**3)
    Ddot = (297.85036306 + (1236*r + 307.1114800)*T_TT - 0.0019142*T_TT**2 + 5.3E-6*T_TT**3)
    lamMoon = (125.04452222 - (5*r + 134.1362608)*T_TT + 0.0020708*T_TT**2 + 2.2E-6*T_TT**3)
    alpha = np.array([Mmoon, Mdot, uMoon, Ddot, lamMoon])*deg2rad

    # # IAU1980 Theory of Nutation model
    # dat_file = "nut80.dat"  

    # #nutaton model column names
    # column_names = ['ki1', 'ki2', 'ki3', 'ki4', 'ki5', 'Aj', 'Bj', 'Cj', 'Dj', 'j']

    # #nutation dataframe
    # df = pd.read_csv(dat_file, sep="\s+", names=column_names)

    #nutation in lam
    del_psi = np.dot((df['Aj']*10**-4 + df['Bj']*10**-4*T_TT)*arc_sec_to_rad, np.sin(np.dot(df[df.columns[0:5]], alpha)))

    #nutation in obliquity
    del_epsilon = np.dot((df['Cj']*10**-4 + df['Dj']*10**-4*T_TT)*arc_sec_to_rad, np.cos(np.dot(df[df.columns[0:5]], alpha)))

    #mean obliquity of the ecliptic
    epsilon_m = 84381.448 - 46.8150*T_TT - 0.00059*T_TT**2 + 0.001813*T_TT**3

    #EOP corrections
    # ddel_psi = -104.524E-3
    # ddel_epsilon = -8.685E-3

    #conversion to radians
    epsilon_m = epsilon_m*arc_sec_to_rad

    #true obliquity of the ecliptic
    epsilon = epsilon_m + del_epsilon

    #equation of the equinoxes
    Eq_eq = del_psi*np.cos(epsilon_m) + 0.000063*arc_sec_to_rad*np.sin(2*alpha[4]) + 0.00264*arc_sec_to_rad*np.sin(alpha[4])

    #greenwich apparent sidereal time
    GAST = GMST + Eq_eq

    #sidereal rotation matrix
    R = R3(-GAST)
    # r_TOD = np.matmul(R, r_PEF)

    #nutation matrix R1, R3, R1
    N = np.matmul(np.matmul(R1(-epsilon_m), R3(del_psi)), R1(epsilon))
    # r_mod = np.matmul(N, r_TOD)

    #precession angles
    C_a = (2306.2181*T_TT + 0.30188*T_TT**2 + 0.017998*T_TT**3)*arc_sec_to_rad
    theta_a = (2004.3109*T_TT - 0.42665*T_TT**2 - 0.041833*T_TT**3)*arc_sec_to_rad
    z_a = (2306.2181*T_TT + 1.09468*T_TT**2 + 0.018203*T_TT**3)*arc_sec_to_rad

    #precession matrix
    P = np.matmul(np.matmul(R3(C_a), R2(-theta_a)), R3(z_a))

    r_ECI = np.matmul(np.matmul(np.matmul(np.matmul(P, N), R), W), r_ECEF)
    return r_ECI

def sun_position_vector(JD_UTC, del_UT1, leap_sec):
    '''
    Returns the position vector of the sun in ECI coordinates

    Inputs:
        JD_UTC: Julian Date in UTC
        del_UT1: difference between UT1 and UTC in milliseconds
        leap_sec: number of leap seconds
    Returns:
        r_ECI: position vector of the sun in ECI coordinates
    '''
    
    # Constants
    deg2rad = np.pi / 180.0
    au = 149597870.691*1000 # Astronomical unit [m]
    arc_sec_to_rad = np.pi/(180*3600)
    # Time variables
    
    JD2000 = 2451545.0

    del_UT1 /=1000

    #T_UT1
    JD_UT1 = JD_UTC + del_UT1/86400
    T_UT1 = (JD_UT1-JD2000)/36525

    #T_TT
    TAI = JD_UTC + leap_sec/86400
    JD_TT = TAI + 32.184/86400
    T_TT = (JD_TT-JD2000)/36525

    # Mean lam of the Sun
    l = (280.460 + 36000.771285 * T_UT1) %360

    # Mean anomaly of the Sun
    M = (357.528 + 35999.050957 * T_UT1) %360

    # Ecliptic lam of the Sun
    lambda_sun = l + 1.915 * np.sin(M * deg2rad) + 0.020 * np.sin(2 * M * deg2rad)

    # Obliquity of the ecliptic
    epsilon = 23.439291 - 0.01461 * T_UT1

    #magnitude of the sun
    R = 1.00014 - 0.01671 * np.cos(M * deg2rad) - 0.00014 * np.cos(2 * M * deg2rad)

    #sun position vector in ecliptic coordinates
    r_ecliptic = np.array([R * np.cos(lambda_sun * deg2rad), 
                           R * np.cos(epsilon * deg2rad) * np.sin(lambda_sun * deg2rad), 
                           R * np.sin(epsilon * deg2rad) * np.sin(lambda_sun * deg2rad)])
    
    #rotation from TOD to ECI

    #anamolies
    r = 360
    Mmoon = (134.96298139 + (1325*r + 198.8673981)*T_TT + 0.0086972*T_TT**2 + 1.78E-5*T_TT**3)
    Mdot = (357.52772333 + (99*r + 359.0503400)*T_TT - 0.0001603*T_TT**2 - 3.3E-6*T_TT**3)
    uMoon = (93.27191028 + (1342*r + 82.0175381)*T_TT - 0.0036825*T_TT**2 + 3.1E-6*T_TT**3)
    Ddot = (297.85036306 + (1236*r + 307.1114800)*T_TT - 0.0019142*T_TT**2 + 5.3E-6*T_TT**3)
    lamMoon = (125.04452222 - (5*r + 134.1362608)*T_TT + 0.0020708*T_TT**2 + 2.2E-6*T_TT**3)
    alpha = np.array([Mmoon, Mdot, uMoon, Ddot, lamMoon])*deg2rad


    #nutation in lam
    del_psi = np.dot((df['Aj']*10**-4 + df['Bj']*10**-4*T_TT)*arc_sec_to_rad, np.sin(np.dot(df[df.columns[0:5]], alpha)))

    #nutation in obliquity
    del_epsilon = np.dot((df['Cj']*10**-4 + df['Dj']*10**-4*T_TT)*arc_sec_to_rad, np.cos(np.dot(df[df.columns[0:5]], alpha)))

    #mean obliquity of the ecliptic
    epsilon_m = 84381.448 - 46.8150*T_TT - 0.00059*T_TT**2 + 0.001813*T_TT**3

    #conversion to radians
    epsilon_m = epsilon_m*arc_sec_to_rad

    #true obliquity of the ecliptic
    epsilon = epsilon_m + del_epsilon

    #nutation matrix R1, R3, R1
    N = np.matmul(np.matmul(R1(-epsilon_m), R3(del_psi)), R1(epsilon))

    #precession angles
    C_a = (2306.2181*T_TT + 0.30188*T_TT**2 + 0.017998*T_TT**3)*arc_sec_to_rad
    theta_a = (2004.3109*T_TT - 0.42665*T_TT**2 - 0.041833*T_TT**3)*arc_sec_to_rad
    z_a = (2306.2181*T_TT + 1.09468*T_TT**2 + 0.018203*T_TT**3)*arc_sec_to_rad

    #precession matrix
    P = np.matmul(np.matmul(R3(C_a), R2(-theta_a)), R3(z_a))

    #sun position vector in ECI
    r_ECI = np.matmul(P, np.matmul(N, r_ecliptic))*au

    return r_ECI


def moon_position_vector(JD_UTC, del_UT1, leap_sec):
    '''
    Calculates the position vector of the moon in ECI coordinates
    
    Inputs:
    JD_UTC - Julian date in UTC
    del_UT1 - UT1-UTC in ms
    leap_sec - leap seconds

    Returns:
    r_ECI - position vector of the moon in ECI coordinates
    '''

    # Constants
    deg2rad = np.pi / 180.0
    # Time variables
    
    JD2000 = 2451545.0

    #T_UT1
    JD_UT1 = JD_UTC + del_UT1/86400
    T_UT1 = (JD_UT1-JD2000)/36525

    #T_TT
    TAI = JD_UTC + leap_sec/86400
  
    JD_TT = TAI + 32.184/86400
    T_TT = (JD_TT-JD2000)/36525
  
    # Mean lam of the Moon
    l = (218.32 + 481267.8813*T_TT + 6.29*np.sin((134.9 + 477198.85*T_TT)*deg2rad) \
        - 1.27*np.sin((259.2 - 413335.38*T_TT)*deg2rad) + 0.66*np.sin((235.7 + 890534.23*T_TT)*deg2rad) \
        + 0.21*np.sin((269.9 + 954397.70*T_TT)*deg2rad) - 0.19*np.sin((357.5 + 35999.05*T_TT)*deg2rad) \
        - 0.11*np.sin((186.6 + 966404.05*T_TT)*deg2rad)) % 360
   
    #ecliptic lattitude of the Moon
    phi = (5.13*np.sin((93.3 + 483202.03*T_TT)*deg2rad) + 0.28*np.sin((228.2 + 960400.87*T_TT)*deg2rad) \
        - 0.28*np.sin((318.3 + 6003.18*T_TT)*deg2rad) - 0.17*np.sin((217.6 - 407332.20*T_TT)*deg2rad)) %360
  
    # Horizontal parallax of the Moon
    O = (0.9508 + 0.0518*np.cos((134.9 + 477198.85*T_TT)*deg2rad) \
        + 0.0095*np.cos((259.2 - 413335.38*T_TT)*deg2rad) + 0.0078*np.cos((235.7 + 890534.23*T_TT)*deg2rad) \
        + 0.0028*np.cos((269.9 + 954397.70*T_TT)*deg2rad)) % 360
    
    #oblquity of the ecliptic
    epsilon = (23.439291 - 0.0130042*T_TT - 1.64E-7*T_TT**2 + 5.04E-7*T_TT**3) % 360
    #magnitude of the vector from the Earth to the Moon
    R_earth = 6378.1363*1000 #m
    r_moon = R_earth/np.sin(O*deg2rad)
    #moon position vector in ecliptic coordinates
    r_ecliptic = np.array([r_moon*np.cos(phi*deg2rad)*np.cos(l*deg2rad), \
                           r_moon*(np.cos(epsilon*deg2rad)*np.cos(phi*deg2rad)*np.sin(l*deg2rad) - np.sin(epsilon*deg2rad)*np.sin(phi*deg2rad)), \
                            r_moon*(np.sin(epsilon*deg2rad)*np.cos(phi*deg2rad)*np.sin(l*deg2rad) + np.cos(epsilon*deg2rad)*np.sin(phi*deg2rad))])
   
    #rotation from TOD to ECI
    arc_sec_to_rad = np.pi/(180*3600)
    #anamolies
    r = 360
    Mmoon = (134.96298139 + (1325*r + 198.8673981)*T_TT + 0.0086972*T_TT**2 + 1.78E-5*T_TT**3)
    Mdot = (357.52772333 + (99*r + 359.0503400)*T_TT - 0.0001603*T_TT**2 - 3.3E-6*T_TT**3)
    uMoon = (93.27191028 + (1342*r + 82.0175381)*T_TT - 0.0036825*T_TT**2 + 3.1E-6*T_TT**3)
    Ddot = (297.85036306 + (1236*r + 307.1114800)*T_TT - 0.0019142*T_TT**2 + 5.3E-6*T_TT**3)
    lamMoon = (125.04452222 - (5*r + 134.1362608)*T_TT + 0.0020708*T_TT**2 + 2.2E-6*T_TT**3)
    alpha = np.array([Mmoon, Mdot, uMoon, Ddot, lamMoon])*deg2rad

    #nutation in lam
    del_psi = np.dot((df['Aj']*10**-4 + df['Bj']*10**-4*T_TT)*arc_sec_to_rad, np.sin(np.dot(df[df.columns[0:5]], alpha)))

    #nutation in obliquity
    del_epsilon = np.dot((df['Cj']*10**-4 + df['Dj']*10**-4*T_TT)*arc_sec_to_rad, np.cos(np.dot(df[df.columns[0:5]], alpha)))

    #mean obliquity of the ecliptic
    epsilon_m = 84381.448 - 46.8150*T_TT - 0.00059*T_TT**2 + 0.001813*T_TT**3

    #conversion to radians
    epsilon_m = epsilon_m*arc_sec_to_rad

    #true obliquity of the ecliptic
    epsilon = epsilon_m + del_epsilon

    #nutation matrix R1, R3, R1
    N = np.matmul(np.matmul(R1(-epsilon_m), R3(del_psi)), R1(epsilon))

    #precession angles
    C_a = (2306.2181*T_TT + 0.30188*T_TT**2 + 0.017998*T_TT**3)*arc_sec_to_rad
    theta_a = (2004.3109*T_TT - 0.42665*T_TT**2 - 0.041833*T_TT**3)*arc_sec_to_rad
    z_a = (2306.2181*T_TT + 1.09468*T_TT**2 + 0.018203*T_TT**3)*arc_sec_to_rad

    #precession matrix
    P = np.matmul(np.matmul(R3(C_a), R2(-theta_a)), R3(z_a))

    #sun position vector in ECI
    r_ECI = np.matmul(P, np.matmul(N, r_ecliptic))


    return r_ECI

def satellite_motion_phi(t, R, A, JD_UTC):
    '''Calculates the derivative of the state vector for the satellite motion
    Inputs:
        t: time in seconds
        R: state vector
        A: matrix of the linearized dynamics
        JD_UTC: Julian date in UTC
        
    Outputs:
        r_ddot: derivative of the position vector
        phi: derivative of the state vector

    '''

    mu = 398600.4415*1000**3 #m^3/s^2
    phi = R[6:].reshape(7, 7)
    r = R[0:3]
    r_dot = R[3:6]
    x, y, z = R[0:3]
    x_dot, y_dot, z_dot = R[3:6]
    JD_UTC += t/86400

    #J2
    r_ddot_gravity = a_gravity_J2(r)

    #drag
    A_Cross = 6
    C_D = 1.88
    r_ddot_drag = a_drag(C_D, r, r_dot, A_Cross)

    #solar
    r_sun = np.zeros((1, 3))
    r_sun[0] = sun_position_vector(JD_UTC, leap_sec, del_UT1[0])
    C_s = 0.04
    C_d = 0.04
    A_Cross_sol = 15
    r_ddot_sol = a_solar(r, r_sun[0], C_s, C_d, A_Cross_sol)
    

    #third body
    r_moon = np.zeros((1, 3))
    r_moon[0] = moon_position_vector(JD_UTC, leap_sec, del_UT1[0])
    r_ddot_tb = a_third_body(r, r_sun[0], r_moon[0])


    #total acceleration
    r_ddot = r_ddot_gravity + r_ddot_drag  + r_ddot_sol + r_ddot_tb
    
    #A matrix
    A_1 = np.array(A(x, y, z, x_dot, y_dot, z_dot, C_D, A_Cross, A_Cross_sol, r_sun, r_moon))
    
    #state transition matrix
    phi_dot = np.matmul(A_1, phi)
   
    dydt = np.concatenate((r_dot, r_ddot, phi_dot.ravel()))
    
    return dydt

def satellite_motion(t, R, JD_UTC):
    '''
    Calculates the state vector of a satellite in ECI coordinates

    '''

    r = R[0:3]
    r_dot = R[3:6]
    x, y, z = R[0:3]
    JD_UTC += t/86400
    
    #J2
    r_ddot_gravity = a_gravity_J2(r)

    # #drag
    A_Cross = 6
    C_D = 1.88
    r_ddot_drag = a_drag(C_D, r, r_dot, A_Cross)

    # #solar
    r_sun = np.zeros((1, 3))
    r_sun[0] = sun_position_vector(JD_UTC, leap_sec, del_UT1[0])
    C_s = 0.04
    C_d = 0.04
    A_Cross_sol = 15
    r_ddot_sol = a_solar(r, r_sun[0], C_s, C_d, A_Cross_sol)

    # #third body
    r_moon = np.zeros((1, 3))
    r_moon[0] = moon_position_vector(JD_UTC, leap_sec, del_UT1[0])
    r_ddot_tb = a_third_body(r, r_sun[0], r_moon[0])


    #total acceleration
    r_ddot = r_ddot_gravity + r_ddot_drag  + r_ddot_sol + r_ddot_tb

    dydt = np.concatenate((r_dot, r_ddot))

    return dydt


def light_time_correction(JD_UTC, r_0, v_0, station):
    '''Light time correction for satellite position and velocity
    Inputs:
        JD_UTC: Julian date in UTC
        r_0: satellite position vector in ECI at time t
        v_0: satellite velocity vector in ECI at time t
        station: station position vector in ECEF at time t
    
    Outputs:
        r_0: satellite position vector in ECI at time t - lt
        v_0: satellite velocity vector in ECI at time t - lt
    '''

    c = 299792458 #m/s
    ECI_station = ECEF2ECI(station, JD_UTC, leap_sec, x_p[0], y_p[0], del_UT1[0])
    rho_station = np.linalg.norm(r_0 - ECI_station)
    lt = rho_station/c
    tol = 1e-3 #m
    delta = 1
    old_X_lt = np.zeros(6)
    y0 = np.concatenate((r_0, v_0))
    new_X_lt = y0
    while delta > tol:
        old_X_lt = new_X_lt
        t = JD_UTC - lt/86400
        sol = solve_ivp(satellite_motion, [lt, 0], y0, args=(JD_UTC,), rtol=3E-14, atol=1E-16)
        
        new_station = ECEF2ECI(station, t, leap_sec, x_p[0], y_p[0], del_UT1[0])
        # print(new_station)
        new_X_lt = sol.y.T[-1]
        new_rho = np.linalg.norm(new_X_lt[0:3] - new_station)
        lt = new_rho/c
        delta = np.linalg.norm(new_X_lt[0:3] - old_X_lt[0:3])
        
    return new_X_lt

def A_Matrix(drag=True, gravity=True, solar=True, third_body=True):
    '''Calculates the A matrix for the equations of motion

    Inputs:
    drag: boolean, if True, drag is included in the equations of motion
    gravity: boolean, if True, gravity is included in the equations of motion
    solar: boolean, if True, solar radiation pressure is included in the equations of motion
    third_body: boolean, if True, third body perturbations are included in the equations of motion

    Outputs:
    A: A_Matrix function
    '''

    #base equation of motion
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    z = sym.Symbol('z')
    A_Cross= sym.Symbol('A_Cross')
    A_Cross_Sol = sym.Symbol('A_Cross_Sol')
    x_dot = sym.Symbol('x_dot')
    y_dot = sym.Symbol('y_dot')
    z_dot = sym.Symbol('z_dot')
    C_D = sym.Symbol('C_D')
    r = (x**2 + y**2 + z**2)**(1/2)
    mu = 398600.4415*1000**3 #m^3/s^2
    
    #with gravity
    if gravity:
        R_earth = 6378.1363*1000 #[m]
        J_2 = 0.00108248
        phi = z/r
        F_x = sym.diff(mu/r*(1-J_2*(R_earth/r)**2*(3/2*phi**2-1/2)), x)
        F_y = sym.diff(mu/r*(1-J_2*(R_earth/r)**2*(3/2*phi**2-1/2)), y)
        F_z = sym.diff(mu/r*(1-J_2*(R_earth/r)**2*(3/2*phi**2-1/2)), z)

    
    #with atmospheric drag
    if drag:

        R_earth = 6378.1363*1000 #[m]
        m = 2000 #[kg]
        theta_dot = 7.292115146706979E-5 #[rad/s]
        rho_0 = 3.614E-13 #[kg/m^3]
        H = 88667.0 #[m]
        r0 = (700000.0 + R_earth) #[m]

        rho_A = rho_0*sym.exp(-(r-r0)/H)
        
        V_A_bar = sym.Matrix([x_dot+theta_dot*y, y_dot-theta_dot*x, z_dot])
        V_A = sym.sqrt((x_dot + theta_dot*y)**2 + (y_dot-theta_dot*x)**2 + z_dot**2)
        
        r_ddot = -1/2*C_D*A_Cross/m*rho_A*V_A*V_A_bar
        F_x += r_ddot[0]
        F_y += r_ddot[1]
        F_z += r_ddot[2]

    #with solar radiation pressure
    if solar:
        r_sun = sym.MatrixSymbol('r_sun', 1, 3)

        AU = 149597870700 #[m]
        m = 2000 #kg
        c = 299792458 #m/s
        d = ((r_sun[0]+x)**2 + (r_sun[1]+y)**2 + (r_sun[2]+z)**2)**(1/2)
        phi = 1367 #W/m^2
        C1 = phi/c 
        C_s = 0.04
        C_d = 0.04
        v = 1/3*C_d
        mu = 1/2*C_s
        theta = 0
        
        B = 2*v*sym.cos(theta)+4*mu*sym.cos(theta)**2
        F_x += -C1/(d/AU)**2*(B + (1-mu)*sym.cos(theta))*A_Cross_Sol/m*(r_sun[0]+x)/d
        F_y += -C1/(d/AU)**2*(B + (1-mu)*sym.cos(theta))*A_Cross_Sol/m*(r_sun[1]+y)/d
        F_z += -C1/(d/AU)**2*(B + (1-mu)*sym.cos(theta))*A_Cross_Sol/m*(r_sun[2]+z)/d

    #with thrird body perturbations
    if third_body:
        mu_sun = 132712440018*1000**3 #[m^3/s^2]
        mu_moon = 4902.800066*1000**3 #[m^3/s^2]
        r_sun = sym.MatrixSymbol('r_sun', 1, 3)
        r_moon = sym.MatrixSymbol('r_moon', 1, 3)
        r_sun_mag = (r_sun[0]**2 + r_sun[1]**2 + r_sun[2]**2)**(1/2)
        r_moon_mag = (r_moon[0]**2 + r_moon[1]**2 + r_moon[2]**2)**(1/2)
        
        del_sun_mag = ((r_sun[0]+x)**2 + (r_sun[1]+y)**2 + (r_sun[2]+z)**2)**(1/2)
        del_moon_mag = ((r_moon[0]+x)**2 + (r_moon[1]+y)**2 + (r_moon[2]+z)**2)**(1/2)
        F_x += mu_sun*((r_sun[0]+x)/(del_sun_mag)**3 - r_sun[0]/r_sun_mag**3) + mu_moon*((r_moon[0]+x)/(del_moon_mag**3) - r_moon[0]/r_moon_mag**3)
        F_y += mu_sun*((r_sun[1]+y)/(del_sun_mag)**3 - r_sun[1]/r_sun_mag**3) + mu_moon*((r_moon[1]+y)/(del_moon_mag**3) - r_moon[1]/r_moon_mag**3)
        F_z += mu_sun*((r_sun[2]+z)/(del_sun_mag)**3 - r_sun[2]/r_sun_mag**3) + mu_moon*((r_moon[2]+z)/(del_moon_mag**3) - r_moon[2]/r_moon_mag**3)

    #F functions
    F1 = x_dot
    F2 = y_dot
    F3 = z_dot
    F4, F5, F6 = F_x, F_y, F_z
    F7 = 0

    #A matrix
    A = [[sym.diff(F1, x), sym.diff(F1, y), sym.diff(F1, z), sym.diff(F1, x_dot), sym.diff(F1, y_dot), sym.diff(F1, z_dot), sym.diff(F1, C_D)],
        [sym.diff(F2, x), sym.diff(F2, y), sym.diff(F2, z), sym.diff(F2, x_dot), sym.diff(F2, y_dot), sym.diff(F2, z_dot), sym.diff(F2, C_D)],
        [sym.diff(F3, x), sym.diff(F3, y), sym.diff(F3, z), sym.diff(F3, x_dot), sym.diff(F3, y_dot), sym.diff(F3, z_dot), sym.diff(F3, C_D)],
        [sym.diff(F4, x), sym.diff(F4, y), sym.diff(F4, z), sym.diff(F4, x_dot), sym.diff(F4, y_dot), sym.diff(F4, z_dot), sym.diff(F4, C_D)],
        [sym.diff(F5, x), sym.diff(F5, y), sym.diff(F5, z), sym.diff(F5, x_dot), sym.diff(F5, y_dot), sym.diff(F5, z_dot), sym.diff(F5, C_D)],
        [sym.diff(F6, x), sym.diff(F6, y), sym.diff(F6, z), sym.diff(F6, x_dot), sym.diff(F6, y_dot), sym.diff(F6, z_dot), sym.diff(F6, C_D)],
        [sym.diff(F7, x), sym.diff(F7, y), sym.diff(F7, z), sym.diff(F7, x_dot), sym.diff(F7, y_dot), sym.diff(F7, z_dot), sym.diff(F7, C_D)]]
    
    if gravity and drag and solar and third_body:
     A = sym.lambdify([x, y, z, x_dot, y_dot, z_dot, C_D, A_Cross, A_Cross_Sol, r_sun, r_moon], A)
     
    elif gravity:
        A = sym.lambdify([x, y, z], A)

    elif drag:
        A = sym.lambdify([x, y, z, x_dot, y_dot, z_dot, C_D, A_Cross], A)

    elif solar:
        A = sym.lambdify([x, y, z, A_Cross_Sol, d, m, theta], A)

    elif third_body:
        A = sym.lambdify([x, y, z, r_sun, r_moon], A)

    else:
        A = sym.lambdify([x, y, z], A)

    return A

#H_tilde
def H_tilde_matrix():
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    z = sym.Symbol('z')
    x_dot = sym.Symbol('x_dot')
    y_dot = sym.Symbol('y_dot')
    z_dot = sym.Symbol('z_dot')
    C_D = sym.Symbol('C_D')

    x_s = sym.Symbol('x_s')
    y_s = sym.Symbol('y_s')
    z_s = sym.Symbol('z_s')

    
    rho = sym.sqrt((x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2)
    #for project omega x r ECEF frame
    #vallado chapter 4 ECEF to ECI transformation
    omega_earth = np.array([0, 0, 7.292115146706979E-5]) #rad/s
    station_dot = np.cross(np.array([x_s, y_s, z_s]), omega_earth)
    rho_dot = ((x-x_s)*(x_dot-station_dot[0]) + (y-y_s)*(y_dot-station_dot[1]) + (z-z_s)*(z_dot-station_dot[2]))/rho

    H_tilde_sym = [[sym.diff(rho, x), sym.diff(rho, y), sym.diff(rho, z), sym.diff(rho, x_dot), sym.diff(rho, y_dot), sym.diff(rho, z_dot), sym.diff(rho, C_D)],[
           sym.diff(rho_dot, x), sym.diff(rho_dot, y), sym.diff(rho_dot, z), sym.diff(rho_dot, x_dot), sym.diff(rho_dot, y_dot), sym.diff(rho_dot, z_dot), sym.diff(rho_dot, C_D)]]
    
    H_tilde_func = sym.lambdify((x, y, z, x_dot, y_dot, z_dot, x_s, y_s, z_s, C_D), H_tilde_sym, 'numpy')
    
    return H_tilde_func

def a_third_body(r, r_sun, r_moon):
    '''
    Calculates the acceleration due to third body perturbations

    Inputs:
        r: position vector of satellite
        r_sun: position vector of sun
        r_moon: position vector of moon

    Outputs:
        a_x: acceleration in x direction
        a_y: acceleration in y direction
        a_z: acceleration in z direction
    '''

    x = r[0]
    y = r[1]
    z = r[2]
    mu_sun = 32712440018*1000**3 #m^3/s^2
    mu_moon = 4902.800066*1000**3 #m^3/s^2
    r_sun_mag = np.linalg.norm(r_sun)
    r_moon_mag = np.linalg.norm(r_moon)

    del_sun_mag = ((r_sun[0]+x)**2 + (r_sun[1]+y)**2 + (r_sun[2]+z)**2)**(1/2)
    del_moon_mag = ((r_moon[0]+x)**2 + (r_moon[1]+y)**2 + (r_moon[2]+z)**2)**(1/2)
    a_x = mu_sun*((r_sun[0]+x)/(del_sun_mag)**3 - r_sun[0]/r_sun_mag**3) + mu_moon*((r_moon[0]+x)/(del_moon_mag**3) - r_moon[0]/r_moon_mag**3)
    a_y = mu_sun*((r_sun[1]+y)/(del_sun_mag)**3 - r_sun[1]/r_sun_mag**3) + mu_moon*((r_moon[1]+y)/(del_moon_mag**3) - r_moon[1]/r_moon_mag**3)
    a_z = mu_sun*((r_sun[2]+z)/(del_sun_mag)**3 - r_sun[2]/r_sun_mag**3) + mu_moon*((r_moon[2]+z)/(del_moon_mag**3) - r_moon[2]/r_moon_mag**3)

    return np.array([a_x, a_y, a_z])

def a_solar(r, s, C_s, C_d, A_Cross_sol):
    '''
    Calculates the acceleration due to solar radiation pressure

    Inputs:
        r: position vector of satellite
        s: position vector of sun
        C_s: solar radiation pressure coefficient
        C_d: solar radiation pressure coefficient
        A_Cross_sol: cross sectional area of satellite
    
    Outputs:
        r_ddot: acceleration vector of satellite
    '''
    
    r_ddot = np.zeros(3)
    tau_min = (np.linalg.norm(r)**2 - np.dot(r, s))/(np.linalg.norm(r)**2 + np.linalg.norm(s)**2 - 2*np.dot(r, s))
    
    if tau_min < 0:
        m = 2000 #kg
        c = 299792458 #m/s
        AU = 149597870.7*1000 #m
        d = np.linalg.norm(s+r)/AU #distance from sun

        phi = 1367 #W/m^2
        C1 = phi/c 
        v = 1/3*C_d
        mu = 1/2*C_s
        theta = 0
        B = 2*v*np.cos(theta)+4*mu*np.cos(theta)**2
        u = (s+r)/np.linalg.norm(s+r)
        
        r_ddot = (-C1/d**2*(B + (1-mu)*np.cos(theta))*A_Cross_sol/m)*u
   
    return r_ddot

def a_drag(C_D, r, v, A_Cross):

    '''
    Computes the acceleration due to atmospheric drag

    Inputs:
    r - position vector in ECI frame [m]
    v - velocity vector in ECI frame [m/s]
    A_Cross - cross sectional area of satellite [m^2]

    Outputs:
    a_drag - acceleration due to atmospheric drag [m/s^2]
    
    '''

    r_mag = np.linalg.norm(r)
    #drag parameters
    R_earth = 6378.1363*1000 #[m]
    m = 2000 #[kg]
    theta_dot = 7.292115146706979E-5 #[rad/s]
    rho_0 = 3.614E-13 #[kg/m^3]
    H = 88667.0 #[m]
    r0 = (700000.0 + R_earth) #[m]
    rho_A = rho_0*np.exp(-(r_mag-r0)/H)
    
    V_A_bar = np.array([v[0]+theta_dot*r[1], v[1]-theta_dot*r[0], v[2]])
    # print(V_A_bar)
    V_A = np.sqrt((v[0] + theta_dot*r[1])**2 +(v[1]-theta_dot*r[0])**2 + v[2]**2)

    a_drag = -1/2*C_D*A_Cross/m*rho_A*V_A*V_A_bar
    return a_drag

def a_gravity_J2(r):
    '''
    Computes the acceleration due to J2 perturbation
    
    Inputs:
    r - position vector in ECI frame [m]
    
    Outputs:
    a_gravity_J2 - acceleration due to J2 perturbation [m/s^2]
    '''
    x = r[0]
    y = r[1]
    z = r[2]
    J_2 = 0.00108248
    R_earth = 6378.1363*1000
    mu = 398600.4415*1000**3
    dUdx = -1.0*mu*x*(-J_2*R_earth**2*(1.5*z**2/(x**2 + y**2 + z**2)**1.0 - 0.5)/(x**2 + y**2 + z**2)**1.0 + 1)/(x**2 + y**2 + z**2)**1.5 \
        + mu*(3.0*J_2*R_earth**2*x*z**2/(x**2 + y**2 + z**2)**3.0 + 2.0*J_2*R_earth**2*x*(1.5*z**2/(x**2 + y**2 + z**2)**1.0 - 0.5)/(x**2 \
        + y**2 + z**2)**2.0)/(x**2 + y**2 + z**2)**0.5
    dUdy = -1.0*mu*y*(-J_2*R_earth**2*(1.5*z**2/(x**2 + y**2 + z**2)**1.0 - 0.5)/(x**2 + y**2 + z**2)**1.0 + 1)/(x**2 + y**2 + z**2)**1.5 \
        + mu*(3.0*J_2*R_earth**2*y*z**2/(x**2 + y**2 + z**2)**3.0 + 2.0*J_2*R_earth**2*y*(1.5*z**2/(x**2 + y**2 + z**2)**1.0 - 0.5)/(x**2 \
        + y**2 + z**2)**2.0)/(x**2 + y**2 + z**2)**0.5
    dUdz = -1.0*mu*z*(-J_2*R_earth**2*(1.5*z**2/(x**2 + y**2 + z**2)**1.0 - 0.5)/(x**2 + y**2 + z**2)**1.0 + 1)/(x**2 + y**2 + z**2)**1.5\
        + mu*(2.0*J_2*R_earth**2*z*(1.5*z**2/(x**2 + y**2 + z**2)**1.0 - 0.5)/(x**2 + y**2 + z**2)**2.0 - J_2*R_earth**2*(-3.0*z**3/(x**2 \
            + y**2 + z**2)**2.0 + 3.0*z/(x**2 + y**2 + z**2)**1.0)/(x**2 + y**2 + z**2)**1.0)/(x**2 + y**2 + z**2)**0.5
    
    a_gravity_J2 = np.array([dUdx, dUdy, dUdz])

    return a_gravity_J2

def load_egm96_coefficients():
    '''
    Loads the EGM96 coefficients from the CSV files
    
    Inputs:
    None
    
    Outputs:
    C - EGM96 C coefficients
    S - EGM96 S coefficients
    '''

    EGM96_C_file = 'EGM96_C.csv'
    EGM96_S_file = 'EGM96_S.csv'

    # Load EGM96 coefficients from file
    C = []
    S = []
    # Read in the CSV file and populate the matrix
    with open(EGM96_C_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            C.append(row)
    # S = np.loadtxt(EGM96_S_file)
    with open(EGM96_S_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            S.append(row)
    C = np.array(C).astype(float)
    S = np.array(S).astype(float)
    return C, S

C, S = load_egm96_coefficients()

def loc_gravLegendre(phi, maxdeg):

    '''
    This function computes the fully normalized associated Legendre functions
    and their derivatives up to degree and order maxdeg at latitude phi.

    input:
        phi: latitude [rad]
        maxdeg: maximum degree and order of the spherical harmonic expansion

    output:
        P: fully normalized associated Legendre functions
        scaleFactor: scaling factor for the fully normalized associated Legendre functions
        
    '''
    # Initialize arrays
    P = np.zeros((maxdeg+3, maxdeg+3, 1))
    scaleFactor = np.zeros((maxdeg+3, maxdeg+3, 1))
    cphi = np.cos(np.pi/2-phi)
    sphi = np.sin(np.pi/2-phi)

    # Force numerically zero values to be exactly zero
    if np.abs(cphi) <= np.finfo(float).eps:
        cphi = 0
    if np.abs(sphi) <= np.finfo(float).eps:
        sphi = 0

    # Seeds for recursion formula
    P[0,0,:] = 1     # n = 0, m = 0;
    P[1,0,:] = np.sqrt(3)*cphi   # n = 1, m = 0;
    scaleFactor[0,0,:] = 0
    scaleFactor[1,0,:] = 1
    P[1,1,:] = np.sqrt(3)*sphi   # n = 1, m = 1;
    scaleFactor[1,1,:] = 0

    for n in range(2, maxdeg+3):
        k = n + 1
        for m in range(0, n+1):
            p = m + 1
            # Compute normalized associated legendre polynomials, P, via recursion relations 
            # Scale Factor needed for normalization of dUdphi partial derivative
            if n == m:
                P[k-1,k-1,:] = np.sqrt(2*n+1)/np.sqrt(2*n)*sphi*P[k-2,k-2,:]
                scaleFactor[k-1,k-1,:] = 0
            elif m == 0:
                P[k-1,p,:] = (np.sqrt(2*n+1)/n)*(np.sqrt(2*n-1)*cphi*P[k-2,p,:] - (n-1)/np.sqrt(2*n-3)*P[k-3,p,:])
                scaleFactor[k-1,p,:] = np.sqrt((n+1)*n/2)
            else:
                P[k-1,p,:] = np.sqrt(2*n+1)/(np.sqrt(n+m)*np.sqrt(n-m))*(np.sqrt(2*n-1)*cphi*P[k-2,p,:] - np.sqrt(n+m-1)*np.sqrt(n-m-1)/np.sqrt(2*n-3)*P[k-3,p,:])
                scaleFactor[k-1,p,:] = np.sqrt((n+m+1)*(n-m))

    return P, scaleFactor

def loc_gravityPCPF(p, maxdeg, P, C, S, smlambda, cmlambda, GM, Re, r, scaleFactor):
    '''
    Computes the gravity acceleration in the ECEF frame

    input:
        p: position vector in ECEF frame [m]
        maxdeg: maximum degree and order of the spherical harmonic expansion
        P: fully normalized associated Legendre functions
        C: cosine spherical harmonic coefficients
        S: sine spherical harmonic coefficients
        smlambda: sine of the product of the longitude and degree
        cmlambda: cosine of the product of the longitude and degree
        GM: gravitational constant times the mass of the Earth [m^3/s^2]
        Re: mean radius of the Earth [m]
        r: magnitude of the position vector [m]
        scaleFactor: scaling factor for the fully normalized associated Legendre functions

    output:
        gx: x gravity acceleration in the ECEF frame [m/s^2]
        gy: y gravity acceleration in the ECEF frame [m/s^2]
        gz: z gravity acceleration in the ECEF frame [m/s^2]
    '''
    
    rRatio   = Re/r
    rRatio_n = rRatio.copy()
    
    # initialize summation of gravity in radial coordinates
    dUdrSumN      = 1
    dUdphiSumN    = 0
    dUdlambdaSumN = 0
    # summation of gravity in radial coordinates
    for n in range(2, maxdeg+1):
        k = n+1
        rRatio_n      = rRatio_n*rRatio
        dUdrSumM      = 0
        dUdphiSumM    = 0
        dUdlambdaSumM = 0
        for m in range(n+1):
            j = m
            dUdrSumM      = dUdrSumM + P[k-1,j]*(C[k-1,j]*cmlambda[:,j] + S[k-1,j]*smlambda[:,j])
            dUdphiSumM    = dUdphiSumM + ((P[k-1,j+1]*scaleFactor[k-1,j,:] - p[2]/np.sqrt(p[0]**2 + p[1]**2)*m*P[k-1,j])*(C[k-1,j]*cmlambda[:,j] + S[k-1,j]*smlambda[:,j]))
            dUdlambdaSumM = dUdlambdaSumM + m*P[k-1,j]*(S[k-1,j]*cmlambda[:,j] - C[k-1,j]*smlambda[:,j])
        dUdrSumN      = dUdrSumN      + dUdrSumM*rRatio_n*k
        dUdphiSumN    = dUdphiSumN    + dUdphiSumM*rRatio_n
        dUdlambdaSumN = dUdlambdaSumN + dUdlambdaSumM*rRatio_n
    
    # gravity in spherical coordinates
    dUdr      = -GM/(r**2)*dUdrSumN
    dUdphi    =  GM/r*dUdphiSumN
    dUdlambda =  GM/r*dUdlambdaSumN
    
    # gravity in ECEF coordinates
    gx = ((1/r)*dUdr - (p[2]/(r**2*np.sqrt(p[0]**2 + p[1]**2)))*dUdphi)*p[0] \
          - (dUdlambda/(p[0]**2 + p[1]**2))*p[1]
    gy = ((1/r)*dUdr - (p[2]/(r**2*np.sqrt(p[0]**2 + p[1]**2)))*dUdphi)*p[1] \
          + (dUdlambda/(p[0]**2 + p[1]**2))*p[0]
    gz = (1.0/r)*dUdr*p[2] + ((np.sqrt(p[0]*p[0] + p[1]*p[1]))/(r*r))*dUdphi

    # special case for poles
    atPole = np.abs(np.arctan2(p[2], np.sqrt(p[0]*p[0] + p[1]*p[1]))) == np.pi/2
    if np.any(atPole):
        gx[atPole] = 0
        gy[atPole] = 0
        gz[atPole] = (1.0/r[atPole])*dUdr[atPole]*p[atPole,2]
    # print(gx, gy, gz)
    return gx[0], gy[0], gz[0]

def F_gravity_vallado(p):
    ''' 
    # F_GRAVITY_VALLADO computes the acceleration due to gravity in the
    # Earth-Centered Inertial (ECI) frame using the Vallado algorithm.

    # Inputs:
    #   p_ECI - Nx3 array of ECI positions [m]

    # Outputs:
    #   g_ECEF - Nx3 array of ECI accelerations [m/s^2]'''
   

    maxdeg = 2
    mu = 3.986004418e14 # m^3/s^2
    Re = 6378.145*1000 #[m] # m
    r = np.linalg.norm(p)
    
    # Compute geocentric latitude
    phic = np.arcsin(p[2] / r)

    # Compute lambda
    lambda_ = np.arctan2(p[1], p[0])

    smlambda = np.zeros((p.shape[0], maxdeg+1))
    cmlambda = np.zeros((p.shape[0], maxdeg+1))

    slambda = np.sin(lambda_)
    clambda = np.cos(lambda_)
    smlambda[:,0] = 0
    cmlambda[:,0] = 1
    smlambda[:,1] = slambda
    cmlambda[:,1] = clambda

    for m in range(2, maxdeg+1):
        smlambda[:,m] = 2.0 * clambda * smlambda[:, m-1] - smlambda[:, m-2]
        cmlambda[:,m] = 2.0 * clambda * cmlambda[:, m-1] - cmlambda[:, m-2]

    # compute normalized legendre polynomials
    P, scaleFactor = loc_gravLegendre(phic, maxdeg)
    # print(P.shape)

    # Compute gravity in ECEF coordinates
    gx, gy, gz = loc_gravityPCPF(p, maxdeg, P, C[0:maxdeg+1, 0:maxdeg+1],
                                S[0:maxdeg+1, 0:maxdeg+1], smlambda,
                                cmlambda, mu, Re, r, scaleFactor)
    
    g_ECEF = np.array([gx, gy, gz])
    return g_ECEF

def range_range_rate(r, v, station):
    '''
    Compute range and range rate from a station to a satellite
    
    Inputs:
        r - position vector of satellite in ECI frame [m]
        v - velocity vector of satellite in ECI frame [m/s]
        station - position vector of station in ECI frame [m]
    
    Outputs:
        rho - range from station to satellite [m]
        rho_dot - range rate from station to satellite [m/s]
    '''

    omega_earth = np.array([0, 0, 7.292115146706979E-5]) #rad/s
    station_dot = np.cross(omega_earth, station)
    rho = np.linalg.norm(r - station)
    rho_dot = np.dot(r - station, v - station_dot)/rho

    return rho, rho_dot