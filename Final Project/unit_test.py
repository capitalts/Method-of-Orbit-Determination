import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.integrate import odeint, solve_ivp
from scipy.special import lpmn
from scipy.io import savemat
import pandas as pd
from datetime import datetime
import sympy as sym
import csv
from filterpy.kalman import ExtendedKalmanFilter
from fp_utils import *

def main():

    #constants
    mu = 398600.4415 #m^3/s^2
    # R_earth = 6378.137 #km

    e_earth = 0.0818191908426215
    omega_earth = 7.2921158553E-5 #rad/s
    AU = 149597870.6 #km

    #initial state
    # r_ECI = np.array([6984.45711518852, 1612.2547582643, 13.0925904314402])*1000 #m

    # v_ECI = np.array([-1.67667852227336, 7.26143715396544, 0.259889857225218])*1000 #m/s

    r_ECI = np.array([6978.883860074742, 1615.7622342575391, 19.48509793007517])
    v_ECI = np.array([-1.6622408967796873, 7.26097148769649, 0.2705834331704228])



    #station locations
    stations = np.array([[-6143.584, 1364.250, 1033.743], 
                        [1907.295, 6030.810, -817.119],
                        [2390.310, -5564.341, 1994.578]]) #m

    #time
    JD_UTC = gregorian_to_jd(2018, 3, 23, 8, 55, 3)
    leap_sec = 37 #s
    x_p = np.array([20.816, 22.156, 23.439, 24.368, 25.676, 26.952, 28.108])/1000 #arcsec
    y_p = np.array([381.008, 382.613, 384.264, 385.509, 386.420, 387.394, 388.997])/1000 #arcsec
    del_UT1 = np.array([144.0585, 143.1048, 142.2335, 141.3570, 140.4078, 139.3324, 138.1510])/1000 #s
    LOD = np.array([1.0293,0.9174, 0.8401, 0.8810, 1.0141, 1.1555, 1.2568])/1000 #s
    C, S = load_egm96_coefficients()

    #test gravity model
    x_p, y_p, del_UT1, LOD = interp_EOP(JD_UTC)
    leap_sec = 37
    r_ECEF, _= ECI2ECEF(r_ECI, None, JD_UTC, x_p, y_p, leap_sec, del_UT1, LOD)
    # print('r_ECI', np.linalg.norm(r_ECI))
    print('r_ECEF', np.linalg.norm(r_ECEF))
    r_ddot_ECEF = grav_odp(r_ECEF)

if __name__ == "__main__":
    main()
