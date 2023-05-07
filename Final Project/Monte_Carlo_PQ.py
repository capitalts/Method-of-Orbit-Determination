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

   #read in LEO_DATA_APPARENT.csv
    leo_app_3d = "LEO_DATA_Apparent_3Days.csv"
    leo_app_4_6 = "LEO_DATA_Apparent_Days4-6.csv"

    #observations column names
    column_names = ['id', 'time', 'range', 'range_rate']
    seconds = 86400
    # cases = ['C', 'D', 'E', 'F', 'G']
    cases = ['C']


    #initial state
    r_ECI = np.array([6984.45711518852, 1612.2547582643, 13.0925904314402])*1000 #m

    v_ECI = np.array([-1.67667852227336, 7.26143715396544, 0.259889857225218])*1000 #m/s


    #station locations
    stations = np.array([[-6143584, 1364250, 1033743], 
                        [1907295, 6030810, -817119],
                        [2390310, -5564341 , 1994578]]) #m

    JD_UTC = gregorian_to_jd(2018, 3, 23, 8, 55, 3)
    bias_k = 5
    bias_dg = -5
    bias_a = 10
    for case in cases:
    #observations dataframe
        obs_df = pd.read_csv(leo_app_3d, names=column_names)
        obs_df_4_6 = pd.read_csv(leo_app_4_6, names=column_names)
        obs_df_4_6['time'] += 3*86400
        obs_df = pd.concat([obs_df, obs_df_4_6], ignore_index=True)
        # with open('Results/Case' + case + '/range_rate_RMS_case'  + case + '_PQ.csv', 'w') as csvfile:
        #                 writer = csv.writer(csvfile, delimiter=',',
        #                                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #                 writer.writerow(['KR', 'KRR', 'DR', 'DRR', 'AR', 'ARR', 'Ps', 'Psv', 'Qs', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        P_sigma = np.arange(4, 6, 1)
        P_sigma_v= np.arange(0, 2, 1)
        Q_sigma = np.arange(-4, -6, -1)
        for Ps in P_sigma:
            for Psv in P_sigma_v:
                for Qs in Q_sigma:
                    print("Ps:", Ps, "Psv:", Psv, "Qs:", Qs)
                    y_arr, P_hat, P_bar, P_ZZ, obs_df, Q, rho, rho_dot, phi = EKF(case, r_ECI, v_ECI, obs_df, stations, JD_UTC, seconds, 10.0**Ps, 10.0**Psv, 10.0**Qs, bias_k, bias_dg, bias_a)


                    #write EKF results to csv
                    obs_time = obs_df['time'].values
                    obs_id = obs_df['id'].values
                    m = obs_df['time'].index[obs_time < seconds].values[-1]
                    # print(m)
                    t = obs_df['time'][0:m].values

                    with open('range_rate_case' + case + '.csv', 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for i in range(0, len(t)):
                            station_index = obs_id[i]-1
                            writer.writerow([station_index+1, obs_time[i], rho[i]/1000, rho_dot[i]/1000])

                    
                    RMS_Kwaj_r = 0
                    RMS_Kwaj_rr = 0
                    RMS_Diego_r = 0
                    RMS_Diego_rr = 0
                    RMS_Arecibo_r = 0
                    RMS_Arecibo_rr = 0
                    print('Plotting')
                    #postfit RMS
                    df_calc = pd.read_csv('range_rate_case' + case + '.csv', names=column_names)
                    #Comparing Pre-fit residuals
                    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

                    if case == 'A' or case == 'B' or case == 'C' or case == 'F' or case == 'G':
                        r_calc_Kwaj = df_calc['range'][df_calc.index[df_calc['id']==1]]
                        rr_calc_Kwaj = df_calc['range_rate'][df_calc.index[df_calc['id']==1]]

                        r_obs_Kwaj = obs_df.iloc[:m]['range'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==1]]
                        rr_obs_Kwaj = obs_df.iloc[:m]['range_rate'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==1]]

                        RMS_Kwaj_r = np.sqrt(np.mean((r_obs_Kwaj - r_calc_Kwaj)**2))
                        RMS_Kwaj_rr = np.sqrt(np.mean((rr_obs_Kwaj - rr_calc_Kwaj)**2))

                        t_kwaj = obs_df.iloc[:m]['time'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==1]]

                        print("Kwaj Range RMS:", RMS_Kwaj_r, "[km]")
                        print("Kwaj Range Rate RMS:", RMS_Kwaj_rr,"[km/s]")

                        ax[0].scatter(t_kwaj, r_obs_Kwaj - r_calc_Kwaj, label='Kwaj Range Residuals')
                        ax[1].scatter(t_kwaj, rr_obs_Kwaj - rr_calc_Kwaj, label='Kwaj Range Rate Residuals')

                    if case == 'A' or case == 'B' or case == 'D' or case == 'F' or case == 'G':

                        r_calc_Diego = df_calc['range'][df_calc.index[df_calc['id']==2]]
                        rr_calc_Diego = df_calc['range_rate'][df_calc.index[df_calc['id']==2]]

                        r_obs_Diego = obs_df.iloc[:m]['range'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==2]]
                        rr_obs_Diego = obs_df.iloc[:m]['range_rate'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==2]]

                        RMS_Diego_r = np.sqrt(np.mean((r_obs_Diego - r_calc_Diego)**2))
                        RMS_Diego_rr = np.sqrt(np.mean((rr_obs_Diego - rr_calc_Diego)**2))

                        t_diego = obs_df.iloc[:m]['time'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==2]]

                        print("Diego Range RMS:", RMS_Diego_r, "[km]")
                        print("Diego Range Rate RMS:", RMS_Diego_rr,"[km/s]")

                        ax[0].scatter(t_diego, r_obs_Diego - r_calc_Diego, label='Diego Range Residuals')
                        ax[1].scatter(t_diego, rr_obs_Diego - rr_calc_Diego, label='Diego Range Rate Residuals')


                    if case == 'A' or case == 'B' or case == 'E' or case == 'F' or case == 'G':

                        r_calc_Arecibo = df_calc['range'][df_calc.index[df_calc['id']==3]]
                        rr_calc_Arecibo = df_calc['range_rate'][df_calc.index[df_calc['id']==3]]


                        r_obs_Arecibo = obs_df.iloc[:m]['range'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==3]]
                        rr_obs_Arecibo = obs_df.iloc[:m]['range_rate'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==3]]

                        RMS_Arecibo_r = np.sqrt(np.mean((r_obs_Arecibo - r_calc_Arecibo)**2))
                        RMS_Arecibo_rr = np.sqrt(np.mean((rr_obs_Arecibo - rr_calc_Arecibo)**2))

                        t_arecibo = obs_df.iloc[:m]['time'][obs_df.iloc[:m].index[obs_df.iloc[:m]['id']==3]]

                        print("Arecibo Range RMS:", RMS_Arecibo_r, "[km]")
                        print("Arecibo Range Rate RMS:", RMS_Arecibo_rr,"[km/s]")

                        ax[0].scatter(t_arecibo, r_obs_Arecibo - r_calc_Arecibo, label='Arecibo Range Residuals')
                        ax[1].scatter(t_arecibo, rr_obs_Arecibo - rr_calc_Arecibo, label='Arecibo Range Rate Residuals')


                    three_sigma_pos = np.array([np.sqrt(P_ZZ[i][0]/1000**2)*3 for i in range(len(t))])
                    three_sigma_vel = np.array([np.sqrt(P_ZZ[i][3]/1000**2)*3 for i in range(len(t))])

                    # print(three_sigma_pos)
                    ax[0].plot(t, three_sigma_pos, label='+3$\sigma$')
                    ax[0].plot(t, -three_sigma_pos, label='-3$\sigma$')
                    ax[1].plot(t, three_sigma_vel, label='+3$\sigma$')
                    ax[1].plot(t, -three_sigma_vel, label='-3$\sigma$')



                    ax[0].set_title('Range Residuals')
                    ax[0].set_xlabel('Time [s]')
                    ax[0].set_ylabel('Range Residuals [km]')
                    ax[0].set_ylim(-0.25, 0.25)
                    ax[0].legend()


                    ax[1].set_title('Range Rate Residuals')
                    ax[1].set_xlabel('Time [s]')
                    ax[1].set_ylabel('Range Rate Residuals [km/s]')
                    ax[1].set_ylim(-0.0005, 0.0005)
                    ax[1].legend()
                    fig.savefig('Results/Case' + case + '/Plots/residuals_case' + case + '_' + str(Ps) + '_' + str(Psv) + '_' + str(Qs) + '.png', dpi=300)
                    plt.close()
                    F = A_Matrix()
                    del_t = seconds-t[-1]
                    phi = np.eye(6)
                    t_eval = np.arange(t[-1], 86460, 60)
                    y0 = np.concatenate((y_arr[-1][0:6], phi.ravel()))
                    print('Propagating final state')
                    sol_final = solve_ivp(satellite_motion_phi, [t[-1], 86400], y0, t_eval=t_eval, args=(F, JD_UTC), rtol=3E-14, atol=1E-16)
                    y_final = sol_final.y[0:6, -1]
                    with open('Results/Case' + case + '/range_rate_RMS_case'  + case + '_PQ.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([RMS_Kwaj_r, RMS_Kwaj_rr, RMS_Diego_r, RMS_Diego_rr, RMS_Arecibo_r, RMS_Arecibo_rr, Ps, Psv, Qs, y_final[0], y_final[1], y_final[2], y_final[3], y_final[4], y_final[5]])


            
if __name__ == "__main__":
    main()
