import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.io import savemat
from fp_utils import *


def eci_to_ric(r_ref, v_ref, r, cov):

    '''
    Transforms a state vector from ECI to RIC frame
    Inputs:
        r_ref: reference position vector
        v_ref: reference velocity vector
        r: position vector to be transformed
        cov: covariance matrix of position vector to be transformed
        
    Outputs:
        r_ric: position vector in RIC frame
        cov_ric: covariance matrix of position vector in RIC frame
    '''


   # Calculate the rotation matrix gamma
    u_R = r_ref / np.linalg.norm(r_ref) # Calculate unit vector in radial direction 
    u_N = np.cross(r_ref, v_ref) / np.linalg.norm(np.cross(r_ref, v_ref)) # Calculate unit vector in normal direction
    u_T = np.cross(u_N, u_R) # Calculate unit vector in tangential direction

    gamma = np.vstack((u_R, u_T, u_N)) # Create rotation matrix from unit vectors

    # Transform position and covariance matrix to radial-inertial frame
    r_ric = gamma @ (r_ref - r) # Transform position vector
    cov_ric = gamma @ cov @ gamma.T # Transform covariance matrix

    return r_ric, cov_ric # Return transformed position and covariance matrix


def plot_ellipses(x_mean, covariance, sig, ax):
    # Mean and Covariance 
    P_xx = covariance
    
    eigenval, eigenvec = np.linalg.eig(P_xx)
    
    # Get the index of the largest eigenvector
    max_evc_ind_c = np.argmax(eigenval)
    max_evc = eigenvec[:, max_evc_ind_c]
    
    # Get the largest eigenvalue
    max_evl = np.max(eigenval)
    min_evl = np.min(eigenval)
    
    
    # Calculate the angle between the x-axis and the largest eigenvector
    phi = np.arctan2(max_evc[1], max_evc[0])
    print('phi', phi)
    
    
    theta_grid = np.linspace(0, 2*np.pi, 100)

    X0  = x_mean[0]
    Y0  = x_mean[1]
    
    # Find semi-major and semi-minor axis for 1,2,3sigma

    a = np.sqrt(sig*max_evl)     # Semi-major axis
    b = np.sqrt(sig*min_evl)     # Semi- minor axis
    
    
    # the ellipse in x and y coordinates 
    ellipse_x_r  = a*np.cos(theta_grid)
    ellipse_y_r  = b*np.sin(theta_grid)
    
    # Define a rotation matrix (x',y') to (x,y)
    R = np.array([[np.cos(phi), -np.sin(phi)], 
                  [np.sin(phi), np.cos(phi)]])
    
    # let's rotate the ellipse to some angle phi
    r_1 = np.zeros((len(theta_grid), 2))

    for j in range(len(theta_grid)):
        r_1[j,:] = R @ np.array([ellipse_x_r[j],ellipse_y_r[j]])

    
    X1 = r_1.T.reshape(2,len(theta_grid))  

    ax.plot(X0 + X1[0,:], Y0 + X1[1,:], color='r', linestyle='--', linewidth=1)
   

    return X1

def plot_error_ellipses(y_final, cov):

    r_ref = np.array([])
    v_ref = np.array([])
    RIC_pred, RIC_cov = eci_to_ric(r_ref, v_ref, y_final[:3], cov[:3, :3])

    #plot
    ax, fig = plt.subplots(1, 3, figsize=(15, 5))

    radial = RIC_pred[0]
    in_track = RIC_pred[1]
    cross_track = RIC_pred[2]

    plot_ellipses(RIC_pred[0:2], RIC_cov[0:2, 0:2],3, ax[0])
    plot_ellipses([RIC_pred[0], RIC_pred[2]], np.array([[RIC_cov[0, 0], RIC_cov[0, 2]],[RIC_cov[2, 0], RIC_cov[2, 2]]]),3, ax[1])
    plot_ellipses(RIC_pred[1:3], RIC_cov[1:3, 1:3],3, ax[2])

    ax[0].plot(0, 0, 'o', label='Truth')
    # ax[0].plot(error_ellipse_RI[0], error_ellipse_RI[1], 'r', label='Error Ellipse')
    ax[0].plot(radial, in_track, 'ro', label='Predicted')

    ax[1].plot(radial, cross_track, 'bo', label='Predicted')
    ax[1].plot(0, 0, 'o', label='Truth')
    # ax[1].plot(error_ellipse_RC[0], error_ellipse_RC[1], 'r', label='Error Ellipse')

    ax[2].plot(in_track, cross_track, 'go', label='Predicted')
    ax[2].plot(0, 0, 'o', label='Truth')
    # ax[2].plot(error_ellipse_IC[0], error_ellipse_IC[1], 'r', label='Error Ellipse')


    ax[0].set_xlabel('Radial (km)')
    ax[0].set_ylabel('In-track (km)')
    ax[0].set_title('Radial vs In-track')
    # ax[0].set_xlim(-0.05, 0.05)
    # ax[0].set_ylim(-0.05, 0.05)
    ax[0].legend(loc = 'upper left')

    ax[1].set_xlabel('Radial (km)')
    ax[1].set_ylabel('Cross-track (km)')
    # ax[1].set_xlim(-0.05, 0.05)
    # ax[1].set_ylim(-0.05, 0.05)
    ax[1].set_title('Radial vs Cross-track')
    ax[1].legend(loc = 'upper left')

    ax[2].set_xlabel('In-track (km)')
    ax[2].set_ylabel('Cross-track (km)')
    ax[2].set_title('In-track vs Cross-track')
    # ax[2].set_xlim(-0.05, 0.05)
    # ax[2].set_ylim(-0.05, 0.05)
    ax[2].legend(loc = 'upper left')

def plot_residuals(case, obs_mat, residuals, res_type, t, P_ZZ):
    '''
    Saves the plot of the residuals for each station

    Inputs:
        case: string of the case letter
        obs_mat: data of the observations
        column_names: list of the column names for the observations
        t: time array
        P_ZZ: covariance matrix
        m: number of observations
    

    '''
    #postfit RMS
    # df_calc = pd.read_csv('range_rate_case' + case + '.csv', names=column_names)
    #Comparing Pre-fit residuals
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    if case == 'A' or case == 'B' or case == 'C' or case == 'F' or case == 'G':

        r_residuals_kwaj = residuals[np.where(obs_mat[:, 0] == 1)[0], 0]
        rr_residuals_kwaj = residuals[np.where(obs_mat[:, 0] == 1)[0], 1]

        RMS_Kwaj_r = np.sqrt(np.mean((r_residuals_kwaj)**2))
        RMS_Kwaj_rr = np.sqrt(np.mean((rr_residuals_kwaj)**2))

 
        t_kwaj = obs_mat[np.where(obs_mat[:, 0] == 1)[0], 1]

        print("Kwaj Range RMS:", RMS_Kwaj_r, "[km]")
        print("Kwaj Range Rate RMS:", RMS_Kwaj_rr,"[km/s]")

        ax[0].scatter(t_kwaj, r_residuals_kwaj, label='Kwaj Range Residuals')
        ax[1].scatter(t_kwaj, rr_residuals_kwaj, label='Kwaj Range Rate Residuals')

    if case == 'A' or case == 'B' or case == 'D' or case == 'F' or case == 'G':


        r_residuals_dg = residuals[np.where(obs_mat[:, 0] == 2)[0], 0]
        rr_residuals_dg = residuals[np.where(obs_mat[:, 0] == 2)[0], 1]

        RMS_Diego_r = np.sqrt(np.mean((r_residuals_dg)**2))
        RMS_Diego_rr = np.sqrt(np.mean((rr_residuals_dg)**2))

        t_diego = obs_mat[np.where(obs_mat[:, 0] == 2)[0], 1]

        print("Diego Range RMS:", RMS_Diego_r, "[km]")
        print("Diego Range Rate RMS:", RMS_Diego_rr,"[km/s]")

        ax[0].scatter(t_diego, r_residuals_dg, label='Diego Range Residuals')
        ax[1].scatter(t_diego, rr_residuals_dg, label='Diego Range Rate Residuals')
    

    if case == 'A' or case == 'B' or case == 'E' or case == 'F' or case == 'G':


        r_residuals_arecibo = residuals[np.where(obs_mat[:, 0] == 3)[0], 0]
        rr_residuals_arecibo = residuals[np.where(obs_mat[:, 0] == 3)[0], 1]

        RMS_Arecibo_r = np.sqrt(np.mean((r_residuals_arecibo)**2))
        RMS_Arecibo_rr = np.sqrt(np.mean((rr_residuals_arecibo)**2))

        t_arecibo = obs_mat[np.where(obs_mat[:, 0] == 3)[0], 1]

        print("Arecibo Range RMS:", RMS_Arecibo_r, "[km]")
        print("Arecibo Range Rate RMS:", RMS_Arecibo_rr,"[km/s]")

        ax[0].scatter(t_arecibo, r_residuals_arecibo, label='Arecibo Range Residuals')
        ax[1].scatter(t_arecibo, rr_residuals_arecibo, label='Arecibo Range Rate Residuals')


    three_sigma_pos = np.array([np.sqrt(P_ZZ[i][0])*3 for i in range(len(t))])
    three_sigma_vel = np.array([np.sqrt(P_ZZ[i][3])*3 for i in range(len(t))])

    # print(three_sigma_pos)
    ax[0].plot(t, three_sigma_pos, label='+3$\sigma$')
    ax[0].plot(t, -three_sigma_pos, label='-3$\sigma$')
    ax[1].plot(t, three_sigma_vel, label='+3$\sigma$')
    ax[1].plot(t, -three_sigma_vel, label='-3$\sigma$')



    ax[0].set_title('Range Residuals')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Range Residuals [km]')
    ax[0].set_ylim(-0.05, 0.05)
    ax[0].legend()


    ax[1].set_title('Range Rate Residuals')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Range Rate Residuals [km/s]')
    ax[1].set_ylim(-0.00002, 0.00002)
    ax[1].legend()
    # plt.show()
    fig.savefig(res_type + 'residuals_case' + case+ '.png', dpi=300)

def main():



    # cases = ['B']
    # cases = ['F', 'G']
    # cases = ['A','B', 'C', 'D', 'E', 'F', 'G']
    cases = ['D']
    # cases = ['G']


    #initial state
    r_ECI = np.array([6984.45711518852, 1612.2547582643, 13.0925904314402]) #km

    v_ECI = np.array([-1.67667852227336, 7.26143715396544, 0.259889857225218]) #km/s

    # r_ECI = np.array([6978.747999052552, 1616.0839015395757, 19.491496566388246])

    # v_ECI = np.array([-1.6625083343901703, 7.260899012707677, 0.2705614557194724])

    # r_ECI = np.array([6985.31847092135, 1621.22693026875, 15.7331358497728])
    # v_ECI = np.array([-1.67429646939288, 7.25923523079474, 0.261383935118773])

    # r_ECI = np.array([6895.2407083407, 1887.176775002, 163.898659816479])*1000

    # v_ECI = np.array([-1.94866558602947, 7.20540487140514, 0.201852373851337])*1000

    one_day_truth = np.array([-6330.16736001325, 3306.81591178162, 127.736863438565, -3.43787907681733, -6.63350511630163, -0.235613730204275]) #km
    best_case_pos = np.array([445.825159257239,  -7100.15487544519, -183.846642900717]) #km
    best_case_vel = np.array([7.48607847654087, 0.468696626655197, 0.177504090224814]) #km/s
    #station locations
    stations = np.array([[-6143.584, 1364.250, 1033.743], 
                        [1907.295, 6030.810, -817.119],
                        [2390.310, -5564.341 , 1994.578]]) #km

    JD_UTC = gregorian_to_jd(2018, 3, 23, 8, 55, 3)
    
    seconds = 7*86400
    #switch for observations
    for case in cases:

        obs_mat_3d = loadmat('LEO_DATA_Apparent_3Days.mat')
        obs_mat_4_6 = loadmat('LEO_DATA_Apparent_Days4-6.mat')
        obs_mat_4_6['LEO_DATA_Apparent'][:,1] += 3*86400
        obs_mat = np.concatenate((obs_mat_3d['LEO_DATA_Apparent'], obs_mat_4_6['LEO_DATA_Apparent']), axis=0)

        
        y_arr, P_hat, P_bar, P_ZZ, obs_mat, Q, pre_fit_res, post_fit_res, phi = EKF(case, r_ECI, v_ECI, stations, JD_UTC, seconds)
        
        t = obs_mat[:, 1]

        #plot residuals
        plot_residuals(case, obs_mat, pre_fit_res, 'pre', t, P_ZZ)
        plot_residuals(case, obs_mat, post_fit_res, 'post', t, P_ZZ)

        print('propogating state...')
        #propagate state
        del_t = seconds-t[-1]
        phi = np.eye(6)
        F = A_Matrix()
        t_eval = np.arange(t[-1], seconds + 60, 60)
        y0 = np.concatenate((y_arr[-1][0:6], phi.ravel()))
        sol_final = solve_ivp(satellite_motion_phi, [t[-1], seconds], y0, t_eval=t_eval, args=(F, JD_UTC), rtol=3E-14, atol=1E-16)

        y_final = sol_final.y[0:6, -1]
        print('y_final:', y_final)
        if seconds == 86400:
            print('residuals:', one_day_truth - y_final)
            print('distance:', np.linalg.norm(one_day_truth - y_final))
        elif seconds == 7*86400:
            
            print('residuals:', best_case_pos - y_final[0:3])
            print('distance:', np.linalg.norm(best_case_pos - y_final[0:3]))
        phi_k = sol_final.y[6:, -1].reshape(6, 6)


        #process noise
        Q_k = Q(del_t)

        #propogate covariance
        P_hat_k = P_hat[-1].reshape(6,6)
        P_bar_k_1 = phi_k @ P_hat_k @ phi_k.T + Q_k
        cov = P_bar_k_1
        print('covariance:', cov)
        print('writing to mat file...')
        #save to .mat file
        savemat('smithT_pos_vel_case' + case + '.mat', {'smithT_pos_vel_case' + case : y_final}, oned_as='column')
        savemat('smithT_poscov_case' + case + '.mat', {'smithT_pos_case' + case : y_final[0:3], 'smithT_poscov_case' + case : cov[0:3, 0:3]}, oned_as='column')
    

if __name__ == "__main__":
    main()
        