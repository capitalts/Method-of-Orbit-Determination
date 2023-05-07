import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.io import savemat
from fp_utils import *

def main():
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(0, 0, 'o', label='Truth')
    ax[1].plot(0, 0, 'o', label='Truth')
    ax[2].plot(0, 0, 'o', label='Truth')
    
    seven_day_truth = np.array([445.825159257239,  -7100.15487544519, -183.846642900717, 7.48607847654087, 0.468696626655197, 0.177504090224814])

    # cases = ['A','B', 'C', 'D', 'E', 'F', 'G']
    cases = ['A', 'B']
    best_case = loadmat('smithT_best.mat')
    best_pos = best_case['smithT_pos_vel_caseD'].T[0][0:3]
    best_vel = best_case['smithT_pos_vel_caseD'].T[0][3:6]
    # case_pos_cov = loadmat('smithT_FDragCSol_0_-3_-12')
    # cases = ['A', 'B', 'C', 'D', 'E']
    for i in range(len(cases)):
        y_final, cov = loadmat('smithT_poscov_case' + cases[i] + '.mat')['smithT_pos_case' + cases[i]], loadmat('smithT_poscov_case' + cases[i]+ '.mat')['smithT_poscov_case' + cases[i]]
        # y_final = case_pos_cov['smithT_pos_case' + cases[i]]
        # cov = case_pos_cov['smithT_poscov_case' + cases[i]]
        y_final = y_final.T[0]
        # RIC_pred, RIC_cov = eci_to_ric(best_pos, best_vel, y_final, cov[0:3, 0:3].T)
        RIC_pred, RIC_cov = eci_to_ric(seven_day_truth[0:3], seven_day_truth[3:6], y_final, cov[0:3, 0:3].T)


        radial = RIC_pred[0]
        in_track = RIC_pred[1]
        cross_track = RIC_pred[2]

        
        ax[0].plot(radial, in_track, 'o')
        plot_ellipses(RIC_pred[0:2], RIC_cov[0:2, 0:2],3, ax[0], cases[i])

        ax[1].plot(radial, cross_track, 'o')
        
        plot_ellipses([RIC_pred[0], RIC_pred[2]], np.array([[RIC_cov[0, 0], RIC_cov[0, 2]],[RIC_cov[2, 0], RIC_cov[2, 2]]]),3, ax[1], cases[i])

        ax[2].plot(in_track, cross_track, 'o')
        
        plot_ellipses(RIC_pred[1:3], RIC_cov[1:3, 1:3],3, ax[2], cases[i])


        ax[0].set_xlabel('Radial (km)')
        ax[0].set_ylabel('In-track (km)')
        ax[0].set_title('Radial vs In-track')
        ax[0].legend(loc = 'upper left')
        ax[0].grid()
        ax[0].set_xlim(-0.3, 0.3)
        ax[0].set_ylim(-0.3, 0.3)

        ax[1].set_xlabel('Radial (km)')
        ax[1].set_ylabel('Cross-track (km)')
        ax[1].set_title('Radial vs Cross-track')
        ax[1].legend(loc = 'upper left')
        ax[1].grid()
        ax[1].set_xlim(-0.3, 0.3)
        ax[1].set_ylim(-0.3, 0.3)

        ax[2].set_xlabel('In-track (km)')
        ax[2].set_ylabel('Cross-track (km)')
        ax[2].set_title('In-track vs Cross-track')
        ax[2].legend(loc = 'upper left')
        ax[2].grid()
        ax[2].set_xlim(-0.3, 0.3)
        ax[2].set_ylim(-0.3, 0.3)
    plt.show()



if __name__ == "__main__":
    main()
        