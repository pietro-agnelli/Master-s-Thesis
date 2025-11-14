import threading
import time
from typing import List
import math

import meshcat.geometry as mgeom
import meshcat.transformations as tf
import meshcat_shapes
import numpy as np
import pinocchio as pin
import quadprog
from example_robot_data import load
from pinocchio.visualize import MeshcatVisualizer

from interpolator import SegmentedSE3Trap
from visualization_daemon import VisualizationDaemon
from pinocchio import SE3
import math


import scipy.io as sio
import matplotlib.pyplot as plt

# ---------------------------- CONSTANTS --------------------------------------
C = 0.25
Tr = 0.15
a_s = 2.5
v_h = 1.6
v_max = 20
Tc = 2e-3
v_pfl = 0.25
gamma = 20.0  
D_max =3.0
D_min = 0.0

h_threshold = 0.2  

NUM_D_GRID = 30
NUM_V_GRID = 30
OVERLAP = 2.0


def compute_dcrit(v_pfl, Tr, a_s):
    """
    Distanza critica calcolata come la distanza
    di frenata alla velocità v = -v_pfl.
    
    """
    d_reazione = -(-v_pfl) * Tr
    d_frenata = (-v_pfl)**2 / (2.0 * a_s)
    
    return d_reazione + d_frenata


def compute_safe_velocity_limit(d, v_max, Tr, v_pfl, a_s):
    """
    Calcola v_rel(d)
    
    Parametri: 
        d : distanza attuale
        v_max : velocità massima ammessa 
        Tr : tempo di reazione del controllore
        v_pfl : velocità massima di impatto ammessa
        a_s : accelerazione / decelerazione massima
    """
    d_crit = compute_dcrit(v_pfl, Tr, a_s)
    if d <= d_crit:
        return -v_pfl
    
    arg_sqrt = (Tr**2) + (2.0 * d / a_s)
    if arg_sqrt < 0:
        return -v_pfl
        
    v_curve = a_s * (Tr - math.sqrt(arg_sqrt))
    v_bound = max(v_curve, -v_max)
    return v_bound


def create_ellipse(
    num_d, num_v, d_min, d_max, 
    v_max, v_pfl, Tr, a_s, overlap_factor=1.5
    ):
    """
    Genera una griglia di ellissi
    Parametri: 
        num_d: numero di suddivisioni dell'asse d
        num_v: numero di suddivisioni dell'asse v
        d_min: distanza minima esperimento (0)
        d_max: distanza massima esperimento 
        v_max: velocità massima ammessa
        v_pfl: velocità massima di impatto ammessa
        Tr: Tempo di reazione del controllore
        a_s: accelerazione/ decelerazione massima
        Overlap_factor
    """
    ellipses_list = []

    d_grid = np.linspace(d_min, d_max, num_d)
    v_grid = np.linspace(-v_max, 1.0, num_v)

    
    step_d = (d_grid[-1] - d_grid[0]) / (num_d - 1)
   
    step_v = (v_grid[-1] - v_grid[0]) / (num_v - 1)
   
    
    a_i = (step_d / 2.0) * overlap_factor
    b_i = (step_v / 2.0) * overlap_factor
    
    
    
    grid_Di, grid_Vi = np.meshgrid(d_grid, v_grid)

    
    for Di, Vi in zip(grid_Di.flat, grid_Vi.flat):
        v_boundary = compute_safe_velocity_limit(Di, v_max, Tr, v_pfl, a_s)
        
        if Vi >= v_boundary and Vi <= v_max:
            ellipses_list.append((Di, Vi, a_i, b_i))
            
    return ellipses_list
    

    
def compute_h_PFL(d, v_rel, ellipses_list):
    """
        Calcola il valore di h(X) e restituisce i parametri dell'ellisse che massimizza h
        
        Parametri: 
            d: distanza attuale
            v_rel : distanza relativa
            ellipses_list: lista di ellissi
        
    """
    h_max = 0.0
    active_ellipse = None
    
    for ellipse in ellipses_list:
        (Di, Vi, ai, bi) = ellipse
        
        s = ((d - Di) / (ai + 1e-9)) ** 2 + ((v_rel - Vi) / (bi + 1e-9)) ** 2
        h_i = 1.0 - s
        
        if h_i > h_max:
            h_max = h_i
            active_ellipse = ellipse
            
            
    return h_max, active_ellipse



def compute_jacobian_h(d, v_rel, ellipse):
    """
    Restituisce dh/d[d, v_rel]

    """
    
    (Di, Vi, ai, bi) = ellipse
    
    dh_dd = 2.0 * (d - Di) / (ai**2 + 1e-9)
    dh_dv = 2.0 * (v_rel - Vi) / (bi**2 + 1e-9)
    
    return np.array([[dh_dd, dh_dv]])


def range_state_derivative(v_lin, v_human):
    zero3 = np.zeros(3)
    f = np.concatenate([v_lin, v_human, zero3, zero3])
    g = np.zeros((12, 3))
    g[6:9] = np.eye(3)
    return f, g


def jacobian_psi(p_r, p_h, v_lin, v_human):
    diff = p_r - p_h
    norm = np.linalg.norm(diff)
    if norm < 1e-9:
        norm = 1e-9

    u_rh = (diff / norm).reshape(3, 1)
    P = np.eye(3) - u_rh @ u_rh.T

    w = v_lin - v_human
    wP_over_d = (w @ P) / norm

    row_d = np.hstack((
        u_rh.T,
        -u_rh.T,
        np.zeros((1, 3)),
        np.zeros((1, 3)),
    ))

    row_vrel = np.hstack((
        wP_over_d.reshape(1, -1),
        -wP_over_d.reshape(1, -1),
        u_rh.T,
        -u_rh.T,
    ))

    return np.vstack((row_d, row_vrel))


def damped_pinv_svd(J, lam=1e-4):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + lam ** 2)
    return (Vt.T * S_damped) @ U.T


def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    model_wrapper = load("ur10")
    model = model_wrapper.model
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    obstacle_positions: List[np.ndarray] = [np.array([0.8, 0.7, 0.5])]

    for i, pos in enumerate(obstacle_positions):
        viz.viewer[f"obstacle_{i}"].set_object(
            mgeom.Sphere(0.1), mgeom.MeshLambertMaterial(color=0xFF0000)
        )

    side = 0.2
    viz.viewer["goal"].set_object(
        mgeom.Box([side, side, side / 10]), mgeom.MeshLambertMaterial(color=0x00FF00)
    )

    CBF = False
    renderer = VisualizationDaemon(viz)

    # --------------------------- CONTROL INITIALISATION --------------------------
    data = model.createData()
    q = np.zeros(model.nq)
    q[1] = -np.pi / 2
    q[2] = np.pi / 2
    q[4] = np.pi / 4

    dq = np.zeros(model.nq)
    ddq = np.zeros(model.nq)

    tool_frame_id = model.getFrameId("tool0")

    pin.framesForwardKinematics(model, data, q)

    wn = 300
    xi = 0.9
    Kp_tra = np.array([1, 1, 1]) * wn ** 2
    Kd_tra = np.array([1, 1, 1]) * 2.0 * xi * wn
    Kp_rot = np.array([1, 1, 1]) * wn ** 2
    Kd_rot = np.array([1, 1, 1]) * 2.0 * xi * wn

    twist_goal = np.zeros(6)

    planner = SegmentedSE3Trap(
        vlin_max=0.6, vang_max=1.2,
        alin_max=1.8, aang_max=2.0
    )

    def pose_eul(z, y, x, xyz):
        R = pin.utils.rotate('z', z) @ pin.utils.rotate('y', y) @ pin.utils.rotate('x', x)
        return SE3(R, np.array(xyz))

    goal_pose = data.oMf[tool_frame_id].copy()

    planner.addWayPoint(goal_pose * SE3.Identity())
    planner.addWayPoint(goal_pose * pose_eul(0.0, 0.0, 0.0, [0.30, 0.00, 0.0]))
    planner.addWayPoint(goal_pose * pose_eul(math.pi / 4, 0.0, 0.0, [0.30, -0.1, 0.020]))
    planner.addWayPoint(goal_pose * pose_eul(math.pi / 4, 0.0, -math.pi / 4, [0.3, -0.1, 0.2]))
    planner.addWayPoint(goal_pose * pose_eul(-math.pi / 4, 0.0, 0.0, [0.30, 0.0, 0.0]))
    planner.addWayPoint(goal_pose * SE3.Identity())

    T_total = planner.computeTime()
    
    
    #GENERO ELLISSI
    ellipses_list = create_ellipse(
        NUM_D_GRID, NUM_V_GRID, D_min, D_max, 
        v_max, v_pfl, Tr, a_s, OVERLAP
    )
    
        
    renderer.publishPath(planner.publishPath())
    print(f"Total time = {T_total:.3f} s")
    
    
    #Setuppo variabili per la visualizzazione dei grafici
    log_time = []
    log_pos_actual = []
    log_pos_nominal = []
    log_distance = []
    log_v_rel = []
    log_h = []
    
    # ------------------------------ MAIN LOOP ------------------------------------
    try:
        t = 0.0
        trajectory_time = 0.0
        
        Dtrajectory_time = 1.0 
        DDtrajectory_time = 0.0 

        h_prev = 1.0

        while t < 150.0:
            loop_start = time.perf_counter()
            
            Dtrajectory_time = 1.0 
            DDtrajectory_time = 0.0  
            
            
            goal_act_pose, nominal_twist_goal, nominal_goal_dtwist = planner.getMotionLaw(
                trajectory_time % T_total
            )
            
            
            twist_goal = nominal_twist_goal * Dtrajectory_time
            goal_dtwist = (nominal_goal_dtwist * Dtrajectory_time ** 2.0 + 
                           nominal_twist_goal * DDtrajectory_time)

            CBF = (t % 40) < 20.0

            G = goal_act_pose.translation
            Rbg = goal_act_pose.rotation.copy()

            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, ddq)

            Tbt = data.oMf[tool_frame_id]
            translation_bt = Tbt.translation
            Rbt = Tbt.rotation.copy()

            Rtg = Rbt.T @ Rbg
            error_rot = Rbt @ pin.log3(Rtg)

            twist = pin.getFrameVelocity(
                model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            vel_lineare = twist.linear
            vel_angolare = twist.angular

            J = pin.computeFrameJacobian(
                model,
                data,
                q,
                tool_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            dJ = pin.frameJacobianTimeVariation(
                model,
                data,
                q,
                dq,
                tool_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            Jlin = J[:3, :]
            dJlin = dJ[:3, :]

            acc_lin = Kp_tra * (G - translation_bt) + Kd_tra * (twist_goal[:3] - vel_lineare) + goal_dtwist[:3]
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - vel_angolare) + goal_dtwist[3:]
            dtwist_tool = np.hstack([acc_lin, acc_ang])

            # ------------------------- CBF QP SET-UP ------------------------------
            constraint_matrix = np.empty((0, model.nq))
            constraint_vector = np.empty((0, 1))

            h = h_threshold
            distance = 3.0
            v_rel = 0.0
            active_ellipse = None 

            for i, obs_pos in enumerate(obstacle_positions):
                w1 = 2 * np.pi / 2
                w2 = 2 * np.pi / 2.1
                obs_pos[0] = 0.8 - 0.25 * np.sin(w1 * t)
                obs_pos[1] = 0.4 + 0.1 * np.sin(w2 * t)

                v_obs = np.array([0.0, 0.0, 0.0])
                v_obs[0] = -0.25 * np.cos(w1 * t) * w1
                v_obs[1] = 0.1 * np.cos(w2 * t) * w2

                r = translation_bt - obs_pos
                distance = np.linalg.norm(r)
                if distance < 1e-6: distance = 1e-6
                
                u_hr = r / distance
                v_rel_vector = vel_lineare - v_obs
                v_rel = np.dot(v_rel_vector, u_hr)
                
                
                h, active_ellipse = compute_h_PFL(distance, v_rel, ellipses_list)
                
                
                if CBF and active_ellipse is not None:
                    f, g = range_state_derivative(vel_lineare, v_obs)
                    Jh_psi = compute_jacobian_h(distance, v_rel, active_ellipse)
                    Jpsi_chi = jacobian_psi(translation_bt, obs_pos, vel_lineare, v_obs)

                    Lie_f_h = Jh_psi @ Jpsi_chi @ f
                    Lie_g_h = Jh_psi @ Jpsi_chi @ g

                    constraint_matrix = np.append(
                        constraint_matrix,
                        (Lie_g_h @ Jlin).reshape(1, -1),
                        axis=0
                    )
                    constraint_vector = np.append(
                        constraint_vector,
                        (-Lie_g_h @ dJlin @ dq - Lie_f_h - gamma * h).reshape(1, -1),
                        axis=0,
                    )
            
            h_prev = h # h_prev ora è usato solo per la visualizzazione

            # ----------------------------- QP SOLVE -----------------------------
            P = J.T @ J + 1e-6 * np.eye(model.nv)
            b = (J.T @ (dtwist_tool - dJ @ dq)).flatten()
            constraint_vector = constraint_vector.flatten()
            
            ddq_nominal = damped_pinv_svd(J) @ (dtwist_tool - dJ @ dq)

            if CBF and constraint_matrix.shape[0] > 0:
                try:
                    ddq, *_ = quadprog.solve_qp(
                        P,
                        b,
                        constraint_matrix.T,
                        constraint_vector,
                        0,
                    )
                except ValueError as err:
                     if "constraints are inconsistent" in str(err) or "No solution" in str(err):
                         print(f"QP infeasible (h={h_prev:.2f}) – applying fallback damping.")
                         ddq = -10.0 * dq
                     else:
                         raise
            else:
                 ddq = ddq_nominal

            # --------------------------- INTEGRATION ----------------------------
            q += dq * Tc + 0.5 * ddq * Tc ** 2
            dq += dq * Tc

            vizualization_string = f"h = {h:.2f}, s = {Dtrajectory_time:.2f}, CBF={CBF}"
            renderer.push_state(
                q,
                goal_act_pose,
                obstacle_positions,
                vizualization_string
            )
            
            log_time.append(t)
            log_pos_actual.append(translation_bt.copy())
            log_pos_nominal.append(G.copy())
            log_distance.append(distance)
            log_v_rel.append(v_rel)
            log_h.append(h)
            
            # ----------------------------- TIMING -------------------------------
            t += Tc
            
            
            trajectory_time += Tc 
            Dtrajectory_time += DDtrajectory_time * Tc
            

            if t > T_total:
                print(f"Completed one task cycle (T_total = {T_total:.3f} s). Stopping simulation.")
                break

            elapsed = time.perf_counter() - loop_start
            rest = Tc - elapsed
            if rest > 0:
                time.sleep(rest)
            
            
            
            
            

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    
    finally:
        print("\nSimulation loop ended. Generating plots...")
        
        if not log_time:
            print("No data logged, cannot plot.")
        else:
            time_arr = np.array(log_time)
            pos_actual_arr = np.array(log_pos_actual)
            pos_nominal_arr = np.array(log_pos_nominal)
            distance_arr = np.array(log_distance)
            v_rel_arr = np.array(log_v_rel)
            h_arr = np.array(log_h)
            
            fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            
            plot_titles_fig = ['x-axis', 'y-axis', 'z-axis']
            y_labels_fig = ['x [m]', 'y [m]', 'z [m]']
            
            for i in range(3):
                axs1[i].plot(time_arr, pos_actual_arr[:, i], 'r-', label='Reale (CBF)')
                axs1[i].plot(time_arr, pos_nominal_arr[:, i], 'k--', label='Nominale')
                axs1[i].set_ylabel(y_labels_fig[i])
                axs1[i].set_title(plot_titles_fig[i])
                axs1[i].grid(True)
                
            
            axs1[0].legend(loc='upper right')
            axs1[2].set_xlabel('t [s]')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            fig2, ax7_left = plt.subplots(figsize=(10, 6))
            

            ax7_left.set_xlabel('t [s]', fontsize=12)
            ax7_left.set_ylabel('d [m]', color='r', fontsize=12)
            ax7_left.plot(time_arr, distance_arr, 'r-', label='Distanza (d)')
            ax7_left.tick_params(axis='y', labelcolor='r')
            ax7_left.grid(True, axis='x')

            ax7_right = ax7_left.twinx()
            ax7_right.set_ylabel('v_rel [m/s]', color='g', fontsize=12)
            ax7_right.plot(time_arr, v_rel_arr, 'g-', label='Velocità Relativa (v_rel)')
            ax7_right.axhline(y=-v_pfl, color='k', linestyle=':', linewidth=2, label=f'-v_PFL ({v_pfl:.2f} m/s)')
            ax7_right.tick_params(axis='y', labelcolor='g')
            
            lines_left, labels_left = ax7_left.get_legend_handles_labels()
            lines_right, labels_right = ax7_right.get_legend_handles_labels()
            ax7_right.legend(lines_left + lines_right, labels_left + labels_right, loc='best')
            
            fig3, ax_h = plt.subplots(figsize=(10, 4))
            ax_h.plot(time_arr, h_arr, 'b-', label='h(chi)')
            ax_h.axhline(y=h_threshold, color='k', linestyle=':', linewidth=2, label=f'h_threshold ({h_threshold:.2f})')
            ax_h.set_xlabel('t [s]')
            ax_h.set_ylabel('Valore CBF h')
            ax_h.set_title('Andamento Valore CBF h(chi)')
            ax_h.grid(True)
            ax_h.legend()
            
            plt.tight_layout()
            plt.show(block=True)



if __name__ == "__main__":
    main()