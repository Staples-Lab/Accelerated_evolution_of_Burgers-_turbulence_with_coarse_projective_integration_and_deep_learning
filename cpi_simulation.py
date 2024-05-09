import numpy as np
import matplotlib.pyplot as plt
import time

from one_D_burgers import parameters, solve_vel, derivative, noise
from utils import normalize_spectrum, normalize_velocity_field, mse_velocity_signals

def energy_spectrum(velocity_field, dx = 2*np.pi / 8192):
   
    N = velocity_field.shape[1]
    ft_velocity_field = np.fft.fft(velocity_field) / N
    ft_velocity_field = np.abs(ft_velocity_field)**2
    ft_velocity_field = ft_velocity_field[:, :N//2] 
    k = np.arange(N//2) / (N * dx)
    ft_velocity_field = ft_velocity_field
    return k, ft_velocity_field

def coarse_project_euler(energy_t1, energy_t2, delta_t_fine, delta_t_coarse):
    """Project the energy spectrum into the future using Euler's method."""
    d_energy_dt = (energy_t2 - energy_t1) / delta_t_fine

    energy_future = energy_t2 + delta_t_coarse * d_energy_dt
    energy_future[energy_future < 0] = 0

    return energy_future

def coarse_project_adams_bashforth_4th_order(energy_t0, energy_t1, energy_t2, energy_t3, fs_dt, cs_dt):
    """Project the energy spectrum into the future using 4th-order Adams-Bashforth method."""

    d_energy_dt0 = (energy_t1 - energy_t0) / fs_dt
    d_energy_dt1 = (energy_t2 - energy_t1) / fs_dt
    d_energy_dt2 = (energy_t3 - energy_t2) / fs_dt

    energy_future = energy_t3 + (cs_dt / 24) * ((55 * d_energy_dt2) - (59 * d_energy_dt1) + (37 * d_energy_dt0))
    energy_future[energy_future < 0] = 0
    
    return energy_future

def heal_vel(visc, damp, dt, u, clock_time, nt, heal_threshold, mse_pred, proj_step_vel, patience):

    out_t = []
    out_k = []
    out_u = []

    t1 = time.time()
    
    ##Saving predicted velocity
    u_pred_save = u

    save_t = 0
    best_mse = np.inf
    steps_without_improvement = 0

    for t in range(int(nt)):

        derivs = derivative(u=u,dx=dx)
        du2dx  = derivs['du2dx']
        d2udx2 = derivs['d2udx2']

        fbm = noise(0.75,nx)

        rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*damp/dt)*fbm

        if t == 0:
            u_new = u + dt*rhs
        else:
            u_new = u + dt*(1.5*rhs - 0.5*rhsp)

        fu_new     = np.fft.fft(u_new)
        fu_new[mp] = 0
        u_new      = np.real(np.fft.ifft(fu_new))
        u          = u_new
        rhsp       = rhs

        MSE_ = mse_velocity_signals(u, proj_step_vel)

        if t % 1000 == 0:
             print(f"MSE_HEAL: {MSE_}, mse_pred: {mse_pred}, timesteps: {t}")

        if MSE_ < (heal_threshold * mse_pred):
            best_mse = MSE_
            steps_without_improvement = 0
            # kinetic energy
            tke  = 0.5*np.var(u)
            
            # save to disk
            out_t.append((t+1)*dt) 
            out_k.append(tke)
            out_u.append(u)
            
            save_t += 1

            # time info
            t2 = time.time()
            tt = t2 - t1

            vel = np.array(out_u)
            clock_time = (t+1)*dt + clock_time
            
            break
            
        else:
            steps_without_improvement += 1

        if steps_without_improvement > patience:
            print("Early stopping condition reached.")
            vel = u_pred_save.reshape(1, 8192)
            break

        if t == nt - 1:
            tke  = 0.5*np.var(u)
            out_t.append((t+1)*dt) 
            out_k.append(tke)
            out_u.append(u)
            save_t += 1

            t2 = time.time()
            tt = t2 - t1

            vel = np.array(out_u)
            clock_time = (t+1)*dt + clock_time
        else:
            save_t += 1
            t2 = time.time()
            tt = t2 - t1

    print("\n[pyBurgers: DNS] \t Done! Completed in %0.2f seconds"%tt)
    print("##############################################################")

    return vel, tt, clock_time, t

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def main_scheme():

    mse_heal_store = []
    total_time_store = []
    proj_step_store = []

    vel_store = []
    true_vel_store = []
    enr_store = []

    for coarse_step in CS_dt:
        
        for fine_step in FS_dt:
            print('currently solving for {} coarse step and {} fine step'.format(coarse_step, fine_step))

            # Variable parameters
            fs_dt = fine_step
            cs_dt = coarse_step #2 
            heal_steps = 2e4 #1e5
            heal_trsh = 1 #This parameter could be lowered for better accuracy but higher healing times

            clock_time = 0
            projection_step = 0

            stat_stable_state = 100
                    
            # Step 1: initializing parameters
            nt, dt, visc, damp, nx, mp, dx, u, rhsp, out_x = parameters()

            sim_timesteps = 2 * fs_dt / dt

            time0 = time.time()
    #         for i in range(1):
            while clock_time < stat_stable_state:
                
                projection_step = projection_step + 1
                print('current CPI cycle:', projection_step)
                
    #             np.random.seed(43)

                # Step 2: Microscopic flow simulator (Burgers' 1-D turbulence)
                vel, tt, clock_time = solve_vel(visc, damp, dt, u , clock_time, nt = sim_timesteps)

                print('simulator clock time:', clock_time)

                vel1 = vel[0].reshape(1, 8192)
                vel2 = vel[1].reshape(1, 8192)
    #             vel3 = vel[2].reshape(1, 8192)
    #             vel4 = vel[3].reshape(1, 8192)
                
                # Step 3: Restriction (VEL --> ENR)
                k1, step1 = energy_spectrum(vel1, dx = 2*np.pi / 8192)
                k2, step2 = energy_spectrum(vel2, dx = 2*np.pi / 8192)
    #             k3, step3 = energy_spectrum(vel3, dx = 2*np.pi / 8192)
    #             k4, step4 = energy_spectrum(vel4, dx = 2*np.pi / 8192)
                
                # Filter the E1 and E2 energy timesteps for projection
    #             cutoff_wavenumber = 100
    #             window_length = 51  # Choose an odd window length
    #             polynomial_order = 3  # Choose the order of the polynomial to fit
    #             step1_fil = savgol_filter(step1, window_length, polynomial_order)
    #             step2_fil = savgol_filter(step2, window_length, polynomial_order)
    #             high_freq_k = step2 - step2_fil
                
                # Step 4: Coarse projective integration
                energy_future = coarse_project_euler(step1, step2, fs_dt, cs_dt)
    #             energy_future = coarse_project_adams_bashforth_4th_order(step1, step2, step3, step4, fs_dt, cs_dt)
    #             energy_future = coarse_project_euler(step1_fil, step2_fil, fs_dt, cs_dt)
    #             energy_future = energy_future + high_freq_k

                # Filters for projected energy signal
    #              
    #             energy_future = savgol_filter(energy_future, 101, 3)
    #             sigma = 10  # Adjust this value to control the amount of smoothing
    #             energy_future = apply_gaussian_filter(energy_future, sigma)
                
                clock_time = clock_time + cs_dt
                
                #Plot projected energy spectrum
                plt.figure(figsize=(20,5))
                plt.loglog(step2.flatten(), '--',label='unfiltered E4')
    #             plt.loglog(step2_fil.flatten(), '--',label='filtered E2')
                plt.loglog(energy_future.flatten(), '--',label='Projected energy')
                plt.xlabel('k')
                plt.ylabel('E(k)')
                plt.title(f'Iteration at clock time: {clock_time}')
                plt.legend()
                plt.show()
                
                # Filter the energy spectrum
    #             cutoff_wavenumber = 1000
    #             ef_nor = butter_lowpass_filter(ef_nor, cutoff_wavenumber, 8192 // 2)
        
                # Normalizing the projected energy spectrum
                ef_nor, m, s = normalize_spectrum(energy_future)
        
                # Step 5: Lifting the projected step to fine scale
                Y_input_test_dummy = np.zeros((1, 1, 8193)) #Place holder
                pred = model.predict([(ef_nor).reshape(1, 1, 4096), Y_input_test_dummy])

                print('projected clock time:', clock_time)

                # Current projected timestep
                proj_step = ds_step_calc(clock_time, ds_sample_size=200)
                print('current_simulation timestep:',proj_step)
                print("##############################################################")
                    
                # Mean and std models
                scaling_ = np.column_stack((ef_nor, m, s))
                scaling_ = scaling_.reshape(1, -1)
                M = mean_model.predict(scaling_)
                S = std_model.predict(scaling_)
                
                print('scaling preds:', M, S)
                    
                # Infered velocity post-processing
                u_pred = pred[0, 0, :8192]
                some_vel, m, s = normalize_velocity_field(VEL[proj_step])
    #             some_vel, m, s = normalize_velocity_field(vel2) ##Taking the scalings from prev step
                print('scaling actual:', m, s)
                
                u_pred_us = u_pred * s  + m
    #             u_pred_us = u_pred * S  + M

                # Create the plot
                plt.figure(figsize=(20,5))
                plt.plot(u_pred_us, label="CPI_pred")
    #             plt.plot(u_pred_US, label="trained_scaling")
                plt.plot(VEL[proj_step], label="truth")
                plt.xlabel('x')
                plt.ylabel('u')
                plt.title(f'Iteration at clock time: {clock_time}')
                plt.legend()
                plt.show()

                # MSE between the true velocity and predicted velocity
                mse_pred = mse_velocity_signals(u_pred_us, VEL[proj_step])
                print('MSE between true velocity and predicted velocity:',mse_pred)
                print("##############################################################")


                # Step 6: Healing
                u_heal, tt, clock_time, ts = heal_vel(visc, damp, dt, u_pred_us , clock_time, heal_steps, heal_trsh,
                                                mse_pred, VEL[proj_step], patience = 1e4)

                print('healed clock time:', clock_time)
                print('timesteps heal:', ts)

                # MSE between the true velocity and healed velocity
                mse_heal = mse_velocity_signals(u_heal[-1], VEL[proj_step])
                print('MSE between true velocity and healed velocity:',mse_heal)

                # Velocity re-initialization for next CPI step
                u = u_heal[-1]
                
                vel_store.append(u)
                true_vel_store.append(VEL[proj_step])
                
                # Create plot for phase initialized signal
                FT_vel = np.square(np.abs(np.fft.fft(VEL[proj_step])))  
                FT_vel = np.sqrt(FT_vel)
                random_u_tilde = fftnoise(FT_vel)
                
                #Create plot for E(K) visualization post healing
                kf, stepf = energy_spectrum(u.reshape(1, u.shape[0]), dx = 2*np.pi / 8192)
                
                enr_store.append(stepf)
                
    #             kr, stepr = energy_spectrum(random_u_tilde.reshape(1, u.shape[0]), dx = 2*np.pi / 8192)
                
    #             # Create the plot
    #             plt.figure(figsize=(20,5))
    #             plt.loglog(stepf.flatten() * (kf)**(-5/3), '--',label='CPI')
    #             plt.loglog(ENR[proj_step] * (kf)**(-5/3), '--',label='truth')
    # #             plt.loglog(stepr.flatten() * (kf)**(-5/3), label='rand_phase')
    #             plt.xlabel('k')
    #             plt.ylabel('E(k)')
    #             plt.title(f'Iteration at clock time: {clock_time}')
    #             plt.legend()
    #             plt.show()
                
                # MSE between the true velocity and rand velocity
                mse_rand = mse_velocity_signals(random_u_tilde, VEL[proj_step])
                print('MSE between true velocity and rand phase velocity:',mse_rand)

                # Create the plot
                plt.figure(figsize=(20,5))
                plt.plot(u, label="CPI")
                plt.plot(VEL[proj_step], label="truth")
                plt.plot(random_u_tilde, label="rand_phase")
                plt.xlabel('x')
                plt.ylabel('u')
                plt.title(f'Iteration at clock time: {clock_time}')
                plt.legend()
                plt.show()
                
    #             # Create plot for phase initialized signal
    #             FT_vel = np.square(np.abs(np.fft.fft(VEL[proj_step])))  
    #             FT_vel = np.sqrt(FT_vel)
    #             random_u_tilde = fftnoise(FT_vel)
    #             plt.plot(random_u_tilde, label="rand_phase")
    #             plt.xlabel('x')
    #             plt.ylabel('u')
    #             plt.title(f'Iteration at clock time: {clock_time}')
    #             plt.legend()
    #             plt.show()
                
            timef = time.time()
            timefinal = timef - time0
            
            # Saving to disk
            mse_heal_store.append(mse_heal)
            total_time_store.append(timefinal)
            proj_step_store.append(proj_step)
            
            print("##############################################################")
            print('TKE has reached a statistically steady state')
            print('time taken for the whole process:', timefinal)