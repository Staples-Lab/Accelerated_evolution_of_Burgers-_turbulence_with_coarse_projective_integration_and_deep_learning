import numpy as np
import math

## Parameters
def parameters(nt = 2000):

    # nt = 1000
    dt = 1E-4
    visc = 1E-5
    damp = 1E-6
    nx = 8192
    mp = int(nx/2)

    dx = 2*np.pi/nx

    u = np.zeros(nx)

    np.random.seed(1)

    rhsp = 0

    out_x = np.arange(0,2*np.pi,dx)

    # ## Time stepping for the energy data
    # m = 10
    # new_time = nt*dt + m*dt

    # print('Large time step is:', new_time)
    return nt, dt, visc, damp, nx, mp, dx, u, rhsp, out_x

def noise(alpha,n):
        x     = np.sqrt(n)*norm.ppf(np.random.rand(n))
        m     = int(n/2)
        k     = np.abs(np.fft.fftfreq(n,d=1/n))
        k[0]  = 1
        fx    = np.fft.fft(x)
        fx[0] = 0
        fx[m] = 0
        fx1   = fx * ( k**(-alpha/2) )
        x1    = np.real(np.fft.ifft(fx1))
        
        return x1

def derivative(u,dx):
        
        # signal shape information
        n = int(u.shape[0])
        m = int(n/2)
        
        # Fourier colocation method
        h       = 2*np.pi/n
        fac     = h/dx
        k       = np.fft.fftfreq(n,d=1/n)
        k[m]    = 0
        fu      = np.fft.fft(u)
        dudx    = fac*np.real(np.fft.ifft(cm.sqrt(-1)*k*fu))
        d2udx2  = fac**2 * np.real(np.fft.ifft(-k*k*fu))
        d3udx3  = fac**3 * np.real(np.fft.ifft(-cm.sqrt(-1)*k**3*fu))
        
        # dealiasing needed for du2dx using zero-padding 
        zeroPad = np.zeros(n)
        fu_p    = np.insert(fu,m,zeroPad)
        u_p     = np.real(np.fft.ifft(fu_p))        
        u2_p    = u_p**2
        fu2_p   = np.fft.fft(u2_p)
        fu2     = fu2_p[0:m]
        fu2     = np.append(fu2,fu2_p[n+m:])
        du2dx   = 2*fac*np.real(np.fft.ifft(cm.sqrt(-1)*k*fu2))

        # store derivatives in a dictionary for selective access
        derivatives = {
            'dudx'  :   dudx,
            'du2dx' :   du2dx,
            'd2udx2':   d2udx2,
            'd3udx3':   d3udx3
        }
        
        return derivatives

def solve_vel(visc, damp, dt, u, clock_time, nt):

    out_t = []
    out_k = []
    out_u = []

    ## Time integration
    t1 = time.time()

    save_t = 0

    for t in range(int(nt)):

        # compute derivatives
        derivs = derivative(u=u,dx=dx)
        du2dx  = derivs['du2dx']
        d2udx2 = derivs['d2udx2'] 

        # add fractional Brownian motion (FBM) noise
        fbm = noise(0.75,nx)

        # compute right hand side
        rhs = visc * d2udx2 - 0.5*du2dx + np.sqrt(2*damp/dt)*fbm
        
        # time integration
        if t == 0:
            # Euler for first time step
            u_new = u + dt*rhs
        else:
            # 2nd-order Adams-Bashforth
            u_new = u + dt*(1.5*rhs - 0.5*rhsp)
        
        # set Nyquist to zero
        fu_new     = np.fft.fft(u_new)
        fu_new[mp] = 0
        u_new      = np.real(np.fft.ifft(fu_new))
        u          = u_new
        rhsp       = rhs

        # output to file every 1000 time steps (0.1 seconds)
        if ((t+1) % int(nt/2) ==0): #Changed (nt/2) to (nt/4)          
            
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

    print("\n[pyBurgers: DNS] \t Done! Completed in %0.2f seconds"%tt)
    print("##############################################################")

    return vel, tt, clock_time