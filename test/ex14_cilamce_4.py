import numpy as np
from elastopy import statics
from elastopy import gmsh
from elastopy import plotter
from elastopy import data
from diffuspy import steadystate
from diffuspy import transient
from diffuspy import plotter as plottert
import matplotlib.pyplot as plt
from elastopy import stress
from thermalstrain import thermal_strain


print('Initializing script...')

mesh_name = 'dam-2mat'

model = gmsh.Parse(mesh_name)

material = data.Collect()
s = list(model.surf.keys())

# Concrete Dam
material.cndtvt[s[0]] = 2.6     # J/(m s C) Leger1993
material.spcfht[s[0]] = 900      # J/kg C Leger1993
material.dnsty[s[0]] = 2400     # kg/m3 Leger1993
material.E[s[0]] = 28e+9      # N/m2 Leger1993
material.nu[s[0]] = 0.2
material.thrmexp[s[0]] = 9e-6  # 1/C

# Foundation Rock
material.dnsty[s[1]] = 3011
material.cndtvt[s[1]] = 4       # Leger1993
material.thrmexp[s[1]] = 9e-6   # Leger1993
material.E[s[1]] = 28e+9      # Leger1993
material.nu[s[1]] = 0.33        # Leger1993
material.spcfht[s[1]] = 840     # Leger1993

g = 9.81


def b_force(x1, x2, t=1):
    return np.array([0.0,
                     (x2 >= 70.0) * -material.dnsty[s[0]] * g +
                     (x2 < 70.0) * -material.dnsty[s[1]] * g])

gma = 9820.0


def trac_bc(x1, x2, t=1):
    return {('line', 7): [(x2 <= 186.13)*(gma*(186.13-x2)), 0.0],
            ('line', 8): [(17.07 * gma - 116.13 * gma) * (x2 - 70.0) /
                          (169.06 - 70.0) + 116.13 * gma,
                          -((17.07 * gma - 116.13 * gma) * (x2 - 70.0) /
                            (169.06 - 70.0) + 116.13 * gma)],
            ('line', 9): [0.0, - 116.13 * gma]}


def displ_bc(x1, x2):
    return {('line', 0): [0.0, 0.0],
            ('line', 10): [0.0, 'free'],
            ('line', 1): [0.0, 'free']}

P = 24*60*60                    # convert days to seconds


def b_heat(x1, x2, t=1):        # [J/(s m3)]
    if t == 0:
        return 0
    else:
        return ((x2 >= 70) * 12 * (1/864 * np.exp(3.47 +
                                                  0.4*np.log(t/P) -
                                                  0.475*(np.log(t/P))**2)))


def flux_bc(x1, x2, t=1):
    return {}


def temp_bc(x1, x2, t=1):
    return {0: 10,
            1: 10/70*x2 + 10,
            2: 20 + 15*np.sin(2*np.pi/(24*60*60)*(t-9*60*60)),
            3: 20 + 15*np.sin(2*np.pi/(24*60*60)*(t-9*60*60)),
            4: 20 + 15*np.sin(2*np.pi/(24*60*60)*(t-9*60*60)),
            5: 20 + 15*np.sin(2*np.pi/(24*60*60)*(t-9*60*60)),
            6: 20 + 15*np.sin(2*np.pi/(24*60*60)*(t-9*60*60)),
            7: 20 + 15*np.sin(2*np.pi/(24*60*60)*(t-9*60*60)),
            8: ((10/(169-70)*(x2 - 70)) + 10 +
                10*np.sin(2*np.pi/(24*60*60)*(t-9*60*60))),
            9: 10,
            10: 10}

t_int = 60*60*24*2             # Interval
dt = t_int/300             # step size
N = int(t_int/dt)+1             # number of steps

U = np.zeros((model.ndof, N))   # Array for each time step
SIG = np.zeros((model.nn, 3, N))
T = np.zeros((model.nn, N))

T[:, 0] = steadystate.solver(model, material, b_heat, flux_bc, temp_bc)
U[:, 0], SIG[:, :, 0] = statics.solver(model, material,
                                       b_force, trac_bc, displ_bc)
print('Beginning computations...')
for n in range(1, N):
    t = n * dt                  # time

    T[:, n] = steadystate.solver(model, material, b_heat, flux_bc,
                                 temp_bc, t)

    EPS0 = thermal_strain(model, material, T[:, n], T[:, n-1])

    U[:, n], SIG[:, :, n] = statics.solver(model, material,
                                           b_force, trac_bc, displ_bc, EPS0, t)

print('Computations done!')

"""
PRINT THE BODY INTERNAL ENERGY GENERATION (HEAT) THROUGH TIME
"""
# print(b_heat(1, 70, 80))
x, y = [], []
for t in range(0, 24*20):
    x.append(t/24)
    t_s = t*60*60
    y.append(b_heat(1, 70, t_s))
fig, ax = plt.subplots()
ax.plot(x, y, '-b')
ax.set_xlabel('Time (days)')
ax.set_ylabel(r'Internal heat $ \sigma_q \quad (W/m^3)$')
plt.savefig('Q_evo.pdf', bbox_inches='tight')

"""
PLOT SPMAX, SPMIN, T AND U FIELD AT SPECIFIC TIMES
"""
print('Beginning plotting...')
plot_spmax = False
plot_spmin = False
plot_displ = False
plot_temp = False
plot_spmax_ani = False
plot_spmin_ani = False
plot_displ_ani = False
plot_temp_ani = True

time = [0, 1, 2, 3, 4, 6, 8,
        10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44]  # in [h]

if plot_spmax_ani is True:
    plotter.stress_animation(SIG, model, t_int, dt, spmax=True,
                             name="spmax.gif", vmin=-1.1, vmax=2,
                             interval=100, ftr=1e6, lev=10)

if plot_spmin_ani is True:
    plotter.stress_animation(SIG, model, t_int, dt, spmin=True,
                             name="spmin.gif", vmin=-6.5, vmax=0,
                             interval=100, ftr=1e6, lev=10)

if plot_temp_ani is True:
    print('Plotting temperature animation...')
    T_mod = np.zeros((model.nn, 3, N))
    T_mod[:, 0, :] = T
    plotter.stress_animation(T_mod, model, t_int, dt, s11=True,
                             name="temp.gif", vmin=0, vmax=100,
                             interval=100, ftr=1, lev=10)

T_point = []
time_point = []
for ti in time:
    time_index = int((t_int/dt)/t_int*(60*60*ti))
    """
    PRINCIPAL STRESS MAXIMUM
    """
    if plot_spmax is True:
        spmax = stress.principal_max(SIG[:, 0, time_index],
                                     SIG[:, 1, time_index],
                                     SIG[:, 2, time_index])
        # print('spmax min and max', np.amax(spmax)/1e6, np.amin(spmax)/1e6)
        plotter.stresses(model, SIG[:, :, time_index], ftr=1e6, spmax=True,
                         vmin=-2, vmax=3, lev=10,
                         title='Time: '+str(ti)+'h')
        plt.savefig('spmax'+str(ti)+'h.pdf', bbox_inches='tight')

    """
    PRINCIPAL STRESS MINIMUM
    """
    if plot_spmin is True:
        spmin = stress.principal_min(SIG[:, 0, time_index],
                                     SIG[:, 1, time_index],
                                     SIG[:, 2, time_index])
        # print(np.amax(spmin)/1e6, np.amin(spmin)/1e6)
        plotter.stresses(model, SIG[:, :, time_index], ftr=1e6, spmin=True,
                         vmin=-7, vmax=0, lev=10,
                         title='Time: '+str(ti)+'h')
        plt.savefig('spmin'+str(ti)+'h.pdf', bbox_inches='tight')

    # TEMPERATURE
    if plot_temp is True:
        T_point.append(T[105, time_index])
        time_point.append(ti)
        plottert.contour(model, T[:, time_index],
                         title='Time: '+str(ti)+'h', vmin=0, vmax=100)
        plt.savefig('temp'+str(ti)+'h.pdf', bbox_inches='tight')

    # DISPLACEMENT
    if plot_displ is True:
        plotter.model_deformed(model, U[:, time_index], magf=5000, color='r')
        plt.gca().set_title('Time: '+str(ti)+'h')
        plt.savefig('displ'+str(ti)+'h.pdf', bbox_inches='tight')

if plot_temp is True:
    fig, ax = plt.subplots()
    ax.plot(time_point, T_point, '-b')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Temperature (C)')
    plt.savefig('T_evo.pdf', bbox_inches='tight')

if plot_displ_ani is True:
    plotter.displ_animation(U, model, t_int, dt, name='displ.gif', magf=5000,
                            interval=100)
print('Plotting done!')
