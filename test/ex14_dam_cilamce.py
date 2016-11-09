import numpy as np
from elastopy import dynamics
from elastopy import gmsh
from elastopy import plotter
from elastopy import data
from diffuspy import steadystate
from diffuspy import transient
from diffuspy import plotter as plottert
import matplotlib.pyplot as plt
from elastopy import stress


mesh_name = 'dam-2mat'
"""
MATERIAL PARAMETERS
"""
model = gmsh.Parse(mesh_name)
# plotter.model(model, edges_label=True)
# plotter.show()
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


"""
PRINT THE BODY INTERNAL ENERGY GENERATION (HEAT) THROUGH TIME
"""
# print(b_heat(1, 70, 80))
# x, y = [], []
# for t in range(0, 24*30):
#     x.append(t/24)
#     t_s = t*60*60
#     y.append(b_heat(1, 70, t_s))
# plt.plot(x, y, '-b')
# plt.xlabel('Time (days)')
# plt.ylabel(r'Density of heat current $(W/m^3)$')
# plotter.show()


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

"""
PLOT SENOIDAL CURVE
"""
# t = np.linspace(0, 100)
# T = 22.1 + 7.57*np.cos(np.pi/6*(t-6.5)) - 1.22*np.cos(np.pi/3*(t-6.5))
# plt.plot(t, T, '-b')
# plt.xlim(0, 100)
# plt.show()


T0 = 20
t_int = 60*60*24*2             # Interval
dt = t_int/10


"""
PLOT TEMPERATURE ANIMATION
"""
# frames = transient.solver(model, material, b_heat, flux_bc,
#                           temp_bc, T0, t_int, dt, vmin=0., vmax=100.)
# Tani = plottert.anime(frames, dt)
# plotter.show()

"""
SOLVE FOR DISPLACEMENT AND STRESS
"""
U, SIG = dynamics.solver(model, material, b_force,
                         trac_bc, displ_bc, t_int, dt,
                         b_heat=b_heat, flux_bc=flux_bc,
                         temp_bc=temp_bc, T0=T0)


"""
PLOT SPMAX and SPMIN AT SPECIFIC TIMES AT A SPECIFIC POINT
"""
nodes = [105, 95, 200, 130, 180]

for n in nodes:
    spmax2, spmin2 = [], []
    time = []
    for time_index in range(int(t_int/dt)+1):
        spmaxl = stress.principal_max(SIG[:, 0, time_index],
                                      SIG[:, 1, time_index],
                                      SIG[:, 2, time_index])
        spminl = stress.principal_min(SIG[:, 0, time_index],
                                      SIG[:, 1, time_index],
                                      SIG[:, 2, time_index])
        spmax2.append(spmaxl[n]/1e6)
        spmin2.append(spminl[n]/1e6)
        time.append(time_index*dt/(60*60))
    fig, ax = plt.subplots()
    ax.plot(time, spmax2, '-b', label=r'$\sigma_1$')
    ax.plot(time, spmin2, '-g', label=r'$\sigma_2$')
    ax.set_title('Principal Stresses at node '+str(n))
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Principal stress (MPa)')
    plt.legend()
# plotter.model(model, nodes_label=True)
plt.show()


"""
PLOT SPMAX, SPMIN, T AND U FIELD AT SPECIFIC TIMES
"""
time = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44]  # in [h]


T_point = []
time_point = []
for ti in time:
    time_index = int((t_int/dt)/t_int*(60*60*ti))

    """
    PRINCIPAL STRESS MAXIMUM
    """
    # spmax = stress.principal_max(SIG[:, 0, time_index],
    #                              SIG[:, 1, time_index],
    #                              SIG[:, 2, time_index])
    # print(np.amax(spmax)/1e6, np.amin(spmax)/1e6)
    # plotter.stresses(model, SIG[:, :, time_index], ftr=1e6, spmax=True,
    #                  vmin=-5.9, vmax=5.5, lev=20, title='Time: '+str(ti)+'h')

    """
    PRINCIPAL STRESS MINIMUM
    """
    # spmin = stress.principal_min(SIG[:, 0, time_index],
    #                              SIG[:, 1, time_index],
    #                              SIG[:, 2, time_index])
    # print(np.amax(spmin)/1e6, np.amin(spmin)/1e6)
    # plotter.stresses(model, SIG[:, :, time_index], ftr=1e6, spmin=True,
    #                  vmin=-12.2, vmax=0, lev=20, title='Time: '+str(ti)+'h')

    # DISPLACEMENT
    plotter.model_deformed(model, U[:, time_index], magf=1000, color='r')
    plt.gca().set_title('Time: '+str(ti)+'h')

    # # TEMPERATURE
    # T = steadystate.solver(model, material, b_heat, flux_bc,
    #                        temp_bc, t=ti*60*60)
  #   T_point.append(T[105])
#     time_point.append(ti)
#     plottert.contour(model, T, title='Time: '+str(ti)+'h', vmin=0, vmax=90)

    plt.savefig('displ'+str(ti)+'h.pdf', bbox_inches='tight')

# fig, ax = plt.subplots()
# ax.plot(time_point, T_point, '-b')
# ax.set_xlabel('Time (h)')
# ax.set_ylabel('Temperature (C)')
# plt.savefig('T_evo.pdf', bbox_inches='tight')

plt.show()




"""
PLOT ANIMATIONS OF STRESS AND DISPLACEMENT
"""
# plotter.stress_animation(SIG, model, t_int, dt, spmax=True,
#                          name="stre-cilamce1.gif", vmin=-50, vmax=150,
#                          interval=100, ftr=1e6)

# plotter.displ_animation(U, model, t_int, dt, name='00-displ.gif', magf=100,
#                         interval=100)
