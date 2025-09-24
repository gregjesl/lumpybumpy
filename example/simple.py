from lumpybumpy.celestial import LUNA
from lumpybumpy.kepler import KeplerianOrbit
import lumpybumpy.lagrange
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MASSFRAC = 0.0001
MU_CENTRAL = LUNA.mu * (1.0 - MASSFRAC)
MU_POINT = LUNA.mu * MASSFRAC
POINT_R = np.array([LUNA.radius, 0.0, 0.0])

def dynamics(t, y):
    r = y[:3]
    v = y[3:]

    # Compute the main acceleration
    central = lumpybumpy.lagrange.accel(MU_CENTRAL, r)
    point_r = LUNA.fixed_to_inertial(t, POINT_R)
    point = lumpybumpy.lagrange.accel(MU_POINT, r, point_r)
    return np.concatenate((v, central + point))

def main():
    # 40x60 orbit about the moon
    orbit0 = KeplerianOrbit.from_altitudes(
        LUNA.mu,
        LUNA.radius_of_altitude(40.0),
        LUNA.radius_of_altitude(60.0),
        math.radians(90.0),
        math.radians(90.0),
        0.0
        )
    
    pos, vel = orbit0.to_position_velocity(0.0)
    x0 = np.concatenate((pos, vel))
    print(x0)

    tspan = (0, LUNA.period * 2.0 * 3600.0)
    
    sol = solve_ivp(dynamics, tspan, x0, method='DOP853', rtol = 1e-9)

    ecc = []
    for i in range(len(sol.t)):
        r = np.array([sol.y[0][i], sol.y[1][i], sol.y[2][i]])
        v = np.array([sol.y[3][i], sol.y[4][i], sol.y[5][i]])
        orbit, ecc_anom = KeplerianOrbit.from_position_velocity(
            LUNA.mu,
            r,
            v
        )
        ecc.append(orbit.e())

    plt.plot(sol.t, ecc)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Orbital Trajectory')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()