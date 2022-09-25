import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from constants import GAMMA, PI
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class spin:

    def __init__(self, T1: float, T2: float, pos: np.ndarray, M0: float = 1):
        
        # Store
        self.T1 = T1
        self.T2 = T2
        self.pos = pos

        # Magnetization xy component
        self.Mxy = 0 + 0j

        # Magnetization z component
        self.M0 = M0
        self.Mz = M0
    
    def d_Mxy_dt(self, Bxy: np.complex128, Bz: float) -> np.complex128:
        Mxy = self.Mxy
        Mz = self.Mz
        return -1j * GAMMA * Bz * Mxy + 1j * GAMMA * Bxy * Mz - Mxy / self.T2

    def d_Mz_dt(self, Bxy: np.complex128) -> float:
        Mxy = self.Mxy
        Mz = self.Mz
        M0 = self.M0
        return GAMMA * (Mxy.real * Bxy.imag - Mxy.imag * Bxy.real) - (Mz - M0) / self.T1

    def fix_lenth(sig: np.ndarray, desired_len: int) -> np.ndarray:
        diff = desired_len - len(sig)
        if diff < 0:
            return sig[:desired_len]
        elif diff > 0:
            return np.append(sig, np.zeros(diff, dtype=sig.dtype))
        else:
            return sig + 0.0    
    
    def sim_bloch(self, dt: float, duration: float, Bxy: np.ndarray, Bz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Fix Bxy and Bz lengths to length of time axis
        time_axis = np.arange(0, duration, dt)
        N = len(time_axis)
        Bxy = spin.fix_lenth(Bxy, N)
        Bz = spin.fix_lenth(Bz, N)

        # Return temporal dynamics
        Mxys = np.zeros(N, dtype=np.complex128)
        Mzs = np.zeros(N)

        for i, t in enumerate(time_axis):

            # Progress print
            if (i + 1) % (N // 100) == 0:
                print(f'Simulation is {100 * (i + 1) / N:02.2f}% Done', end='\r', flush=True)

            # Calculate derivatives
            Mxy_der = self.d_Mxy_dt(Bxy[i], Bz[i])
            Mz_der =  self.d_Mz_dt(Bxy[i])
            
            # Update
            self.Mxy += dt * Mxy_der
            self.Mz += dt * Mz_der

            # Store
            Mxys[i] = self.Mxy
            Mzs[i] = self.Mz
        
        return time_axis, Mxys, Mzs

    def spin_figure_longitudal(self, time_axis: np.ndarray, Mzs: np.ndarray):

        # Make sure lengths are okay
        assert len(time_axis) == len(Mzs)
        
        # Plot Mz
        plt.figure()
        plt.title('Longitudal')
        plt.ylabel(r'$M_z$')
        plt.xlabel('Time [msec]')
        plt.plot(time_axis * 1e3, Mzs)

    def spin_figure_transverse(self, time_axis: np.ndarray, Mxys: np.ndarray):

        # Make sure lengths are okay
        assert len(time_axis) == len(Mxys)

        # Plot magnitude and phase
        plt.figure()
        plt.subplot(211)
        plt.title('Magnitude Transverse')
        plt.ylabel(r'$|M_{xy}|$')
        plt.xlabel('Time [msec]')
        plt.plot(time_axis * 1e3, np.abs(Mxys))
        plt.subplot(212)
        plt.title('Phase Transverse')
        plt.ylabel(r'$\angle{M_{xy}}$ $[^\circ]$')
        plt.xlabel('Time [msec]')
        plt.plot(time_axis * 1e3, np.angle(Mxys) * 180 / PI)
        