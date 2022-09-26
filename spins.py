import numpy as np
import matplotlib.pyplot as plt

from sequence import sequence
from typing import Optional, Tuple
from constants import GAMMA, PI

class spins:

    def __init__(self, res: list, fovs: list, M0: float = 1):
        
        # Resolutions/fovs as x, y, z res
        assert len(res) == 3
        assert len(fovs) == 3
        self.fovs = fovs
        self.res = res

        # Make arrays for 3 spatial dimensions
        xs = np.arange(-fovs[0]/2, fovs[0]/2, res[0])
        ys = np.arange(-fovs[1]/2, fovs[1]/2, res[1])
        zs = np.arange(-fovs[2]/2, fovs[2]/2, res[2])
        self.xs, self.ys, self.zs = xs, ys, zs
        self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs)

        # Magnetization xy component
        self.Mxys = np.zeros_like(self.X, dtype=np.complex128)

        # Magnetization z component
        self.M0 = M0
        self.Mzs = np.ones_like(self.X) * M0
    

    def set_parameter_maps(self, 
                           PD: Optional[np.ndarray] = None,
                           T1: Optional[np.ndarray] = None, 
                           T2: Optional[np.ndarray] = None, 
                           B0: Optional[np.ndarray] = None):

        # Default values
        if PD is None:
            # Ellipse thing
            a = self.fovs[0] * 0.4
            b = self.fovs[1] * 0.4
            c = self.fovs[2] * 0.4
            PD = (self.X / a) ** 2 + (self.Y / b) ** 2 + (self.Z / c) ** 2 < 1
            PD = PD * 1
        else:
            assert PD.shape == self.X.shape
        if T1 is None:
            # Linearly varying T1 values along X
            T1 = self.X - self.X.min()
            T1 = T1 / T1.max()
            T1 = T1 * 100e-3 + 100e-3
        else:
            assert T1.shape == self.X.shape
        if T2 is None:
            # Linearly varying T2 values along Y
            T2 = self.X - self.X.min()
            T2 = T2 / T2.max()
            T2 = T2 * 70e-3 + 30e-3
        else:
            assert T2.shape == self.X.shape
        if B0 is None:
            B0 = self.X * 0
        else:
            assert B0.shape == self.X.shape

        # Store
        self.PD = PD
        self.T1 = T1
        self.T2 = T2
        self.B0 = B0


    def d_Mxy_dt(self, Bxy: np.complex128, Bz: np.ndarray) -> np.ndarray:
        Mxys = self.Mxys
        Mzs = self.Mzs
        return -1j * GAMMA * Bz * Mxys + 1j * GAMMA * Bxy * Mzs - Mxys / self.T2


    def d_Mz_dt(self, Bxy: np.complex128) -> np.ndarray:
        Mxys = self.Mxys
        Mzs = self.Mzs
        M0 = self.M0
        return GAMMA * (Mxys.real * Bxy.imag - Mxys.imag * Bxy.real) - (Mzs - M0) / self.T1

    
    def sim_bloch(self, dt: float, duration: float, seq: sequence, pos_probe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # We can extract Bxy and Bz from the pulse sequence
        Bxy = seq.B1
        Bz  = seq.Gx[np.newaxis, np.newaxis, np.newaxis, :] * self.X[:, :, :, np.newaxis]
        Bz += seq.Gy[np.newaxis, np.newaxis, np.newaxis, :] * self.Y[:, :, :, np.newaxis]
        Bz += seq.Gz[np.newaxis, np.newaxis, np.newaxis, :] * self.Z[:, :, :, np.newaxis]

        # Fix Bxy and Bz lengths to length of time axis
        time_axis = np.arange(0, duration, dt)
        N = len(time_axis)

        # Return temporal dynamics
        Mxys = np.zeros(N, dtype=np.complex128)
        Mzs = np.zeros(N)
        x_ind = np.argmin(np.abs(self.xs - pos_probe[0]))
        y_ind = np.argmin(np.abs(self.ys - pos_probe[1]))
        z_ind = np.argmin(np.abs(self.zs - pos_probe[2]))


        for i, t in enumerate(time_axis):

            # Progress print
            if (i + 1) % (N // 100) == 0:
                if i + 1 == N:
                    end = '\n'
                else:
                    end = '\r'
                print(f'Simulation is {100 * (i + 1) / N:.2f}% Done', end=end, flush=True)

            # Calculate derivatives
            Mxy_ders = self.d_Mxy_dt(Bxy[i], Bz[:, :, :, i])
            Mz_ders =  self.d_Mz_dt(Bxy[i])
            
            # Update
            self.Mxys += dt * Mxy_ders
            self.Mzs += dt * Mz_ders

            # # Store
            Mxys[i] = self.Mxys[y_ind, x_ind, z_ind]
            Mzs[i] = self.Mzs[y_ind, x_ind, z_ind]

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
        