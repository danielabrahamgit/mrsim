import numpy as np

class spin:

    # Gyromagnetic ratio rad Hz / Tesla
    gamma = 2 * np.pi * 42.58e6

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
    
    def d_Mxy_dt(self, Bxy, Bz):
        Mxy = self.Mxy
        Mz = self.Mz
        return -1j * spin.gamma * Bz * Mxy + 1j * spin.gamma * Bxy * Mz - Mxy / self.T2

    def d_Mz_dt(self, Bxy):
        Mxy = self.Mxy
        Mz = self.Mz
        M0 = self.M0
        return spin.gamma * (Mxy.real * Bxy.imag - Mxy.imag * Bxy.real) - (Mz - M0) / self.T1

        
    
    def sim_bloch(self, dt: float, duration: float, Bxy: np.ndarray, Bz: np.ndarray):
        
        # Make sure that magnetic fields are the right lengths
        N = int(duration / dt)
        assert len(Bxy) == len(Bz)
        assert N == len(Bz)

        # Make time axis
        time_axis = np.linspace(0, duration, N)

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
        
        return Mxys, Mzs

