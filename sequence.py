import numpy as np
import matplotlib.pyplot as plt

from typing import List
from constants import GAMMA, PI, GMAX

class sequence:
   
    def __init__(self, dt: float, TR: float) -> None:
        self.dt = dt
        self.TR = TR
        self.time_axis = np.arange(0, TR, dt)
        N = len(self.time_axis)
        self.B1 = np.zeros(N, dtype=np.complex128) # Tesla
        self.Gx = np.zeros(N) # Tesla / m
        self.Gy = np.zeros(N) # Tesla / m
        self.Gz = np.zeros(N) # Tesla / m
        self.adc = np.zeros(N) # 1 or 0 mask


    def set_custom_sequence(self, 
                            B1: np.ndarray, 
                            Gx: np.ndarray, 
                            Gy: np.ndarray, 
                            Gz: np.ndarray, 
                            adc: np.ndarray) -> None:
                            
        # Make sure all lengths are the same
        assert len(B1) == len(Gx)
        assert len(Gy) == len(Gz)
        assert len(Gz) == len(adc)

        # Set terms
        self.B1 = B1 
        self.Gx = Gx
        self.Gy = Gy
        self.Gz = Gz
        self.adc = adc
    

    def set_gre(self, ky: float, kx_range: list, TE: float):

        # Find index of middle of readout
        rf_peak_ind = np.argmax(np.abs(self.B1))
        TE_inds = int(TE / self.dt)
        ro_middle = rf_peak_ind + TE_inds

        # Find start of readout gradient
        Gx = GMAX
        kx_total = kx_range[1] - kx_range[0]
        ro_time = 2 * PI * kx_total / Gx / GAMMA
        ro_inds = int(ro_time / self.dt)
        gx_start = ro_middle - ro_inds

        # Set Gx sequence
        Gx_sig = -Gx * np.ones(ro_inds // 2)
        Gx_sig = np.append(Gx_sig, np.ones(ro_inds) * Gx)
        self.Gx[gx_start:gx_start + len(Gx_sig)] = Gx_sig

        # Set Gy sequence
        gy_time = ro_time / 2
        gy_inds = int(gy_time / self.dt)
        Gy = 2 * PI * ky / gy_time / GAMMA
        self.Gy[gx_start:gx_start + gy_inds] = Gy * np.ones(gy_inds)


    def set_excitation_pulses(self, 
                              alpha: float, # deg 
                              tau: float, # sec
                              phase: float, # dec
                              z: float, # m
                              dz: float, # m
                              t_start: float, # sec
                              TBW: float) -> None:
        
        # Make sure the pulses fit within the TR
        assert t_start + tau * 1.5 < self.TR

        # Start/stop indices
        start = np.argmin(np.abs(self.time_axis - t_start))
        stop = np.argmin(np.abs(self.time_axis - t_start - tau))

        # Calculate B1 term
        alpha_rad = alpha * PI / 180
        phase_rad = phase * PI / 180
        B1 = np.exp(1j * phase_rad) * alpha_rad / GAMMA

        # Make truncated sinc
        BW = TBW / tau
        Gz = 2 * PI * BW / GAMMA / dz
        if Gz > GMAX:
            print('Warning, max gradient exceeded')
        t = np.arange(-tau/2, tau/2, self.dt)
        sinc = np.sinc(t * BW)
        sinc /= np.sum(sinc) * self.dt
        
        # Apply offset frequency
        f_ofs = GAMMA * z * Gz / (2 * PI)
        sinc = np.exp(2j * PI * f_ofs * t) * sinc
        
        # Set B1 and gradient
        self.B1[start:stop] = B1 * sinc
        self.Gz[start:stop] = Gz
        self.Gz[stop:stop + (stop - start)//2] = -Gz
    
    def sequence_figure(self, sigs: List[str]=['B1', 'Gx', 'Gy', 'Gz', 'adc']):

        # Create time axis [msec]
        time_axis = np.arange(0, self.TR, self.dt) * 1e3
        
        # Subplots for N Signals
        plt.figure()
        N = len(sigs)
        for i, sig in enumerate(sigs):
            plt.subplot(N, 1, i + 1)
            arr = eval(f'self.{sig}')
            if 'B' in sig:
                title = 'B1 [mTesla]'
                plt.plot(time_axis, 1e3 * arr.real)
            elif 'G' in sig:
                title = f'{sig} [mTesla / m]'
                plt.plot(time_axis, 1e3 * arr)
            else:
                plt.plot(time_axis, arr)
                title = 'ADC'
            plt.title(title)
            plt.axhline(0, color='black', alpha=0.4)
        plt.xlabel('Time [msec]')