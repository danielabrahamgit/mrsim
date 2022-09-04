import numpy as np
import matplotlib.pyplot as plt

from sequence import sequence
from spin import spin

# Simulation Paremters
dt = 1e-7 # sec
duration = 20e-3 # sec 

# Sequence object
seq = sequence(dt, TR=duration)

# A single spin object
spn = spin(T1=100e-3, T2=10e-3, pos=np.array([0, 0, 0.0]))

# Excitation pulse sequence
seq.set_excitation_pulses(
    alpha=90,
    tau=1e-3,
    phase=0,
    z=0,
    dz=1e-2,
    t_start=0,
    TBW=1)

# Simulate sequence
Bxy = seq.B1
Bz = seq.Gz * spn.pos[2]
Mxy, Mz = spn.sim_bloch(dt=dt, duration=duration, Bxy=Bxy, Bz=Bz)

seq.sequence_figure(sigs=['B1', 'Gz'])

plt.figure()
time_axis = np.linspace(0, duration, int(duration / dt))
plt.plot(time_axis * 1e3, Mz)
plt.xlabel('Time [msec]')
plt.show()