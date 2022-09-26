import numpy as np
import matplotlib.pyplot as plt

from sequence import sequence
from spins import spins

# Simulation Paremters
dt = 1e-6 # sec
duration = 100e-3 # sec 
res = [1e-2, 1e-2, 1e-2] # Resolution in XYZ dirs
fovs = [5e-2, 5e-2, 1e-2] # FOV in XYZ dirs

# Sequence object
seq = sequence(dt, TR=duration)

# A single spin object
spns = spins(res, fovs)
spns.set_parameter_maps()


# Excitation pulse sequence
seq.set_excitation_pulses(
    alpha=90,
    tau=5e-3,
    phase=0,
    z=0,
    dz=20e-2,
    t_start=0,
    TBW=6)

kx_extent = 1 / res[0]
seq.set_gre(0, [-kx_extent/2, kx_extent/2], TE=17e-3)

# Simulate sequence
pos_probe = np.array([10e-2, 0, -4e-2])
time_axis, Mxys, Mzs = spns.sim_bloch(dt=dt, duration=duration, seq=seq, pos_probe=pos_probe)

# Plot Spins
spns.spin_figure_longitudal(time_axis, Mzs)
spns.spin_figure_transverse(time_axis, Mxys)
seq.sequence_figure(sigs=['B1', 'Gz', 'Gx', 'Gy'])
plt.show()