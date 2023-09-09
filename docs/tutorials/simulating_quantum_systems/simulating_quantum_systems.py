import torch
import dynamiqs as dq
import matplotlib.pyplot as plt
plt.style.use('rgpapers')

psi0 = dq.fock(2, 0)

delta = 0.3  # detuning
Omega = 1.   # Rabi frequency
H = delta * dq.sigmaz() + Omega * dq.sigmax()

sim_time = 10.  # total time of evolution
num_save = 300  # number of time slots to save
t_save = torch.linspace(0., sim_time, num_save)

seresult = dq.sesolve(H, psi0, t_save)
print(seresult)

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.plot(t_save, seresult.states[:, 0, 0].abs()**2, label=r'Ground')
ax.plot(t_save, seresult.states[:, 1, 0].abs()**2, label=r'Excited')
ax.set_xlabel('Time')
ax.set_ylabel('State population')
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.legend(frameon=True)
plt.show()

fig.savefig('rabiflopping.jpg', dpi=300,bbox_inches='tight')
