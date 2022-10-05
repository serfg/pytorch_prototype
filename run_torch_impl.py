import torch
from tqdm import tqdm
from ase.io import read, write
import numpy as np
import copy
from ase.build import bulk, make_supercell

from pytorch_prototype.full_torch.invariants import PowerSpectrum
from pytorch_prototype.full_torch.neighbor_list import ase2data
from torch_geometric.loader import DataLoader


rc = 5
gs = 0.3
lmax = 5
nmax = 6
cutoff_smooth_width = 0.5
normalize = True
zeta = 4
device = 'cuda'

def bulk_metal():
    frames = [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Si", "fcc", a=6, cubic=True),
        bulk("Si", "bcc", a=6, cubic=True),
        bulk("Cu", "fcc", a=3.6),
        bulk("Cu", "fcc", a=6, c=3),
        bulk("Si", "bct", a=6, c=3),
        bulk("Bi", "rhombohedral", a=6, alpha=20),
        bulk("Bi", "fcc", a=6),
        bulk("Bi", "bcc", a=6),
        bulk("SiCu", "rocksalt", a=6),
        bulk("SiBi", "rocksalt", a=6),
        bulk("CuBi", "rocksalt", a=6),
        bulk("SiBiCu", "fluorite", a=6),
    ]
    return frames


aa = torch.arange(1, 6)
Ps = torch.cartesian_prod(aa, aa, aa)
Ps = Ps[torch.sort(Ps.sum(dim=1)).indices].to(torch.long).numpy()
frames = []
for frame in bulk_metal():
    for P in Ps:
        frames.append(make_supercell(frame, np.diag(P)))

sps = []
for ff in frames:
    ff.wrap(eps=1e-10)
    sps.extend(ff.get_atomic_numbers())
nsp = len(np.unique(sps))
species = torch.from_numpy(np.unique(sps)).to(dtype=torch.int32)
data_list = [ase2data(ff) for ff in frames]
dataloader = DataLoader(data_list, batch_size=100, shuffle=True)


calculator = PowerSpectrum(nmax, lmax, rc, gs, species, normalize=normalize).to(device=device)

weights = torch.randn((calculator.size(), 1), device=device, dtype=torch.float64)


energies = []
forces = []
for data in tqdm(dataloader, desc='Compute the PowerSpectrum'):
    data.pos.requires_grad_(True)
    data.to(device)

    ps = calculator(data)
    Y = ps @ weights
    F = - torch.autograd.grad(
                Y.sum(),
                data.pos,
            )[0]
    energies.append(Y)
    forces.append(F)


energies = torch.cat(energies, dim=0)
forces = torch.cat(forces, dim=0)

