import torch
import numpy as np
from mpmath import hyp1f1,gamma,exp,power
from itertools import product
from scipy import interpolate

from .interpolator import NaturalCubicSpline

def fit_splined_radial_integrals(nmax, lmax, rc, sigma, cutoff, mesh_size):
    c = 0.5 / sigma**2
    length, channels = mesh_size, nmax * lmax

    dists = np.linspace(0, rc + 1e-3, length, dtype=np.float32)
    x = o_ri_gto(rc, nmax, lmax, dists, c).reshape((length, lmax, nmax))
    x *= cutoff(torch.from_numpy(dists)).numpy()[:, None, None]
    coeffs = torch.zeros(((4, length - 1, lmax, nmax)))
    for l in range(lmax):
        for n in range(nmax):
            ispl = interpolate.CubicSpline(dists, x[:, l, n], bc_type="natural")
            for i in range(4):
                coeffs[i, :, l, n] = torch.from_numpy(ispl.c[-i - 1])

    coeffs = coeffs.view(4, length - 1, -1)
    coeffs = (
        torch.from_numpy(dists),
        coeffs[0],
        coeffs[1],
        coeffs[2],
        coeffs[3],
    )
    return coeffs


def splined_radial_integrals(nmax, lmax, rc, sigma, cutoff, mesh_size=600):
    coeffs = fit_splined_radial_integrals(nmax, lmax, rc, sigma, cutoff, mesh_size)
    Rnl = NaturalCubicSpline(coeffs)
    return Rnl

def dn(n,rcut,nmax):
    sn = rcut*max(np.sqrt(n),1)/nmax
    return 0.5/(sn)**2

def gto_norm(n,rcut,nmax):
    d = dn(n,rcut,nmax)
    return 1/np.sqrt(np.power(d,-n-1.5)*np.power(2,-n-2.5)*float(gamma(n+1.5)))

def ortho_Snn(rcut,nmax):
    Snn = np.zeros((nmax,nmax))
    norms = np.array([gto_norm(n,rcut,nmax) for n in range(nmax)])
    ds = np.array([dn(n,rcut,nmax) for n in range(nmax)])
    for n,m in product(range(nmax),range(nmax)):
        Snn[n,m] = norms[n]*norms[m]*0.5*np.power(ds[n]+ds[m], -0.5*(3+n+m)) * float(gamma(0.5*(3+m+n)))
    eigenvalues, unitary = np.linalg.eig(Snn)
    diagoverlap = np.diag(np.sqrt(eigenvalues) )
    newoverlap = np.dot(np.conj(unitary) ,np.dot(diagoverlap, unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)
    return orthomatrix

def gto(rcut,nmax, r):
    ds = np.array([dn(n,rcut,nmax) for n in range(nmax)])
    ortho = ortho_Snn(rcut,nmax)
    norms = np.array([gto_norm(n,rcut,nmax) for n in range(nmax)])
    res = np.zeros((r.shape[0],nmax))
    for n in range(nmax):
        res[:,n] = norms[n]*np.power(r,n)*np.exp(-ds[n]*r**2)
    res = np.dot(res,ortho)
    return res

def ri_gto(n,l,rij,c,d, norm):
    res = exp(-c*rij**2)*(gamma(0.5*(l+n+3))/gamma(l+1.5)) * power(c*rij,l) * power(c+d,-0.5*(l+n+3))
    res *= hyp1f1(0.5*(n+l+3),l+1.5,power(c*rij,2)/(c+d))
    return norm*float(res)

def o_ri_gto(rcut,nmax,lmax,rij,c):
    ds = np.array([dn(n,rcut,nmax) for n in range(nmax)])
    norms = np.array([gto_norm(n,rcut,nmax) for n in range(nmax)])
    ortho = ortho_Snn(rcut,nmax)
    res = np.zeros((rij.shape[0],lmax,nmax))
    for ii,dist in enumerate(rij):
        for l in range(lmax):
            for n in range(nmax):
                res[ii,l,n] = ri_gto(n,l,float(dist),c,ds[n],norms[n])
    res = np.dot(res,ortho)
    return res