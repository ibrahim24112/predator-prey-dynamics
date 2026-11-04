import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
DATA = ROOT / 'data' / 'Leigh1968_harelynx.csv'
FIG.mkdir(exist_ok=True)

# Logistic prey growth with Holling-I (mass-action) predation.
# H: prey (hares), L: predator (lynx)
r, K, a, b, m = 0.55, 1200.0, 0.012, 0.035, 0.25

def rhs(z):
    H, L = z
    return np.array([r*H*(1-H/K) - a*H*L,
                     b*a*H*L - m*L])

def rk4(z0, t):
    out = np.empty((len(t), 2), float)
    out[0] = z0
    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        z = out[i]
        k1 = rhs(z)
        k2 = rhs(z + 0.5*h*k1)
        k3 = rhs(z + 0.5*h*k2)
        k4 = rhs(z + h*k3)
        out[i+1] = np.maximum(z + h*(k1+2*k2+2*k3+k4)/6, 0)
    return out

# Coexistence equilibrium and Jacobian eigenvalues.
Hstar = m/(b*a)
Lstar = r/a*(1-Hstar/K)
J = np.array([[-r*Hstar/K, -a*Hstar], [b*a*Lstar, 0.0]])
eigs = np.linalg.eigvals(J)
print(f'H*={Hstar:.6f}, L*={Lstar:.6f}')
print('Jacobian=\n', J)
print('eigenvalues=', eigs)

# Main trajectory.
t = np.linspace(0, 120, 2401)
z = rk4(np.array([80.0, 15.0]), t)
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t, z[:,0], label='Hares (H)', lw=2)
ax[0].plot(t, z[:,1], label='Lynx (L)', lw=2)
ax[0].axhline(Hstar, color='C0', ls='--', alpha=.6)
ax[0].axhline(Lstar, color='C1', ls='--', alpha=.6)
ax[0].set_ylabel('Scaled population')
ax[0].set_title('Predator–prey dynamics: logistic prey model')
ax[0].legend(); ax[0].grid(alpha=.25)
ax[1].plot(z[:,0], z[:,1], color='purple', lw=1.5)
ax[1].scatter([Hstar], [Lstar], color='black', zorder=4, label='Coexistence equilibrium')
ax[1].set_xlabel('Hares, H'); ax[1].set_ylabel('Lynx, L')
ax[1].set_title('Phase portrait (Retrato de Fase)')
ax[1].legend(); ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG/'main_dynamics.png', dpi=180); plt.close(fig)

# Compare the classical neutrally-stable LV model against the logistic extension.
def rhs_classical(z):
    H, L = z
    return np.array([r*H-a*H*L, b*a*H*L-m*L])
def rk4_rhs(z0, t, f):
    out=np.empty((len(t),2)); out[0]=z0
    for i in range(len(t)-1):
        h=t[i+1]-t[i]; z=out[i]
        k1=f(z); k2=f(z+.5*h*k1); k3=f(z+.5*h*k2); k4=f(z+h*k3)
        out[i+1]=np.maximum(z+h*(k1+2*k2+2*k3+k4)/6,0)
    return out
zc = rk4_rhs(np.array([80.,15.]), t, rhs_classical)
fig, ax=plt.subplots(figsize=(9,4.8))
ax.plot(t, zc[:,0], label='Classical LV: hares', lw=1.8)
ax.plot(t, zc[:,1], label='Classical LV: lynx', lw=1.8)
ax.plot(t, z[:,0], '--', label='Logistic extension: hares', lw=1.3)
ax.plot(t, z[:,1], '--', label='Logistic extension: lynx', lw=1.3)
ax.set_xlabel('Time'); ax.set_ylabel('Scaled population'); ax.set_title('Classical versus stabilised model')
ax.legend(ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/'model_comparison.png', dpi=180); plt.close(fig)

# Data figure: historical pelt records are observations/proxies, not census counts.
df=pd.read_csv(DATA)
fig, ax=plt.subplots(figsize=(9,4.8))
ax.plot(df.year, df.hare/1000, label='Hare pelts (thousands)', lw=1.8)
ax.plot(df.year, df.lynx/1000, label='Lynx pelts (thousands)', lw=1.8)
ax.set_xlabel('Year'); ax.set_ylabel('Pelts (thousands)')
ax.set_title('Hudson Bay hare–lynx pelt records, 1847–1903')
ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/'historical_data.png', dpi=180); plt.close(fig)

# Simple descriptive lag/correlation diagnostics.
pearson = df.hare.corr(df.lynx)
lagcorr = df.hare.iloc[1:].reset_index(drop=True).corr(df.lynx.iloc[:-1].reset_index(drop=True))
summary = pd.DataFrame({'metric':['Pearson same-year correlation','Hare(t+1) vs Lynx(t) correlation'], 'value':[pearson,lagcorr]})
summary.to_csv(ROOT/'diagnostics.csv', index=False)
print(summary.to_string(index=False))
print('Figures written to', FIG)
# Parameter metadata for reproducibility.
pd.DataFrame({'parameter':['r','K','a','b','m','H*','L*'],'value':[r,K,a,b,m,Hstar,Lstar]}).to_csv(ROOT/'parameters.csv', index=False)
