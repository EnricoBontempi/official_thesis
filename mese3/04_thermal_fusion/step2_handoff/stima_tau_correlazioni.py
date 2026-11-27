import cantera as ct, numpy as np
sig=5.670e-8
def gas_props(T, phi=0.9):
    g=ct.Solution('gri30.yaml'); g.set_equivalence_ratio(phi,'CH4:1','O2:1,N2:3.76'); g.TP=300,ct.one_atm
    g.equilibrate('HP'); Tad=g.T; g.TP=T,ct.one_atm; g.equilibrate('TP')
    return g.density, g.viscosity, g.thermal_conductivity, g.cp_mass, Tad
def nu_cyl_cb(Re,Pr): return 0.3+0.62*Re**0.5*Pr**(1/3)/(1+(0.4/Pr)**(2/3))**0.25*(1+(Re/282000)**(5/8))**0.8
def nu_sph_rm(Re,Pr): return 2+0.6*Re**0.5*Pr**(1/3)
rows=[]
for T in (1000,1500,1900):
  rho,mu,k,cp,Tad=gas_props(T); Pr=mu*cp/k
  for name,d,geom,rhoc,eps in [("filo Pt-Rh 50 um",50e-6,'cyl',3.0e6,0.2),("filo 76 um (T, 40 AWG)",76e-6,'cyl',3.6e6,None),
      ("bead Pt-Rh 0.3 mm",0.3e-3,'sph',3.0e6,0.2),("bead K 0.762 mm",0.762e-3,'sph',4.2e6,None)]:
    for u in (2,10,30):
      Re=rho*u*d/mu; Nu=nu_cyl_cb(Re,Pr) if geom=='cyl' else nu_sph_rm(Re,Pr); h=Nu*k/d
      tau=rhoc*d/(4*h) if geom=='cyl' else rhoc*d/(6*h)
      drad=(eps*sig*(T**4-600**4)/h) if eps else float('nan')
      rows.append((T,name,u,Re,Nu,h,tau*1e3,drad))
print("Tad CH4/air phi=0.9:",round(Tad))
print("T_gas[K] | sensore | u[m/s] | Re | Nu | h[W/m2K] | tau[ms] | dT_rad[K] (eps=0.2, Tw=600K)")
for r in rows: print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.0f} | {r[6]:.1f} | {r[7]:.0f}")
