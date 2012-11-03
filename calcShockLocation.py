# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:14:01 2012

@author: sousae
"""
from math import *
import scipy.optimize
from numpy import *
from pylab import *

# ============= constant input values
Mi = 1.25
gm = 1.4
Ai = 1.398 + 0.347*tanh(0.8*(0.)-4.0)
Ae = 1.398 + 0.347*tanh(0.8*(10.)-4.0)
gm1= gm-1.
gp1= gm+1.

# ============= constant input values
r_inf = 1.0
P_inf = 0.75
a_inf = sqrt(gm*P_inf/r_inf)

# ============= the only inputs
pi = 0.75/(r_inf*a_inf*a_inf)
pe = 1.1

# ============= calculations
A_ratio = sqrt(1./Mi**2*(2./gp1*(1.+0.5*gm1*Mi**2))**(gp1/gm1))
At   = Ai/A_ratio
pc   = pi*(1.+0.5*gm1*Mi**2)**(gm/gm1)

pep01 = pe/pc
AeAt  = Ae/At

Pres = pep01*AeAt

def ME(x):
    return 1./x*(2/gp1)**(gp1/(2.*gm1))*(1.+0.5*gm1*x**2)**(-0.5) - Pres
    
#sol1 = fsolve(ME, 10.0)
sol1 = scipy.optimize.broyden1(ME, [1.0], f_tol=1e-14)
Me = sol1[0]
pep0e = (1.+0.5*gm1*Me**2)**(-gm/gm1)

p2p1 = pep01/pep0e

# ============= solve for the shock Mach number
def F(x):
    
    f = (gp1*x**2/(gm1*x**2+2))**(gm/gm1)*(gp1/(2.*gm*x**2-gm1))**(1./gm1) - p2p1
    
    return f
    
#sol2 = fsolve(F, 10.0)
sol2 = scipy.optimize.broyden1(F, [1.0], f_tol=1e-14)
Ms = sol2[0]

# ============= find area at shock location
A_r2 = sqrt(1./Ms**2*(2./gp1*(1.+0.5*gm1*Ms**2))**(gp1/gm1))
As = A_r2*At
print As
# ============= find the shock location
xloc = (atanh((As-1.398)/.347)+4.0)/0.8

print xloc
