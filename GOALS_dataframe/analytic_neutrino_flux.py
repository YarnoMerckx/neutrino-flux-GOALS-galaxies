import matplotlib.pyplot as plt
import numpy as np 
from astropy import units as u 
from astropy import constants as const
from scipy.integrate import quad

####
## Injection
####

def cross_section(E):
    L = np.log(E/1e3)
    E_th = 1.22
    return (34.3+1.88*L+0.25*pow(L,2))*pow((1-pow((E_th/E),4)),2)

def E_CR(RSN): #RSN in yr^-1
    E_SN_erg = 1e51*u.erg #characteristic energy output
    E_SN_GeV = (E_SN_erg.to(u.eV)).value*1e-9 # GeV
    xi = 0.10 # efficiency ~10%
    R_sn_yr = RSN*pow(u.yr,-1) #SN rate in yrs
    R_sn_s = (R_sn_yr.to(pow(u.second,-1))).value
    return R_sn_s*E_SN_GeV*xi #SN rate in GeV/seconds


def I(p,alpha,p_max):
    mp = 0.938
    return (4*np.pi)*pow(p,2)*pow(p/mp,-alpha)*(np.sqrt(pow(p,2)+pow(0.938,2))-0.938)*np.exp(-p/p_max)

def Qp(p,R,h,plow,pup,alpha,pmax,RSN):
    mp = 0.938
    R_pc = R*u.parsec  #pc to cm conversion
    R_cm = (R_pc.to(u.cm)).value
    h_pc = h*u.pc
    h_cm = (h_pc.to(u.cm)).value
    if h == 0: 
        V_spherical = (4/3)*np.pi*pow(R_cm,3)
        V_SBN = V_spherical
    else:
        V_disk = 2*h_cm*np.pi*pow(R_cm,2)
        V_SBN = V_disk
    mom = np.logspace(np.log10(plow),np.log10(pup),100000)
    Int = np.trapz(I(mom,alpha,pmax),mom)
    N = E_CR(RSN)/Int
    return (N/V_SBN)*pow(p/mp,-alpha)*np.exp(-p/pmax)


####
## Timescale
####

def loss_time(p,nism): 
    E = np.sqrt(pow(p,2)+pow(0.938,2))
    eta = 0.5
    n_cm = nism*pow(u.cm,-3)
    n_m = (n_cm.to(pow(u.m,-3))).value
    sigma = cross_section(E)*1e-31 #1e-31 converts mb -> m2
    return 1/(eta*n_m*sigma*const.c.value)


def tau_wind(R,v,h):
    if h==0:
        R_pc = R*u.parsec 
        R_SBN = (R_pc.to(u.m)).value
        v_wind = v*1000 
        return (R_SBN/v_wind) 
    else:
        h_pc = h*u.parsec
        h_m = (h_pc.to(u.m)).value
        v_wind = v*1000 
        return (h_m/v_wind) 
            

def larmor(p,B): #Larmor radius in m 
    E = np.sqrt(pow(p,2)+pow(0.938,2))
    B_G = (B*u.G)*1e-6 #Gauss
    B_T = (B_G.to(u.T)).value #Tesla
    return 3.3 *((E*1)/(1*B_T))

def W_0_trapz(k_0,d): #integration via Trapezium method
    integral = lambda k,d: pow(k,-d)
    logaxis = np.logspace(np.log10(1),np.log10(1e10),100000)
    I = np.trapz(integral(logaxis,d),logaxis)
    return pow(pow(k_0,d)*I,-1)


def F(k,k_0,d):
    return k*W_0_trapz(k_0,d)*pow(k/k_0,-d)

def D(E,k_0,B,d):
    c = (const.c).value # lightspeed in m/s
    
    k_m = 1/(larmor(E,B)*u.m) # in m^-1
    
    k_pc = (k_m.to(pow(u.parsec,-1))).value # in pc^-1
    
    D_m2_s = ((larmor(E,B)*c)/(3*F(k_pc,k_0,d)))*(pow(u.m,2)/u.s) #in m^2/s
    
    D_pc2_s = (D_m2_s.to(pow(u.pc,2)/u.s)).value
    
    return D_pc2_s # in pc^2/s

def tau_diff_quasi(p,R):
    #! B = 250 muG hardcoded
    return pow(R,2)/D(p,1,250,5/3) # untit: seconds

def tau_lifetime(R,vwind,p , nism,h):
    return pow(pow(tau_wind(R,vwind,h),-1)+pow(loss_time(p,nism),-1) +pow(tau_diff_quasi(p,R),-1),-1)


####
## Momentum distribution
####

def f_p(p,R,v,nism,h,plow,pup,alpha,pmax,RSN):
    return tau_lifetime(R,v,p, nism,h)*Qp(p,R,h,plow,pup,alpha,pmax,RSN)



####
## Neutrino disribution functions
####


def Fmu1(x,Ep):
    if x <= 0.427:
        L= np.log(Ep/1e3)
        y = x/0.427
        B = 1.75+0.204*L+0.010*pow(L,2)
        beta = pow(1.67+0.111*L+0.0038*pow(L,2),-1)
        k = 1.07-0.086*L+0.002*pow(L,2)
        first =  B*(np.log(y)/y)*pow((1-pow(y,beta))/(1+k*pow(y,beta)*(1-pow(y,beta))),4)
        second = (1/np.log(y))-((4*beta*pow(y,beta))/(1-pow(y,beta)))-((4*k*beta*pow(y,beta)*(1-2*pow(y,beta)))/(1+k*pow(y,beta)*(1-pow(y,beta))))
        return first*second
    else:
        return 0
Fmu1_vec = np.vectorize(Fmu1)



def Fe(x,Ep):
    L = np.log(Ep/1e3)
    Be = pow(69.5+2.65*L+0.3*pow(L,2),-1)
    betae= pow(0.201+0.062*L+0.00042*pow(L,2),-1/4)
    ke = (0.279+0.141*L+0.0172*pow(L,2))/(0.3+pow(2.3+L,2))
    first = (pow(1+ke*pow(np.log(x),2),3))/(x*(1+0.3/pow(x,betae)))
    second = pow(-np.log(x),5)
    return Be*first*second
Fe = np.vectorize(Fe)


def Ftot(x,Ep): 
    return 2*Fe(x,Ep)+Fmu1(x,Ep)

Ftot = np.vectorize(Ftot)


####
## q
####

def q(E_nu,R,v,nism,H,gammasn,pmax,RSN):
    x = np.logspace(np.log10(0.0001),np.log10(1),1000)
    c= 3e8*100 #cm/s
    p = np.sqrt(pow(E_nu/x,2)-pow(0.938,2))
    I = np.trapz(Ftot(x,E_nu/x)*cross_section(E_nu/x)*1e-27*(1/x)*4*np.pi*pow(p,2)*f_p(p,R,v,nism,H,0.1,1e9,gammasn,pmax,RSN),x)
    return c*nism*I #GeV-1 cm-3 s-1

q = np.vectorize(q)



####
## Energ-squared scaled flux
####


def Flux(E_nu,R,v,nism,H,gammasn,pmax,RSN,D_L):#energy-squared scaled flux
    DL_pc = (D_L*1e6)*u.parsec #pc
    DL_cm = (DL_pc.to(u.cm)).value #cm
    R_pc = R*u.pc #pc
    R_cm = (R_pc.to(u.cm)).value#cm
    H_pc = H*u.parsec#pc
    H_cm = (H_pc.to(u.cm)).value#cm

    if H == 0:
        V = (4/3)*np.pi*pow(R_cm,3)
        
    if H != 0:
        V = 2*H_cm*np.pi*pow(R_cm,2)
    
    scaled_flux = (1/3)*(V/(4*np.pi*pow(DL_cm,2)))*pow(E_nu,2)*q(E_nu,R,v,nism,H,gammasn,pmax,RSN)# factor 1/3 for oscillations
    
    return scaled_flux
