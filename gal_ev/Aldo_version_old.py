#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import constants as const
from scipy import optimize
import math


# In[2]:


############################Constants##########################
G = 4.299E-9    #Gravitational constant Mpc Msol**-1 (km/s)**2

H0 = 100        #Today's Hubble constant km/s/Mpc

Myear = 1E6

a_year_in_seconds = math.pi * 1E7 #units of seconds

speed_of_light = 3E10 #cm/s

Msolar = 1.99E33 #units of grams


# In[3]:


#############Bolshoi-Planck Cosmological paramters##############
Om_mat_cero = 0.307

Om_lambda_cero = 0.693

Om_baryons = 0.048

sigma8  =  0.829

h  =  0.678


# In[64]:





# In[65]:


#UNITS OF DZ ARE MYRS
#AGE OF UNIVERSE ARE GYEARS


# In[66]:


#DtDz(Om_mat_cero,Om_lambda_cero,h,1)


# In[67]:


def SHMR(z, Mvirz):
    #Rodriguez-Puebla 2017
    #star to halo mass relation
    #Section 3.2 Parametrization of the SHMR (stellar to mass relation halo) Rodriguez-Puebla 2017
    #input: redshift, intial virial mass (halo)
    #units: Mvirz in solar mass
    #output; stellar mass for the given halo mass
    #units of solar mass
    
    
    M = Mvirz

    def P(x, y, z):
        return y*z - x*z/(1+z)

    def Q(z):
        return np.exp(-4/(1.+z)**2)

    def g(x, a, g, d):
        return (-np.log10(10**(-a*x)+1.) +
                d*(np.log10(1.+np.exp(x)))**g/(1.+np.exp(10**(-x))))

    al = (1.975, 0.714, 0.042)
    de = (3.390, -0.472, -0.931)
    ga = (0.498, -0.157)
    ep = (-1.758, 0.110, -0.061, -0.023)
    M0 = (11.548, -1.297, -0.026)

    #Section 5 Rodriguez-Puebla 2017
    #Constrains for the model. Madau & Dickinson (2014)
    
    alpha = al[0] + P(al[1], al[2], z) * Q(z)
    delta = de[0] + P(de[1], de[2], z) * Q(z)
    gamma = ga[0] + P(ga[1], 0, z) * Q(z)

    eps = 10**(ep[0] + P(ep[1], ep[2], z)*Q(z) + P(ep[3], 0, z))
    M1 = 10**(M0[0] + P(M0[1], M0[2], z)*Q(z))

    x = np.log10(M/M1)
    g1 = g(x, alpha, gamma, delta)
    g0 = g(0, alpha, gamma, delta)

    Ms = 10**(np.log10(eps*M1) + g1 - g0)
    
    

    return Ms


# In[68]:


######Codigo Aldo tiene dos funciones separadas para la funcion que yo tengo arriba


# In[69]:


#def halo_mass_assembly(Mvir0, z0, redshift): 
    #Halo mass growth at any z fo a progenitor mass Mvir0 at z0 
    #Apendix B2 Rodriguez-Puebla 2017
    #taken from Rodriguez-Puebla 2016a, Behroozi 2013b
    #Halo-mass assembly graph
    #inputs: Initial virial mass, redshift(0) initial time, redshift array
    #units: solar mass
    #output: halo mass growth 
    #units: solar mass
    
    #z = redshift
  
    
    #def M13(z):
        #return (10**13.6) * (1+z)**2.755 * ((1+(z/2))**-6.351) * np.exp(-0.413*z)
    
    #def aexp0(Mvir0):
    #    return 0.592 - np.log10(((10**15.7)/Mvir0)**0.113 + 1)

    #def g(Mvir0, aexp):
     #   return 1. + np.exp(-3.676*(aexp-aexp0(Mvir0)))

    #def f(Mvir0, z):
    #    aexp = 1. / (1+z)
    #    return np.log10((Mvir0)/M13(0.)) * (g(Mvir0, 1.)/g(Mvir0, aexp))
    
    #def Mvir(Mvir0, z):
    #    return M13(z) * 10**(f(Mvir0, z))
        
    #Mvir_z = Mvir(Mvir0, z-z0)
    
    #return Mvir_z

  


# In[366]:


def halo_mass_assembly(Mvir0, z0, redshift): 
    #Halo mass growth at any z fo a progenitor mass Mvir0 at z0 
    #Improve fit to ecs 18-22 from RP16 Bolshoi-Planck paper
    #Halo-mass assembly graph
    #inputs: Initial virial mass, redshift(0) initial time, redshift array
    #units: solar mass
    #output: halo mass growth 
    #units: solar mass
    
    z= redshift
    
    def a0M0(Mvir0):
        
        X = 26.6272 - Mvir0 
        
        return 1.37132 - np.log10( 10**( 0.077364 * X) + 1.)
    
    def gM0(Mvir0, a_scale):
        
        return 1. + np.exp( -3.79656 * ( a_scale - a0M0(Mvir0) )  )
    
    def M13(z):
        
        log10Mh013 = 13.
    
        alpha = 2.77292
        
        beta = -5.66267
            
        gamma = -0.469065
        
        return log10Mh013 + alpha * np.log10( 1. + z ) + beta * np.log10( 1. + 0.5 * z ) + gamma * z * np.log10( np.exp(1.) )
        
    def fM0z(Mvir, z):
        
        a_scale = 1. / (1. +z)
        
        return (Mvir0 - M13(0.)) * gM0(Mvir0, 1.) / gM0(Mvir0, a_scale)
    
    Mvir_z =  np.log(Mvir0) + np.log(h)
    
    
    return M13(z-z0) + fM0z(Mvir_z, z - z0) - np.log(h)


# In[367]:


###NUEVA
def f_int(Mvir, z):
    #Instantaneous fraction of stellar mass from Mergers
    #all type of mergers Ec; 34-36 RP17
    
    a_scale = 1. / ( 1. + z )
    
    beta_merger = 0.760 + np.log10( 1. + z )
    
    logM_merge = 12.728 - 2.790 * ( a_scale - 1. )
    
    frac_merge = 10**( beta_merger * ( log10Mvir - logM_merge ) )
    
    frac_merge = 1. / ( frac_merge + 1. )
    
    return frac_merge


# In[368]:


# The rate at which dark matter haloes grow will determine the rate at which the cosmological baryonic
# inflow material reaches the ISM of a galaxy. Eventually, when necessary conditions are satisfied, some of 
# this cosmological baryonic material will be transformed into stars. 
# As described in Section 2.2, we use the growth of dark matter haloes to predict the SFHs of galaxies
# without modelling how the cold gas in the galaxy is converted into stars.


# In[369]:


def galaxy_mass_assembly(Mvir0, z0, z):
    #stellar mass growth at any z for a progenitor halo mass Mvir0 @ z0 
    #Halo growth and stellar relationship, galaxy mass evolution
    #inputs: Initial virial mass, redshift(0), redshift array
    #units solar mass
    #output: stellar mass growth within halo
    #units: solar mass
    
    Mvirz = halo_mass_assembly(Mvir0, z0, z)
    
    Ms_z = SHMR(z, Mvirz)
    
    return Ms_z


# In[370]:


#print(AgeUniverse(Om_mat_cero, Om_lambda_cero, h, 2)) 
##########Cosmology#######
def AgeUniverse(Om_mat_cero, Om_lambda_cero, h, z):
    #output units of gyears
    #Hubble constant is H0 * h 
    

    T_Hubble = 1.02E-12

    one_plus_z = 1. + z

    Olz = Om_lambda_cero *one_plus_z**(-3)

    T1 = 2. / np.sqrt( Om_lambda_cero ) / 3. / H0 / T_Hubble / h

    T2 = np.sqrt(Olz)

    T3 = np.sqrt(Olz +  Om_mat_cero)

    TH = T1 * np.log( ( T2 + T3 ) / np.sqrt(Om_mat_cero) )

    return TH / 1E9


def DtDz(Om_mat_cero,Om_lambda_cer0, h, z):   #dT_age / dz

    dz = 0.01
    
    dt = AgeUniverse(Om_mat_cero,Om_lambda_cero,h,z) - AgeUniverse(Om_mat_cero,Om_lambda_cero,h,dz+z)
    
    return dt/dz

def DZ(z,DT): #calculates the redshift z+dz given an interval of time DT and z. DT input is in Myrs

    def Delta_T(z_dz,DT,z):
        
        DT_age = AgeUniverse(Om_mat_cero,Om_lambda_cero,h,z) - AgeUniverse(Om_mat_cero,Om_lambda_cero,h,z_dz)

        return DT - DT_age / 1E6

    z_f = optimize.bisect(Delta_T, 0, 1000, args=(DT,z))

    return z_f


# In[371]:


def dMsdz(Mvir0, z0, redshift):
    #Stellar mass formation rate
    #Derivative of Galaxy mass assembly with respect to Age of Universe
    #input initial virial mass in solar mass
    #output units of solar masses per year
    #if you are using a diff cosmology be sure to change the constants below
    
    z = np.array(redshift)
    
    zi = z
    
    zf = z - 0.01
    
    mi = galaxy_mass_assembly(Mvir0, z0, zi)
    
    mf = galaxy_mass_assembly(Mvir0, z0, zf)
    
    delm = mi - mf
    
    delz = zi - zf
    
    Ti = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zi)
    
    Tf = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zf)
    
    delT = Ti - Tf
    
    return delm / delT / 1E9


# In[372]:


def R_stell_frac(Time):
    #Given by Aldo
    #fraction Stellar mass loss that comes back to ISM in form of gas
    
    
    C0 = 0.05
    
    lam = 1.46E5 #what is this number and what are the units
    
    time = Time * 1E9
    
    return C0*np.log(time/lam + 1.) #untiless, just a fraction


# In[373]:


def SFR(Mvir0, z0, redshift):
    #Stellar mass formation rate corrected, stellar mass loss fraction included
    #Given by Aldo
    #input initial virial mass in solar mass
    #output units of solar masses per year
    
    TU = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, redshift)
    #gotta change the values if I change the cosmology
    
    sfr_gal = dMsdz(Mvir0, z0, redshift) / (1. - R_stell_frac(TU))
    
    return sfr_gal


# In[374]:


#########NUEVA 
#Star formation histories Ec 14 from RP17
def SFR_hist(SHMR, Mvir0, z0, z):

    
    time_burst = 100 
    
    z_dz = DZ(z, time_burst)
    
    log10Mvir = halo_mass_assembly(Mvir0, z0, redshift)
    
    log10Ms = galaxy_mass_assembly(Mvir0, z0, z)
    
    log10Ms_t100 = galaxy_mass_assembly(Mvir0, z0, z_dz)
    
    Dt = time_burst *Myear
    
    DMs = pow(10, Ms) - pow(10, Ms_t100)
    
    sfr = DMs/Dt
    
    if DMs<=0:
        sfr=0;
    
    return sfr * f_int(Mvir,z) / (1. - R_stell_frac(AgeUniverse(Om_mat_cero, Om_lambda_cero, h, z)))


# In[375]:


def Vmax(Mvir,z):
    #max value in DMhalo rotation curve
    #First step to introduce SN feedback
    #Rodriguez-Puebla 2016
    #Equations 4-7
    #input initial virial mass in solar mass
    #units out: km/s
    
    def alpha(z):
        
        aexp = 1. / (1+z)
        
        return 0.346 - 0.059*aexp + 0.025*aexp**2
    
    def beta(z):
        aexp = 1. / (1+z)
        return 10**(2.209 + 0.060*aexp - 0.021*aexp**2)
    
    def E(z):
        #gotta change the values if I change the cosmology
        return np.sqrt(Om_lambda_cero + Om_mat_cero*(1 + z)**3)
    
    def V(Mvir, z):
        
        z = np.array(redshift)
        
        M12 = Mvir / 1E12
        
        return beta(z) * ( M12 * E(z) )**alpha(z)
    
    return V(Mvir,z)


# In[376]:


def Vmax_assembly(Mvir0, z0, z):
    #Peak of the Halo rotation curve with halo mass growth
    #units input: solar mass
    #units outut: km/s

    Mvirz = halo_mass_assembly(Mvir0, z0, z)
    
    vmax_z = Vmax(Mvirz, z)
    
    return vmax_z


# In[377]:


def SNE_feedback(Mvir0, z0, z):
    #Given by Aldo
    #input unitls of solar mass
    #unitless, just a fraction
    
    EK = 0.5 * Vmax_assembly(Mvir0, z0, z)**2
    #kitenic energy of the halo
    
    ESN = 10**7.701147 
    #units of solar mass km^2s^-2
    
    epsilon_SN = 0.05
    #fraction of the SN energy explosion transformed into kinetic Energy, Page 403 Mo d et al. Book
    
    N_SN = 8.0E-3 
    #one supernova per 125 Msol: units solar mass^{-1}
    
    #    E_SFR = SFR(Mvir0, z0, redshift) * ESN * epsilon_SN * N_SN
    E_SFR = ESN * epsilon_SN * N_SN
    #units of km^2s^-2
    
    z = np.array(redshift)  
    
    return E_SFR / EK


# In[378]:


def v_disp(Mvir,z): 
    #First step to introduce SMBH 
    #relationship between velocity dispersion and SMBH due to its potential
    #velocity dispersion of DM halo
    #input units of solar mass
    #output untis of km/s
    
    vmax_x = Vmax(Mvir,z)
    
    return vmax_x / 3**0.5


# In[379]:


def v_disp_assembly(Mvir0,z0,z):
    #Velocity dispersion of halo with mass halo growth
    #input units of solar mass
    #output units km/s
    
    Mvirz = halo_mass_assembly(Mvir0, z0, z)
    
    vdisp_z = v_disp(Mvirz, z)
    
    z = np.array(redshift)
    
    return vdisp_z


# In[380]:


def M_BH(Mvir0, z0, z):
    #Black Hole mass relationship from velocity disperion sigma
    #Woo, Jong-Hak (2013)
    #units of solar mass both input and output
    
    sigma = v_disp_assembly(Mvir0,z0,z)
    #velocity dispersion, units: km/s
    
    alpha = 8.37
    
    beta = 5.31
    
    logM_BH = alpha + beta * np.log10(sigma/200)
    #unitless
    
    M_BH = 10**logM_BH
    
    return M_BH


# In[381]:


def dM_BHdz(Mvir0, z0, redshift):
    #Black Hole mass growth from velocity dispersion sigma 
    #input units of solar mass
    #output units of solar mass per year
    
    z = np.array(redshift)
    
    zi = z
    
    zf = z - 0.01
    
    mi = M_BH(Mvir0, z0, zi)
    
    mf = M_BH(Mvir0, z0, zf)
    
    delm = mi - mf
    
    delz = zi - zf
    
    Ti = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zi)
    
    Tf = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zf)
    
    delT = Ti - Tf
    
    return delm / delT / 1E9
    
        


# In[382]:


def M_bh_from_MS(Mvir0, z0, z, fudge):
    #Blackhole mass aproximation from stellar mass (approx 100 times less than stellar mass)
    #Kormendy & Ho del 2013
    #units of solar mass both input and output
    

    Ms = galaxy_mass_assembly(Mvir0, z0, z)
    
    Mbh = Ms/fudge #fudge can be 1e2 or 1e3 
    
    return Mbh


# In[383]:


def dM_bh_from_MS_dt(Mvir0, z0, z, fudge):
    #Black hole mass growth rate from stellar mass approximation 
    #fudge can be 1e2 or 1e3
    #output units of solar mass per year
    
    z = np.array(redshift)
    
    zi = z
    
    zf = z - 0.01
    
    mi = M_bh_from_MS(Mvir0, z0, zi, fudge)
    
    mf = M_bh_from_MS(Mvir0, z0, zf, fudge)
    
    delm = mi - mf
    
    delz = zi - zf
    
    Ti = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zi)
    
    Tf = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zf)
    
    delT = Ti - Tf
    
    return delm / delT / 1E9


# In[384]:


####################I added these routines ##########################################
def fQ(Ms,z):
    #Fraction of quenched galaxies as a function of stellar mass
    #Equation 44 from Rodriguez-Puebla et al. 2017.
    #input: Solar Mass
    #output: uniteless

    ratio = np.log10(Ms) - (10.2 + 0.6 * z)
    
    ratio = 10**(-1.3 * ratio)
    
    ratio = 1. + ratio
    
    return 1. / ratio


# In[385]:


def fSF(Ms,z):
    #Fraction of star-forming galaxies as a function of stellar mass
    #FQ+FSF = 1.

    return 1. - fQ(Ms,z)


# In[386]:


def Mbh_Ms_SF(Ms,z):
    #Blackhole mass - stellar mass relation for star-forming galaxies
    #Reines & Volonteri 2015.
    #input: Solar Mass
    #output: Solar Mass
    
    Mbh = 7.45 + 1.05 * ( np.log10(Ms) - 11.)
    
    Mbh = 10**Mbh

    return Mbh
###why there's input for z if we're not using it 


# In[387]:


def Mbh_Ms_Q(Ms,z):
    #Blackhole mass - stellar mass relation for quenched galaxies
    #Reines & Volonteri 2015.
    #input: Solar Mass
    #output: Solar Mass

    Mbh = 8.95 + 1.40 *  ( np.log10(Ms) - 11.)

    Mbh = 10**Mbh
    
    return Mbh

#find equation number and look for z


# In[388]:


def M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0, z0, z):
    #Average Blackhole mass - stellar mass relation. Takes into account SFing and quenched galaxies.
    #We are assuming lognormal distributions for both SFing and quenched galaxies.
    #units of solar mass both input and output
    
    Ms = galaxy_mass_assembly(Mvir0, z0, z)

    Mbh = fQ(Ms,z) * np.log10(Mbh_Ms_Q(Ms,z))  + fSF(Ms,z) *  np.log10(Mbh_Ms_SF(Ms,z))
    
    Mbh = 10**Mbh

    return Mbh


# In[389]:


def dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0, z0, redshift):
    #Black hole mass growth rate from the average Blackhole mass - stellar mass relation
    #output units of solar mass per year
    
    z = np.array(redshift)
    
    zi = z
    
    zf = z - 0.01
    
    mi = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0, z0, zi)
    
    mf = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0, z0, zf)
    
    delm = mi - mf
    
    delz = zi - zf
    
    Ti = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zi)
    
    Tf = AgeUniverse(Om_mat_cero, Om_lambda_cero, h, zf)
    
    delT = Ti - Tf
    
    return delm / delT / 1E9


# In[390]:


def Lum_quasar_using_fraction_of_SF_and_quenched(Mvir0, z0, z):
    #Bolometric luminosoty of quasar given my SMBH accreting gas
    #Mbh/dt is in units are in solar mass per yr, and I need it g per sec
    
    
    dMdt = dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0, z0, z) * Msolar / a_year_in_seconds
    
    eps_acc = 0.1
    
    Lqso = ((eps_acc * speed_of_light**2) / (1- eps_acc)) * dMdt
    
    #return Luminosity in units of erg*s^-1
    #erg = g*cm^2/s^2
    
    return Lqso

###################################################################################


# In[391]:


def Lum_quasar(Mvir0, z0, z, fudge):
    #Bolometric luminosoty of quasar given my SMBH accreting gas
    #Mbh/dt is in units are in solar mass per yr, and I need it g per sec
    #fudge can be 1e2 or 1e3
    
    
    dMdt = dM_bh_from_MS_dt(Mvir0, z0, z, fudge) * Msolar / a_year_in_seconds
    
    eps_acc = 0.1
    
    Lqso = ((eps_acc * speed_of_light**2) / (1- eps_acc)) * dMdt
    
    #return Luminosity in units of erg*s^-1
    #erg = g*cm^2/s^2
    
    return Lqso   


# In[392]:


def Lum_eddigton(Mvir0, z0, z, fudge):
    
    #units of ergs
    
    g = const.G.value * (100**3) /1000 #m^3/kgs^2 to cm^3/gs^2
    
    mp = const.m_p.value * 1000 #kg to grams
    
    c = const.c.value *100 #m to cm
    
    solar = 1.99E33 #units of grams
    
    sig = const.sigma_T.value *100**2 #m^2 to cm^2
    
    num = 4*np.pi* g * mp * c
    
    den = sig
    
    Mbh = M_bh_from_MS(Mvir0, z0, z, fudge) #* solar #I commented this multiplication
    
    #check that num/den = 1.26E38
    #originally you code retunr: (num*Mbh) / den
    #but I simplified to 1.26E38 * Mbh
 
    return 1.26E38 * Mbh
   


# In[393]:


def Lum_quasar_sigma(Mvir0, z0, z):
    #Bolometric luminosoty of quasar given by SMBH accreting gas
    #Mbh/dt is in units are in solar mass per yr, and I need it g per sec
    #fudge can be 1e2 or 1e3
    #input units of solar mass
    #output units of ergs per second
    
    dMdt = dM_BHdz(Mvir0, z0, redshift)  * Msolar / a_year_in_seconds
    
    eps_acc = 0.1
    
    Lqso = ((eps_acc * speed_of_light**2) / (1- eps_acc)) * dMdt
    
    #return Luminosity in units of erg*s^-1
    #erg = g*cm^2/s^2
    
    return Lqso 


# In[394]:


def AGN_feedback(Mvir0, z0, z, fudge):
    #feedback due to AGN
    #Calculates energy from quasar and then it's divived by the kinetic 
    #of the halo
    #takes in units of ergs and it's divided...? are units right?
    
    eta = 0.008 #(croton+)
    
    KE = 0.5 * Vmax_assembly(Mvir0, z0, z)**2 #missing a mass, which one, and it should be cm s^-1
    
    E_qso = eta * Lum_quasar(Mvir0, z0, z, fudge)
    
    return E_qso / KE


# In[405]:


z0 = 0    
redshift = np.linspace (z0, 10, 100)
Mvir0 = (1e11, 1e12, 1e13, 1e14, 1e15)
Mstar = np.linspace(1e10, 1e12)
Mhalo = np.logspace(10, 15, 100)
Ms = SHMR(1, Mhalo)

plt.title('Star-to-halo relation')
plt.plot(np.log10(Mhalo), np.log10(Ms), '-k')
plt.xlabel('log Mhalo ($M_\odot$)')
plt.ylabel('log Mstar ($M_\odot$)')
plt.show()


# In[407]:


Mvirz = halo_mass_assembly(Mvir0[0], z0, redshift)
Mvirz2 = halo_mass_assembly(Mvir0[1], z0, redshift)
Mvirz3 = halo_mass_assembly(Mvir0[2], z0, redshift)
Mvirz4 = halo_mass_assembly(Mvir0[3], z0, redshift)
Mvirz5 = halo_mass_assembly(Mvir0[4], z0, redshift)

plt.title('Halo mass assembly of Rodriguez-Puebla et al. 2017')
plt.plot(np.log10(1+redshift), np.log10(Mvirz), color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Mvirz2), color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Mvirz3), color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Mvirz4), color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Mvirz5), color='red', label='mvir 1e15')

plt.xlabel('log 1+z')
plt.ylabel('log Mvir ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[398]:


Ms_z = galaxy_mass_assembly(Mvir0[0], z0, redshift)
Ms_z2 = galaxy_mass_assembly(Mvir0[1], z0, redshift)
Ms_z3 = galaxy_mass_assembly(Mvir0[2], z0, redshift)
Ms_z4 = galaxy_mass_assembly(Mvir0[3], z0, redshift)
Ms_z5 = galaxy_mass_assembly(Mvir0[4], z0, redshift)

plt.title('Galaxy mass assembly')
plt.plot(np.log10(1+redshift), np.log10(Ms_z), color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Ms_z2), color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Ms_z3), color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Ms_z4), color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Ms_z5), color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log Mstar ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[365]:


dmdz1 = dMsdz(Mvir0[0], z0, redshift)
dmdz2 = dMsdz(Mvir0[1], z0, redshift)
dmdz3 = dMsdz(Mvir0[2], z0, redshift)
dmdz4 = dMsdz(Mvir0[3], z0, redshift)
dmdz5 = dMsdz(Mvir0[4], z0, redshift)

plt.title('Star formation rate redshift')
plt.plot(np.log10(1+redshift), np.log10(dmdz1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(dmdz2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(dmdz3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(dmdz4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(dmdz5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log SFR [($M_\odot$)/yr]')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[185]:


plt.title('Star formation rate universe')
plt.plot(AgeUniverse(.3, .7, .7, redshift), np.log10(dmdz1), '-k', color='black', label='mvir 1e11')
plt.plot(AgeUniverse(.3, .7, .7, redshift), np.log10(dmdz2), '-k', color='blue', label='mvir 1e12')
plt.plot(AgeUniverse(.3, .7, .7, redshift), np.log10(dmdz3), '-k', color='green', label='mvir 1e13')
plt.plot(AgeUniverse(.3, .7, .7, redshift), np.log10(dmdz4), '-k', color='yellow', label='mvir 1e14')
plt.plot(AgeUniverse(.3, .7, .7, redshift), np.log10(dmdz5), '-k', color='red', label='mvir 1e15')
plt.xlabel('Age of universe [gyears]')
plt.ylabel('log SFR [($M_\odot$)/yr]')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[186]:


#SFHz = []
#SFHz1 = []
#SFHz2 = []
#SFHz3 = []
#SFHz4 = []
#for i in range(0, 100):
#    redshift = i
#    SFHz.append(SFR_hist(SHMR,Mvir0[0],z0,redshift))
#    SFHz1.append(SFR_hist(SHMR,Mvir0[1],z0,redshift))
#    SFHz2.append(SFR_hist(SHMR,Mvir0[2],z0,redshift))
#    SFHz3.append(SFR_hist(SHMR,Mvir0[3],z0,redshift))
#    SFHz4.append(SFR_hist(SHMR,Mvir0[4],z0,redshift))

#plt.title('Star Formation Histories Rodriguez-Puebla et al. 2017')
#plt.plot(np.log10(1.+z), np.log10(SFHz), color='k',ls='-', label='$M_{vir}(z=0)=10^{15}M_{\odot}$')
#plt.plot(np.log10(1.+z), np.log10(SFHz1), color='b',ls='-', label='$M_{vir}(z=0)=10^{14}M_{\odot}$')
#plt.plot(np.log10(1.+z), np.log10(SFHz2), color='g',ls='-', label='$M_{vir}(z=0)=10^{13}M_{\odot}$')
#plt.plot(np.log10(1.+z), np.log10(SFHz3), color='r',ls='-', label='$M_{vir}(z=0)=10^{12}M_{\odot}$')
#plt.plot(np.log10(1.+z), np.log10(SFHz4), color='y',ls='-', label='$M_{vir}(z=0)=10^{11}M_{\odot}$')
#plt.axis([0, 1.1, -3, 2.2])
#plt.ylabel('$log SFR$')
#plt.xlabel('$log ( 1 + z)$')
#plt.legend(loc='upper right',fontsize=10)
#plt.show()


# In[187]:


Vmax1 = Vmax_assembly(Mvir0[0],z0, redshift)
Vmax2 = Vmax_assembly(Mvir0[1],z0, redshift)
Vmax3 = Vmax_assembly(Mvir0[2],z0, redshift)
Vmax4 = Vmax_assembly(Mvir0[3],z0, redshift)
Vmax5 = Vmax_assembly(Mvir0[4],z0, redshift)

plt.title('Vmax growth')
plt.plot(np.log10(1+redshift), np.log10(Vmax1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Vmax2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Vmax3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Vmax4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Vmax5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log Vmax [km/s]')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[188]:


SNE_f1 = SNE_feedback(Mvir0[0], z0, redshift)
SNE_f2 = SNE_feedback(Mvir0[1], z0, redshift)
SNE_f3 = SNE_feedback(Mvir0[2], z0, redshift)
SNE_f4 = SNE_feedback(Mvir0[3], z0, redshift)
SNE_f5 = SNE_feedback(Mvir0[4], z0, redshift)

plt.title('SN feedback')
plt.plot(np.log10(1+redshift), np.log10(SNE_f1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(SNE_f2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(SNE_f3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(SNE_f4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(SNE_f5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log feedback')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[189]:


Vdisp1 = v_disp_assembly(Mvir0[0],z0, redshift)
Vdisp2 = v_disp_assembly(Mvir0[1],z0, redshift)
Vdisp3 = v_disp_assembly(Mvir0[2],z0, redshift)
Vdisp4 = v_disp_assembly(Mvir0[3],z0, redshift)
Vdisp5 = v_disp_assembly(Mvir0[4],z0, redshift)

plt.title('Vel disp')
plt.plot(np.log10(1+redshift), np.log10(Vdisp1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Vdisp2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Vdisp3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Vdisp4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Vdisp5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('vel disp')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[190]:


M_bh1 = M_BH(Mvir0[0],z0, redshift)
M_bh2 = M_BH(Mvir0[1],z0, redshift)
M_bh3 = M_BH(Mvir0[2],z0, redshift)
M_bh4 = M_BH(Mvir0[3],z0, redshift)
M_bh5 = M_BH(Mvir0[4],z0, redshift)

plt.title('BH mass from volevity disp')
plt.plot(np.log10(1+redshift), np.log10(M_bh1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('mass ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[191]:


dM_bhdz1 = dM_BHdz(Mvir0[0],z0, redshift)
dM_bhdz2 = dM_BHdz(Mvir0[1],z0, redshift)
dM_bhdz3 = dM_BHdz(Mvir0[2],z0, redshift)
dM_bhdz4 = dM_BHdz(Mvir0[3],z0, redshift)
dM_bhdz5 = dM_BHdz(Mvir0[4],z0, redshift)

plt.title('BH mass from velocity dipersion growth rate')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('Log dBH_mass/dt ($M_\odot yr^{-1}$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[192]:


fudge2 = 1e2

M_bh_ms1 = M_bh_from_MS(Mvir0[0],z0, redshift, fudge2)
M_bh_ms2 = M_bh_from_MS(Mvir0[1],z0, redshift, fudge2)
M_bh_ms3 = M_bh_from_MS(Mvir0[2],z0, redshift, fudge2)
M_bh_ms4 = M_bh_from_MS(Mvir0[3],z0, redshift, fudge2)
M_bh_ms5 = M_bh_from_MS(Mvir0[4],z0, redshift, fudge2)

plt.title('BH mass from stellar mass')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('mass ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[193]:


dM_bh_ms_dz1 = dM_bh_from_MS_dt(Mvir0[0],z0, redshift, fudge2)
dM_bh_ms_dz2 = dM_bh_from_MS_dt(Mvir0[1],z0, redshift, fudge2)
dM_bh_ms_dz3 = dM_bh_from_MS_dt(Mvir0[2],z0, redshift, fudge2)
dM_bh_ms_dz4 = dM_bh_from_MS_dt(Mvir0[3],z0, redshift, fudge2)
dM_bh_ms_dz5 = dM_bh_from_MS_dt(Mvir0[4],z0, redshift, fudge2)

plt.title('BH mass from stellar mass growth rate')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('Log dBH_mass/dt ($M_\odot yr^{-1}$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[194]:


fudge = 1e3

M_bh_mse31 = M_bh_from_MS(Mvir0[0],z0, redshift, fudge)
M_bh_mse32 = M_bh_from_MS(Mvir0[1],z0, redshift, fudge)
M_bh_mse33 = M_bh_from_MS(Mvir0[2],z0, redshift, fudge)
M_bh_mse34 = M_bh_from_MS(Mvir0[3],z0, redshift, fudge)
M_bh_mse35 = M_bh_from_MS(Mvir0[4],z0, redshift, fudge)

plt.title('BH mass from vel disp vs from stellar mass factor 100')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms1), '-', color='black', label='BHm stellar mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh_ms5), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(M_bh1), '--', color='black', label='BHm vel disp mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh2), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh3), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh4), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh5), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('mass ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[195]:


plt.title('BH mass from sigma vs from stellar mass factor 100: bh growth rate')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz1), '-', color='black', label='BHm stellar mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz5), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz1), '--', color='black', label='BHm vel disp mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz2), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz3), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz4), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(dM_bhdz5), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('Log dBH_mass/dt ($M_\odot yr^{-1}$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[196]:


plt.title('BH mass from velocity dispersion/from stellar mass, factor of 1000')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse31), '-', color='black', label='BHm stellar mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse32), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse33), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse34), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse35), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(M_bh1), '--', color='black', label='BHm vel disp mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh2), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh3), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh4), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh5), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('mass ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show() 


# In[197]:


plt.title('Fraction of quenched galaxies as a function of stellar mass')
plt.semilogx(Mstar, fQ(Mstar,z0) ,'-', color='black', label='z=0')
plt.semilogx(Mstar, fQ(Mstar,2) ,'-', color='blue', label='z=2')
plt.semilogx(Mstar, fQ(Mstar,5) ,'-', color='green', label='z=5')

plt.xlabel('Stellar Mass (Msun)')
plt.ylabel('Quenched Fraction')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[198]:


Lum_q1 = Lum_quasar(Mvir0[0], z0, redshift, fudge2)
Lum_q2 = Lum_quasar(Mvir0[1], z0, redshift, fudge2)
Lum_q3 = Lum_quasar(Mvir0[2], z0, redshift, fudge2)
Lum_q4 = Lum_quasar(Mvir0[3], z0, redshift, fudge2)
Lum_q5 = Lum_quasar(Mvir0[4], z0, redshift, fudge2)

Lum_q31 = Lum_quasar(Mvir0[0], z0, redshift, fudge)
Lum_q32 = Lum_quasar(Mvir0[1], z0, redshift, fudge)
Lum_q33 = Lum_quasar(Mvir0[2], z0, redshift, fudge)
Lum_q34 = Lum_quasar(Mvir0[3], z0, redshift, fudge)
Lum_q35 = Lum_quasar(Mvir0[4], z0, redshift, fudge)

plt.title('Quasar luminosity from stellar mass: factors 1e2 and 1e3')
plt.plot(np.log10(1+redshift), np.log10(Lum_q1), '-', color='black', label='1e2 factor mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_q2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_q3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_q4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_q5), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(Lum_q31), '--', color='black', label='1e3 factor mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_q32), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_q33), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_q34), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_q35), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log Luminosity (erg/sec)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[199]:


Lum_q_s1= Lum_quasar_sigma(Mvir0[0], z0, redshift)
Lum_q_s2= Lum_quasar_sigma(Mvir0[1], z0, redshift)
Lum_q_s3= Lum_quasar_sigma(Mvir0[2], z0, redshift)
Lum_q_s4= Lum_quasar_sigma(Mvir0[3], z0, redshift)
Lum_q_s5= Lum_quasar_sigma(Mvir0[4], z0, redshift)

plt.title('Quasar luminosity from velocity dispersion VS from stellar mass')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s1), '-', color='black', label='velocity dipersion mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s5), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(Lum_q31), '--', color='black', label='1e3 factor stellar mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_q32), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_q33), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_q34), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_q35), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log Luminosity (erg/sec)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[200]:


Lum_ed1 = Lum_eddigton(Mvir0[0], z0, redshift, fudge2)
Lum_ed2 = Lum_eddigton(Mvir0[1], z0, redshift, fudge2)
Lum_ed3 = Lum_eddigton(Mvir0[2], z0, redshift, fudge2)
Lum_ed4 = Lum_eddigton(Mvir0[3], z0, redshift, fudge2)
Lum_ed5 = Lum_eddigton(Mvir0[4], z0, redshift, fudge2)

Lum_ed31 = Lum_eddigton(Mvir0[0], z0, redshift, fudge)
Lum_ed32 = Lum_eddigton(Mvir0[1], z0, redshift, fudge)
Lum_ed33 = Lum_eddigton(Mvir0[2], z0, redshift, fudge)
Lum_ed34 = Lum_eddigton(Mvir0[3], z0, redshift, fudge)
Lum_ed35 = Lum_eddigton(Mvir0[4], z0, redshift, fudge)

plt.title('Eddington luminosity from stellar mass: factors of 1e2 and 1e3')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed1), '-', color='black', label='1e2 factor mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed5), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed31), '--', color='black', label='1e3 factor mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed32), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed33), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed34), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_ed35), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log Luminosity (erg/sec)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[201]:


agn_fb1 = AGN_feedback(Mvir0[0], z0, redshift, fudge)
agn_fb2 = AGN_feedback(Mvir0[1], z0, redshift, fudge)
agn_fb3 = AGN_feedback(Mvir0[2], z0, redshift, fudge)
agn_fb4 = AGN_feedback(Mvir0[3], z0, redshift, fudge)
agn_fb5 = AGN_feedback(Mvir0[4], z0, redshift, fudge)

plt.title('AGN feedback')
plt.plot(np.log10(1+redshift), np.log10(agn_fb1), '-', color='black', label='1e3 factor mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(agn_fb2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(agn_fb3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(agn_fb4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(agn_fb5), '-', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('feedback')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[202]:


M_bh_mse31 = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[0],z0, redshift,)
M_bh_mse32 = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[1],z0, redshift)
M_bh_mse33 = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[2],z0, redshift)
M_bh_mse34 = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[3],z0, redshift)
M_bh_mse35 = M_bh_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[4],z0, redshift)

plt.title('BH mass from vel disp vs from stellar mass using the fraction of SFing and Quenched')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse31), '-', color='black', label='BHm stellar mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse32), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse33), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse34), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh_mse35), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(M_bh1), '--', color='black', label='BHm vel disp mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(M_bh2), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(M_bh3), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(M_bh4), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(M_bh5), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('mass ($M_\odot$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[203]:


dM_bh_ms_dz1 = dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[0],z0, redshift)
dM_bh_ms_dz2 = dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[1],z0, redshift)
dM_bh_ms_dz3 = dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[2],z0, redshift)
dM_bh_ms_dz4 = dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[3],z0, redshift)
dM_bh_ms_dz5 = dM_bh_dt_from_Ms_using_fraction_of_SF_and_quenched(Mvir0[4],z0, redshift)

plt.title('BH mass from stellar mass growth rate')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz1), '-k', color='black', label='mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz2), '-k', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz3), '-k', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz4), '-k', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(dM_bh_ms_dz5), '-k', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('Log dBH_mass/dt ($M_\odot yr^{-1}$)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[204]:


Lum_q_s1= Lum_quasar_using_fraction_of_SF_and_quenched(Mvir0[0], z0, redshift)
Lum_q_s2= Lum_quasar_using_fraction_of_SF_and_quenched(Mvir0[1], z0, redshift)
Lum_q_s3= Lum_quasar_using_fraction_of_SF_and_quenched(Mvir0[2], z0, redshift)
Lum_q_s4= Lum_quasar_using_fraction_of_SF_and_quenched(Mvir0[3], z0, redshift)
Lum_q_s5= Lum_quasar_using_fraction_of_SF_and_quenched(Mvir0[4], z0, redshift)

plt.title('Quasar luminosity from velocity dispersion VS from stellar mass')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s1), '-', color='black', label='velocity dipersion mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s2), '-', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s3), '-', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s4), '-', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_q_s5), '-', color='red', label='mvir 1e15')
plt.plot(np.log10(1+redshift), np.log10(Lum_q31), '--', color='black', label='1e3 factor stellar mvir 1e11')
plt.plot(np.log10(1+redshift), np.log10(Lum_q32), '--', color='blue', label='mvir 1e12')
plt.plot(np.log10(1+redshift), np.log10(Lum_q33), '--', color='green', label='mvir 1e13')
plt.plot(np.log10(1+redshift), np.log10(Lum_q34), '--', color='yellow', label='mvir 1e14')
plt.plot(np.log10(1+redshift), np.log10(Lum_q35), '--', color='red', label='mvir 1e15')
plt.xlabel('log 1+z')
plt.ylabel('log Luminosity (erg/sec)')
plt.legend(loc=9, bbox_to_anchor=(0.1, -0.1), ncol=1)
plt.show()


# In[ ]:





# In[ ]:




