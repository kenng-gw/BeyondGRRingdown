import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from generate_modes import *
import qnm

chi_f, M_f = 0.692, 0.9525
omegas, taus=[],[]
for n in range(8):
	qnm_mode = qnm.modes_cache(s=-2,l=2,m=2,n=n)
	omega_complex, _, _ = qnm_mode(a=chi_f)
	omega = np.real(omega_complex)/M_f
	tau=np.abs(1./np.imag(omega_complex))*M_f
	omegas.append(omega)
	taus.append(tau)

def damped_sine(t,tau, omega, phi, A):
	return A*np.exp(-t/tau)*np.cos(t*omega+phi)

def damped_sine_complex(t, phi, A, tau=taus[0], omega=omegas[0]):	
	t_real = t[:int(t.size/2)]
	t_imag = t[int(t.size/2):]
	real_out = A*np.exp(-t_real/tau)*np.cos(t_real*omega+phi)
	imag_out = -A*np.exp(-t_imag/tau)*np.sin(t_imag*omega+phi)
	return np.hstack([real_out, imag_out])

def damped_sine_complex_general(t, phi_c, phi_s, A_c, A_s, tau_c, tau_s, omega_c, omega_s):	
	t_real = t[:int(t.size/2)]
	t_imag = t[int(t.size/2):]
	real_out = A_c*np.exp(-t_real/tau_c)*np.cos(t_real/omega_c+phi_c)
	imag_out = -A_s*np.exp(-t_imag/tau_s)*np.sin(t_imag/omega_s+phi_s)
	return np.hstack([real_out, imag_out])

def damped_sine_complex_overtone(t, phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, A0, A1, A2, A3, A4, A5, A6, A7):
	phis = [phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7]
	As = [A0, A1, A2, A3, A4, A5, A6, A7]
	ns = np.arange(1,len(As)+1,1)
	t_real = t[:int(t.size/2)]
	t_imag = t[int(t.size/2):]
	real_out=0.0
	imag_out=0.0
	for (n, tau, omega, phi, A) in zip(ns, taus, omegas, phis, As):
		real_out += A*np.exp(-t_real/tau)*np.cos(t_real*omega+phi)
		imag_out += -A*np.exp(-t_imag/tau)*np.sin(t_imag*omega+phi)
	return np.hstack([real_out, imag_out])

ell=0.0
t_start = 0.0
strain_out=0.+0.j
time, strain_22 = dCS_hlm(ell,(2,2))
time, strain_2m2 = dCS_hlm(ell,(2,-2))
strain_22*=1#SpinWeightedSphericalHarmonic(0.0, 0, -2, mode[0], mode[1])

t_RD_22, h_RD_22 = get_ringdown(time, strain_22)
t_RD = t_RD_22[t_RD_22>=t_start]
h_RD = h_RD_22[t_RD_22>=t_start]
h_RD_vec = np.hstack([np.real(h_RD), np.imag(h_RD)])
t_RD_2m2, h_RD_2m2 = get_ringdown(time, strain_2m2)

#p0=[0.1,0.1]
#popt,pcov = curve_fit(damped_sine_complex, t_RD, h_RD, bounds = ([-np.pi,1.e-30],[np.pi,1]),p0=p0)
#print(omegas[:3], taus[:3])
#omegas[0]+=-0.437*ell**4/M_f
#omegas[1]+=3.92*ell**4/M_f
#omegas[2]+=-1.54*ell**4/M_f
#taus[0]+=-8.13*ell**4*M_f
#taus[1]+=220.1*ell**4*M_f
#taus[2]+=-146.9*ell**4*M_f
#print(omegas[:3], taus[:3])
n=8
p0_phis=[-0.5]*n
p0_As=[n+0.01]*n
p0=np.array([p0_phis,p0_As]).flatten()
plow=np.array([[-np.pi]*n,[1.e-3]*n]).flatten()
pup=np.array([[np.pi]*n,[100]*n]).flatten()
popt,pcov = curve_fit(damped_sine_complex_overtone, np.hstack([t_RD,t_RD]), h_RD_vec,p0=p0, bounds = (plow,pup))
#p0=[0.1,0.1,2,2]
#popt,pcov = curve_fit(damped_sine_complex, np.hstack([t_RD,t_RD]), h_RD_vec, bounds = ([-np.pi,1.e-30,1.e-9,1.e-9],[np.pi,1,100,10]),p0=p0)
#p0=[0.1,0.1,0.01,0.01,15.,15.,2.,2.]
#plow=[-np.pi,-np.pi,1.e-30,1.e-30,1.e-9,1.e-9,1.e-9,1.e-9]
#pup=[np.pi,np.pi,1,1,100,100,100,100]
#popt,pcov = curve_fit(damped_sine_complex_general, np.hstack([t_RD,t_RD]), h_RD_vec, bounds = (plow,pup),p0=p0)

print(popt)
h_RD_fit_vec = damped_sine_complex_overtone(np.hstack([t_RD,t_RD]), *popt)
#h_RD_fit_vec = damped_sine_complex_general(np.hstack([t_RD,t_RD]), *popt)
#h_RD_fit_vec = damped_sine_complex(np.hstack([t_RD,t_RD]), *popt)

h_RD_fit_real = h_RD_fit_vec[:t_RD.size]
h_RD_fit_imag = h_RD_fit_vec[t_RD.size:]
def mismatch(htrue,hfit,t):
	htrue_hfit = np.trapz(np.real(htrue*np.conj(hfit)), x=t)
	htrue_htrue = np.trapz(np.abs(htrue)**2, x=t)
	hfit_hfit = np.trapz(np.abs(hfit)**2, x=t)
	return 1.-htrue_hfit/np.sqrt(htrue_htrue*hfit_hfit)

h_RD_fit = h_RD_fit_real+h_RD_fit_imag*1j
mm = mismatch(h_RD, h_RD_fit, t_RD)
print(mm)

mm_chiral=mismatch(h_RD_22, np.conj(h_RD_2m2), t_RD_22)
print(mm_chiral)

plt.figure()
plt.plot(t_RD, np.real(h_RD))
plt.plot(t_RD, np.imag(h_RD))
#plt.plot(t_RD, h_RD_fit, ls='--')
plt.plot(t_RD, h_RD_fit_real, ls='--')
plt.plot(t_RD, h_RD_fit_imag, ls='--')
plt.savefig("test.pdf")
plt.close()

#plt.figure()
#plt.plot(t_RD, mm)
#plt.savefig("test_error.pdf")
#plt.close()
