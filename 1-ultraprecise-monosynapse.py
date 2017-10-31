from brian2 import *
#from scipy import special,stats,optimize,interpolate#signal,signal,integrate
import numpy as np
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#from tools import STA#,STA_bin


# Simulation parameters
Ntrial = 1000*10
duration = 500.         # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)          

# Biophysical parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm               # membrane time constant
#print("membrane time constant: ",tau/ms,"(ms)")
tauI = 10*ms              # input time constant
print("background input time constant: ",tauI/ms,"(ms)")
El = -70*mV               # resting potential
Vt = El+20*mV             # spike threshold
#print("Threshold voltage: ",Vt/mV,"(mV)")
Vr = El+10*mV             # reset value
refractory_period = 0*ms # refractory period

# Background synaptic input parameters
muI = Vt-1*mV 
sigmaI = 1.*mvolt

# Monosynaptic parameters
tauS = 3*ms                # synaptic conductance time constant
Esyn = 0*mV# muI-5*mV             # synapse reversal potential (mu+20*mV)
print("Synaptic reversal potential: ",Esyn/mV,"(mV)")
PSC = 50*pA               # post-synaptic current ammplitude: (high background noise: 100 pA | low noise regime: 15 pA)
g0 = PSC/(Esyn-muI)
print("synaptic weight: ",g0/nsiemens,"(siemens)")
latency = 1.5*ms

# Stimulus parameters
t0 = 450*ms

# Specify the neuron model: leaky integrate-and-fire
#-----
#-Integrate-and-fire neuron model with monosynapse
eqs = Equations('''
#-Potential
dV/dt = (-V+muI+sigmaI*I-g0/gm*gsyn*(V-Esyn))/tau : volt
#-Individual Backgound Input
I : 1 (linked)
#-Monosynaptic input
dgsyn/dt = -gsyn/tauS : 1
''')
eqs_input = Equations(''' 
dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
''')
#-----Model setting
cell_inj = arange(Ntrial) 
spiketime_inj = t0*ones(Ntrial)
stimulus = SpikeGeneratorGroup(Ntrial,cell_inj,spiketime_inj)
target = NeuronGroup(Ntrial,model=eqs,threshold='V>Vt',reset='V=Vr',refractory=refractory_period,method='euler')
target_no = NeuronGroup(Ntrial,model=eqs,threshold='V>Vt+1000*mV',reset='V=Vr',refractory=refractory_period,method='euler')
background = NeuronGroup(Ntrial,model=eqs_input,threshold='x>1000',reset='x=0',refractory=refractory_period,method='euler')
target.I = linked_var(background,'x')
target_no.I = linked_var(background,'x')
#----Parameters initialization
target.V = muI
target.gsyn = 0
target_no.V = target.V
target_no.gsyn = 0
background.x = 2*rand(Ntrial)-1
#--Synaptic connection
synaptic = Synapses(stimulus,target,
             on_pre='''
             gsyn += 1
             ''')
synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic.delay = latency
synaptic_no = Synapses(stimulus,target_no,
             on_pre='''
             gsyn += 1
             ''')
synaptic_no.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic_no.delay = latency
#-----
#-Record variables
Starg = SpikeMonitor(target)
MtargCond = StateMonitor(target_no,('gsyn'),record=0) 
Mtarg = StateMonitor(target_no,('V'),record=True) 

run(duration*ms)


# Representing the neuron firing behavior
#figure()
#title('Target (postsynaptic) spike trains')
#xlim([400,410])
#plot(Starg.t/ms,Starg.i,'s',markersize=4,color='k')
#show()


# Representing the neuron membrane activity
#n = len(Mtarg.t)
#trace = figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Target cell')
#xlim([350,450])
#ylim(Vr/mV-1,Vt/mV+1)
#plot(Mtarg.t/ms,Vt*np.ones(n)/mV,'--r')
#plot(Mtarg.t/ms,Mtarg.V[0]/mV,'k')
#figure()
#xlabel('Time (ms)',fontsize=18)
#ylabel('Potential (mV)',fontsize=18)
#title('Stimulus Input',fontsize=18)
#xlim(400,500)
#ylim(-1,6)
#plot(Mtarg.t/ms,-(g0/gm*MtargCond.gsyn[0]*(Mtarg.V[0]-Esyn))/mV,'k')
##show()
#trace.savefig('trace-Vm_Ostojic-peak.png',transparent=True)

# Compute the spike-triggered average of Vm
bine = time_step
PSP = mean(Mtarg.V/mV,0)
nb = int64(round(100/bine))
PSP = PSP[-nb:]


# Compute the PSTH
bin_size = 1.
spiketime = sort(Starg.t/ms)
spiketime = int64(floor(spiketime/bin_size))
count = bincount(spiketime,minlength=int(duration/bin_size))/(Ntrial*bin_size*.001)
nb = int64(round(100/bin_size))
PSTH = count[-nb:]#/Ntrial


# Represent the PSTH
bin_size = 1.
nb = int64(round(100/bin_size))
PSTHgraph = figure()
title('PSTH - Use Noisy Trace',fontsize=18)
xlabel('Time lag (ms)',fontsize=18)
ylabel('Normalized Count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-5,20)
xlim(-50,50)
#ylim(0,.03)
lag = (arange(len(PSTH))-len(PSTH)/2)*bin_size
plot(lag,PSTH,'.-k')
STAgraph = figure()
title('PSP - Use Noisy Trace',fontsize=18)
xlabel('Time lag (ms)',fontsize=18)
ylabel('Potential (mV)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-5,20)
xlim(-50,50)
ylim(-51.,-50.5)
nb_point = len(PSP)
lag_max = 50
lag_Vm = linspace(-lag_max,lag_max,nb_point)
dPSP = zeros(len(PSP))
x = lag_Vm
dx = x[1]-x[0]
dPSP[1:-1] = (PSP[2:]-PSP[:-2])/(2*dx)
#--Smoothed PSP derivative
time_step = dx
time = arange(0,100,time_step)
#(m-sqrt(2*m),m+sqrt(2*m)) where m is the number of datapoints in x, y
m = len(PSP)
smooth = (99./100*(m-sqrt(2*m))+1./100*(m+sqrt(2*m)))
print("smoothness",smooth)
#tck = interpolate.splrep(time,PSP,s=.05)
#PSP_smooth = interpolate.splev(time,tck)    
#dPSP = interpolate.splev(time,tck,der=1)
#
nb_Vm = int64(round(100/time_step))
PSP0 =  mean(PSP[:int64(nb_Vm/2.5)])
dPSP0 =  mean(dPSP[:int64(nb_Vm/2.5)])
PSTH0 = mean(PSTH[:int64(nb_Vm/3.)])
print("EPSP amplitude: ",max(PSP)-PSP0)
plot(lag_Vm,PSP+.01,'k')
#plot(lag_Vm,(PSP-PSP0)/(max(PSP)-PSP0)*(max(PSTH)-PSTH0)+PSTH0,'k',linewidth=2)
dPSPm = max(dPSP)
#plot(lag_Vm,(dPSP-dPSP0)/(dPSPm-dPSP0)*(max(PSTH)-PSTH0)+PSTH0,'--k',linewidth=2)
#plot(lag,PSTH,'.-k')
i0 = min(where(PSTH > max(PSTH)/2)[0])
i1 = max(where(PSTH > max(PSTH)/2)[0])
precision = lag[i1]-lag[i0]
print("PSTH peak width: ",precision,"(ms)")
show()

PSTHgraph.savefig('FigDA_PSTH.eps')#,transparent=True)
STAgraph.savefig('FigDA_STA.eps')