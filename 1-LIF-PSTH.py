from brian2 import *
from scipy import stats#,optimize#special,signal,integrate#,interpolate

# Simulation parameters
Ntrial = 1000*500
duration = 500.#*100#*5#    # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)

print("# trials:",Ntrial,"Trial duration: ",duration,"(ms)")

# Biophysical neuron parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm#/10           # membrane time constant
#print("membrane time constant: ",tau/ms,"(ms)")
El = -70*mV               # resting potential
Vt = El+20*mV             # spike threshold
#print("Threshold voltage: ",Vt/mV,"(mV)")
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period


# Biophysical background input parameters
tauI = 10*ms
muI = Vt-1*mV
sigmaI = 1.*mvolt 

# Stimulus parameters
t0 = 450*ms
amplitude = .5*mV

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model 
#-----
eqs = Equations('''
#-Potential
dV/dt = (-V+muI+sigmaI*I+entree)/tau : volt 
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
entree : volt 
''')
#-----Specify the model
cell_inj = arange(Ntrial) 
spiketime_inj = t0*ones(Ntrial)
stimulus = SpikeGeneratorGroup(Ntrial,cell_inj,spiketime_inj)
neuron = NeuronGroup(Ntrial,model=eqs,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
#-----Parameter initialization
neuron.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
neuron.I = 2*rand()-1
neuron.entree = 0*volt
#-----Step input
step = Synapses(stimulus,neuron,
             '''
             w : volt
             ''',
             on_pre='''
             entree += w
             ''')
step.connect(i=arange(Ntrial),j=arange(Ntrial))
step.delay = 0*ms
step.w = amplitude
#-----Record variables
S = SpikeMonitor(neuron)
#M = StateMonitor(neuron,('V','I','mu'),record=0)  

run(duration*ms)

# Representing the basic recorded variables
#n = len(M.t)
#FigVm = figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Membrane')
#xlim([500,1000])
#plot(M.t/ms,Vt*ones(n)/mV,'--r')
#plot(M.t/ms,M.V[0]/mV,'k')
#show()

# Compute the PSTH
bin_size = 1.
spiketime = sort(S.t/ms)
spiketime = int64(floor(spiketime/bin_size))
count = bincount(spiketime)/(Ntrial*bin_size*.001)
nb = int64(round(100/bin_size))
count = count[-nb:]

# Represent the PSTH
FigPSTH = figure()
title('PSTH',fontsize=18)
xlabel('Time lag (ms)',fontsize=18)
ylabel('Rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-50,50)
lag = (arange(len(count))-len(count)/2)*bin_size
plot(lag,count,'.-k')
show()

FigPSTH.savefig('Fig2A-PSTH.eps')