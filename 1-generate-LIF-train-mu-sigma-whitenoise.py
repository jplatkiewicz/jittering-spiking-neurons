# coding: utf-8

# In[1]:

#import brian_no_units
from brian2 import *
#import numpy.matlib
#from scipy import stats,optimize#special,signal,integrate#,interpolate
#from scipy.special import erf,erfc
#%matplotlib inline
#from phy.stats.ccg import correlograms


# Simulation parameters

# In[2]:

Ntrial = 1000*5
duration = 1000.*10       # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)

print(duration)
print("# trials:",Ntrial,"Trial duration: ",duration,"(ms)")

# Biophysical neuron parameters

# In[3]:

#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm#/10.           # membrane time constant
#print("membrane time constant: ",tau/ms,"(ms)")
El = -70*mV               # resting potential
Vt = El+20*mV             # spike threshold
#print("Threshold voltage: ",Vt/mV,"(mV)")
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period


# Biophysical background input parameters
Nsample = 10#100
input_mu = (Vt/mV+.5-Vr/mV)*np.random.random_sample(Nsample)+Vr/mV
input_sigma = (10.-.1)*np.random.random_sample(Nsample)+.1
Ttarg = list([])
mu_test = array([])
sigma_test = array([])
for k in range(Nsample):
    muI = input_mu[k]*mvolt
    sigmaI = input_sigma[k]*mvolt 

    print("muI: ",muI/mV,"(mV); sigma_I: ",sigmaI/mV,"(mV)")
    mu_test = append(mu_test,muI/mV)
    sigma_test = append(sigma_test,sigmaI/mV)

    # Neuron model: Leaky Integrate-and-Fire

    #-----
    #-Integrate-and-fire neuron model 
    #-----
    eqs = Equations('''
        #-Potential
        dV/dt = (-V+muI+sigmaI*tau**.5*xi)/tau : volt
        ''')
    #-----Specify the model
    neuron = NeuronGroup(Ntrial,model=eqs,threshold='V>Vt',reset='V=Vr',refractory=refractory_period,method='euler')
    #-----Parameter initialization
    neuron.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
    #-----Record variables
    S = SpikeMonitor(neuron)
    #M = StateMonitor(neuron,('V','I'),record=True)  

    run(duration*ms)

    #Vm = reshape(M.V,Ntrial*len(M.t))
    #print(mean(Vm),std(Vm))
    #I = reshape(M.I,Ntrial*len(M.t))
    #print(mean(I),std(I))

    # Representing the basic recorded variables

    #n = len(M.t)
    #Fig1 = figure()
    #xlabel('Time (ms)')
    #ylabel('Potential (mV)')
    #title('Membrane')
    ##xlim([500,1000])
    #plot(M.t/ms,Vt*ones(n)/mV,'--r')
    #plot(M.t/ms,M.V[0]/mV,'k')
    #plot(M.t/ms,M.V[1]/mV,'b')
    #Fig2 = figure()
    #xlabel('Time (ms)')
    #ylabel('Cell index')
    #title('Spike raster')
    #xlim([500,900])
    #plot(S.t/ms,S.i,'ok')
    #show()

    #Fig1.savefig('Fig1.png',transparent=True)
    #Fig2.savefig('Fig2.png',transparent=True)


    # Basic firing parameters
    print("# spikes/trial: ",mean(S.count),std(S.count))
    print('Average firing rate: ',sum(S.count)/(Ntrial*duration*.001))

    # Define the target train
    train_targ = sort(S.i*duration+S.t/ms)
    Ttarg.append(train_targ)

    # Check whether it is Poisson
    ISI = diff(train_targ)
    print("Coefficient of variation of the ISI distribution: ",std(ISI)/mean(ISI))

save('muI.npy',mu_test)
save('sigmaI.npy',sigma_test)
save('Ttarg.npy',Ttarg)    