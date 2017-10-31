from brian2 import *
from scipy import stats
from ccg import correlograms


# Simulation parameters
Ntrial = 1000#*10*5
duration = 1000.#*100#*5#    # duration of the trial
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
xmin = muI-1*mV
xmax = muI+1*mV
period = 50.

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model 
#-----
eqs_ref = Equations('''
#-Potential
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt 
''')
eqs_targ = Equations('''
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt (linked)
''')
#-----Specify the model
reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
target.mu = linked_var(reference,'mu')
#-----Parameter initialization
reference.V = (Vt-.1*mV-Vr)*rand()+Vr
target.V = (Vt-.1*mV-Vr)*rand()+Vr
reference.I = 2*rand()-1
target.I = 2*rand()-1
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target) 
Mref = StateMonitor(reference,('V'),record=True) #('V','I','mu'),record=True) 
Mtarg = StateMonitor(target,('V'),record=True) #('V','I'),record=True) 
MrefI = StateMonitor(reference,('I'),record=0) 
MtargI = StateMonitor(target,('I'),record=0) 

run(duration*ms)

# Representing the basic recorded variables
#FigVm = figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Membrane')
#xlim([500,1000])
#plot(M.t/ms,Vt*ones(n)/mV,'--r')
#plot(M.t/ms,M.V[0]/mV,'k')
FigIref = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Input')
#xlim([500,1000])
plot(Mref.t/ms,MrefI.I[0],'k')
FigItarg = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Input')
#xlim([500,1000])
plot(Mtarg.t/ms,MtargI.I[0],'k')
#show()

FigIref.savefig('Fig2B-Iref-trace.eps')
FigItarg.savefig('Fig2B-Itarg-trace.eps')

# Define the target train
train_ref = sort(Sref.i*duration+Sref.t/ms)
train_targ = sort(Starg.i*duration+Starg.t/ms)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))

# Compute the CCG of the spike trains
lagmax = 150.
bine = 1.
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
FigCCG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Raw count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Craw[0,1]/(len(train_ref)*bine*.001),'.-k')
#show()

FigCCG.savefig('Fig2B-CCG.eps')

if Ntrial <= 1000:
    # Compute the STA of the postsynaptic Vm
    #--Function
    def STA(T,Vm,lag_max,duration):
        #--cut the part of the spike train that cannot be used for the STA    
        i = 0
        while T[i] < lag_max:
            i += 1
        i0 = i
        i = len(T)-1
        while T[i] > duration-lag_max:
            i -= 1
        i1 = i
        T = T[i0:i1]
        dt = 0.1
        nb_spike = int(len(T))
        nb_point = int(2*lag_max/dt)
        sample = np.zeros((nb_spike,nb_point))
        for i in range(nb_spike):
            istart = np.int64(round((T[i]-lag_max)/dt))
            istop = np.int64(round((T[i]+lag_max)/dt)) 
            sample[i,:] = Vm[istart:istop]
        ind = np.where(sample[:,0] != 0)[0]
        sample = sample[ind,:]
        average = np.mean(sample,0)
        deviation = np.std(sample,0) 

        return average,deviation

    #--Actual computation
    lag_max = 75
    period = Ntrial*duration
    Vtarg = reshape(Mtarg.V/mV,Ntrial*len(Mtarg.t))
    Vsta,error = STA(train_ref,Vtarg,lag_max,period)
    error = error/sqrt(len(train_ref))

    # Represent the computed spike-triggered averages
    FigSTA = figure()
    title('Spike-triggered average of Postsynaptic $V_m$',fontsize=18)
    xlabel('Time Lag (ms)',fontsize=18)
    ylabel('Potential (mV)',fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    xlim(-lag_max,lag_max)
    ylim(-52,-51)
    lag = linspace(-lag_max,lag_max,len(Vsta))
    fill_between(lag,Vsta-error,Vsta+error,facecolor='k',edgecolor='k',alpha=0.25)
    plot(lag,Vsta,'-k')
    plot(zeros(2),[min(Vsta),max(Vsta)],'--k',linewidth=2)
    #show()

    FigSTA.savefig('Fig2B-STA.eps')

    # Compute the CCF of Vm pre- and post-
    Vref = reshape(Mref.V/mV,Ntrial*len(Mref.t))
    #mu = reshape(Mref.mu/mV,Ntrial*len(Mref.t))
    #Iref = reshape(Mref.I/mV,Ntrial*len(Mref.t))
    #Itarg = reshape(Mtarg.I/mV,Ntrial*len(Mtarg.t))
    m_max = int(lag_max/time_step)
    n = len(Vtarg)
    Vref_avg = mean(Vref)
    Vtarg_avg = mean(Vtarg)
    #mu_avg = mean(mu)
    #Iref_avg = mean(Iref)
    #Itarg_avg = mean(Itarg)    
    CCF = zeros(2*m_max)
    #ACF_mu = zeros(2*m_max)
    #CCF_I = zeros(2*m_max)
    for k in range(2*m_max):
        m = k-m_max
        CCF[k] = sum((Vtarg[abs(m):]-Vtarg_avg)*(Vref[:n-abs(m)]-Vref_avg))/n
        #ACF_mu[k] = sum((mu[abs(m):]-mu_avg)*(mu[:n-abs(m)]-mu_avg))/n
        #CCF_I[k] = sum((Itarg[abs(m):]-Itarg_avg)*(Iref[:n-abs(m)]-Iref_avg))/n 
    lag = arange(-m_max,m_max)*time_step
    gamma_ref = sum((Vref-Vref_avg)*(Vref-Vref_avg))/n
    gamma_targ = sum((Vtarg-Vtarg_avg)*(Vtarg-Vtarg_avg))/n
    normalization = sqrt(gamma_ref*gamma_targ)
    CCF = CCF/normalization
    #ACF_mu = ACF_mu/ACF_mu[lag == 0]
    #gamma_ref = sum((Iref-Iref_avg)*(Iref-Iref_avg))/n
    #gamma_targ = sum((Itarg-Itarg_avg)*(Itarg-Itarg_avg))/n
    #normalization = sqrt(gamma_ref*gamma_targ)
    #CCF_I = CCF_I/normalization    
    #figure()
    #plot(lag,ACF_mu,'-b')
    #plot(lag,CCF_I,'-r')
    #plot(lag,CCF,'-k')
    FigCCF = figure()
    xlim(-lag_max,lag_max)
    plot(lag,CCF,'-k')
    show()

    FigCCF.savefig('Fig2B-CCF.eps')