from brian2 import *
from scipy import stats,signal,optimize
from pandas import Series
from ccg import correlograms


# Simulation parameters
Ntrial = 1000#*10#*5
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
eqs = Equations('''
#-Potential
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt 
''')
#-----Specify the model
neuron = NeuronGroup(Ntrial,model=eqs,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
neuron.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
#-----Parameter initialization
neuron.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
neuron.I = 2*rand()-1
#-----Record variables
S = SpikeMonitor(neuron)
M = StateMonitor(neuron,('V','I','mu'),record=True)  

run(duration*ms)

# Representing the basic recorded variables
n = len(M.t)
FigVm = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Membrane')
#xlim([500,1000])
plot(M.t/ms,Vt*ones(n)/mV,'--r')
plot(M.t/ms,M.V[0]/mV,'k')
FigI = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Input')
#xlim([500,1000])
plot(M.t/ms,M.I[0],'k')
#show()

FigVm.savefig('Fig2A-Vm-trace.eps')
FigI.savefig('Fig2A-I-trace.eps')

# Basic firing parameters
print("# spikes/trial: ",mean(S.count),std(S.count))
print('Average firing rate: ',sum(S.count)/(Ntrial*duration*.001))

# Define the target train
train = sort(S.i*duration+S.t/ms)

# Check whether it is Poisson
ISI = diff(train)
print("Coefficient of variation of the ISI distribution: ",std(ISI)/mean(ISI))

# Compute the ACG of the spike train
lagmax = 150.
bine = 1.
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
cell = int64(zeros(len(train)))
sc = cell[ind_sort]
Araw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Araw[0,0]))-len(Araw[0,0])/2)*bine
FigACG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Auto-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Araw[0,0]/(len(st)*bine*.001),'.-k')
#show()

FigACG.savefig('Fig2A-ACG.eps')

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
    Vm = reshape(M.V/mV,Ntrial*len(M.t))
    Vsta,error = STA(train,Vm,lag_max,period)
    #error = error#/sqrt(len(train))

    # Represent the computed spike-triggered averages
    FigSTA = figure()
    title('Spike-triggered average of Postsynaptic $V_m$',fontsize=18)
    xlabel('Time Lag (ms)',fontsize=18)
    ylabel('Potential (mV)',fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    xlim(-lag_max,lag_max)
    #ylim(-50.5,-49.5)
    lag = linspace(-lag_max,lag_max,len(Vsta))
    fill_between(lag,Vsta-error,Vsta+error,facecolor='k',edgecolor='k',alpha=0.25)
    plot(lag,Vsta,'-k')
    plot(zeros(2),[min(Vsta),max(Vsta)],'--k',linewidth=2)
    #show()

    FigSTA.savefig('Fig2A-STA.eps')

    # Compute the ACF of Vm
    m_max = int(lag_max/time_step)
    n = len(Vm)
    Vm_avg = mean(Vm)
    ACF = zeros(2*m_max)
    for k in range(2*m_max):
        m = k-m_max
        ACF[k] = sum((Vm[abs(m):]-Vm_avg)*(Vm[:n-abs(m)]-Vm_avg))/n
        #sum((Vm[:n-m]-Vm_avg)*(Vm[m:]-Vm_avg))/sum((Vm-Vm_avg)**2)+2.*m/n    
    lag = arange(-m_max,m_max)*time_step
    ACF = ACF/ACF[lag == 0] 
    lag_pos = arange(0,m_max)*time_step
    X = ACF[lag >= 0]
    dX = zeros(len(X))
    dX[1:-1] = (X[2:]-X[:-2])/(2*time_step)
    f = lambda params: params[0]**2*exp(-lag_pos/params[1]**2)-X
    p0 = X[0]-X[-1] 
    p1 = -p0/dX[1]  
    p,_=optimize.leastsq(f,array([p0,p1]))#,p2]))
    A = p[0]**2
    tau = p[1]**2
    print(A,tau)
    FigACF = figure()
    xlim(-lag_max,lag_max)
    plot(lag,ACF,'-k')
    lag_plus = lag_pos[lag_pos <= tau]
    plot(lag_plus,A*(1-lag_plus/tau),'--r')
    plot(lag_pos,A*exp(-lag_pos/tau),'r')
    show()

    FigACF.savefig('Fig2A-ACF.eps')