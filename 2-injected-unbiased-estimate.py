from brian2 import *
#get_ipython().magic('matplotlib inline')
from ccg import correlograms
#from phy.stats.ccg import correlograms

# Simulation parameters
Ntrial = 1000*5#*10
duration = 1000.*100    # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)

print("# trials:",Ntrial,"Trial duration: ",duration,"(ms)")

# Biophysical neuron parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm#/10           # membrane time constant
print("membrane time constant: ",tau/ms,"(ms)")
El = -70*mV               # resting potential
Vt = El+20*mV             # spike threshold
print("Threshold voltage: ",Vt/mV,"(mV)")
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period


# Biophysical background input parameters
tauI = 10*ms              # input time constant
print("background input time constant: ",tauI/ms,"(ms)")
sigmaI = 5.*mvolt       
muI = Vt-2.*mV
xmin = muI-2.5*mV
xmax = muI+2.5*mV
period = 10.#int(duration/10.)

print("background input time constant: ",tauI/ms,"(ms)","Input average amplitude: ",muI/mV,"(mV)","Input amplitude range:",.1*floor((xmax-xmin)/mV/.1),"(mV)","Input standard-deviation",sigmaI/mV,"(mV)","Interval duration: ",period,"(ms)")

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model with monosynapse
eqs_ref = Equations('''
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
reference.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
target.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
reference.I = 2*rand(Ntrial)-1
target.I = 2*rand(Ntrial)-1
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target)
#Mref = StateMonitor(reference,('V'),record=True) 
#Mtarg = StateMonitor(target,('V'),record=0) 

run(duration*ms)


# Basic firing parameters
print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))


# Organize the spike times into two long spike trains
train_ref0 = unique(Sref.i*duration+Sref.t/ms)
train_targ0 = unique(Starg.i*duration+Starg.t/ms)
train0 = append(train_ref0,train_targ0)
cell0 = int64(append(zeros(len(train_ref0)),ones(len(train_targ0))))


# Compute the CCG between the two neurons
lagmax = 100.                   #- in (ms)
bine = 1.                       #- in (ms)
ind_sort = np.argsort(train0)
st = train0[ind_sort]*.001
sc = cell0[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine


# Represent the ACGs and the CCG
FigACG = figure()
title('Auto-correlograms',fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Raw count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Craw[0,0],'.-k')
plot(lag,Craw[1,1],'.-b')
FigCCG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Raw count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Craw[0,1],'.-k')
#show()

FigACG.savefig('FigACG.png')
FigCCG.savefig('FigCCG.png')


# Measure the distribution of synchrony count before injection
synch_width = 1.
Tref0 = synch_width*floor(train_ref0/synch_width)
Ttarg0 = synch_width*floor(train_targ0/synch_width)
Tsynch0 = array(list(set(Tref0) & set(Ttarg0)))
synch_count0 = bincount(int64(floor(Tsynch0/duration)),minlength=Ntrial)


# Represent the distribution of synchrony count
# Inject spikes simultaneously in both trains
inject_count = int(max(synch_count0)/3.)
print("# injected spikes/trial: ",inject_count)
synch_width = 1.
Nwidth = int(duration/synch_width)
allwidths = arange(int(Ntrial*duration/synch_width))
include_index = int64(floor(train0/synch_width))
include_idx = list(set(include_index)) 
mask = zeros(allwidths.shape,dtype=bool)
mask[include_idx] = True
wheretoinject = synch_width*allwidths[~mask]
alreadythere = synch_width*allwidths[mask]
widths = append(wheretoinject,alreadythere)
tags = append(zeros(len(wheretoinject)),ones(len(alreadythere)))
ind_sort = np.argsort(widths)
widths = widths[ind_sort]
tags = tags[ind_sort]
widths = reshape(widths,(Ntrial,Nwidth))
tags = reshape(tags,(Ntrial,Nwidth))
ind_perm = transpose(np.random.permutation(np.mgrid[:Nwidth,:Ntrial][0])) 
widths = widths[arange(shape(widths)[0])[:,newaxis],ind_perm]
tags = tags[arange(shape(tags)[0])[:,newaxis],ind_perm]
ind_sort = argsort(tags,axis=1)
widths = widths[arange(shape(widths)[0])[:,newaxis],ind_sort]
tags = tags[arange(shape(tags)[0])[:,newaxis],ind_sort]
train_inject = ravel(widths[:,:inject_count])
train_ref = unique(append(train_ref0,train_inject))
train_targ = unique(append(train_targ0,train_inject)) 
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))

# Check whether the injection has been well performed
lagmax = 100.
bine = 1.
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
CrawI = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(CrawI[0,1]))-len(CrawI[0,1])/2)*bine
FigCCGi = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,CrawI[0,1]/(len(train_ref)*bine*.001),'.-k')
#show()

FigCCGi.savefig('Fig3B_CCGi.eps')

# Synchrony cout distribution after injection
synch_width = 1.
Tref = synch_width*floor(train_ref/synch_width)
Ttarg = synch_width*floor(train_targ/synch_width)
Tsynch = array(list(set(Tref) & set(Ttarg)))
synch_count = bincount(int64(floor(Tsynch/duration)),minlength=Ntrial)

FigS = figure()
xlabel('Synchrony count',fontsize=18)
ylabel('Normalized count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
bins = arange(min(synch_count0),max(synch_count0)+1,1)
count,base = histogram(synch_count0,bins=bins,density=1)
X = base[:-1]
plot(X,count,'o-b',markeredgecolor='b')
bins = arange(min(synch_count),max(synch_count)+1,1)
count,base = histogram(synch_count,bins=bins,density=1)
X = base[:-1]
plot(X,count,'o-k',markeredgecolor='k')
#show()

FigS.savefig('FigS.png')

# Predict the amount of injected synchrony
#Ninterval_range = linspace(1,2000,99,dtype=int)
#Delta_range = sort(append(period,duration/Ninterval_range))
Delta_range = sort(append(period,[5.,20.,25.,50.,100.,200.,250.,500.,1000.]))
Ntest = len(Delta_range)
injection_Delta = zeros(Ntest)
for k in range(Ntest):
    interval = Delta_range[k]
    Ninterval = int(duration/interval)
    #if interval == period:
    #    Ninterval = int(duration/interval)
    #else:    
    #    Ninterval = Ninterval_range[k]
    count_ref = bincount(int64(floor(train_ref/interval)),minlength=Ninterval*Ntrial)
    count_targ = bincount(int64(floor(train_targ/interval)),minlength=Ninterval*Ntrial)
    count_synch = bincount(int64(floor(Tsynch/interval)),minlength=Ninterval*Ntrial)
    RS_prod = sum(reshape(count_ref*count_synch,(Ntrial,Ninterval)),axis=1)
    RT_prod = sum(reshape(count_ref*count_targ,(Ntrial,Ninterval)),axis=1) 
    injection_Delta[k] = mean((interval*synch_count-RT_prod)/(interval*synch_count-RS_prod)*synch_count)
    if interval == period:
        injection_unbiased = (interval*synch_count-RT_prod)/(interval*synch_count-RS_prod)*synch_count
        injection_naive = synch_count-RT_prod/interval


# Represent the estimated injected synchrony distribution
FigTh = figure()
xlabel('Injected synchrony estimate',fontsize=18)
ylabel('Normalized count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
bins = arange(min(injection_naive),max(injection_naive)+1,1)
count,base = np.histogram(injection_naive,bins=bins,density=1)
X = base[:-1]
plot(X,count,'o-c',markeredgecolor='c')
plot(mean(injection_naive)*ones(2),[0,max(count)],'--c')
bins = arange(min(injection_unbiased),max(injection_unbiased)+1,1)
count,base = np.histogram(injection_unbiased,bins=bins,density=1)
X = base[:-1]
plot(X,count,'o-k',markeredgecolor='k')
plot(mean(injection_unbiased)*ones(2),[0,max(count)],'--k')
plot(inject_count*ones(2),[0,max(count)],'--r')
FigTh_Delta = figure()
xlabel('Interval length (ms)',fontsize=18)
ylabel('Injected synchrony count estimate',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(Delta_range,injection_Delta,'o-k',markeredgecolor='k')
plot([Delta_range[0],Delta_range[-1]],inject_count*ones(2),'--k')
show()

# Represent the estimated injected synchrony distribution
FigTh.savefig('Fig3B_unbiased-estimate.eps')
FigTh_Delta.savefig('Fig3B_Delta-influence-on-estimate.eps')