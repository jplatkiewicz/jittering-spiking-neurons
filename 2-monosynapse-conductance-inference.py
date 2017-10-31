from brian2 import *
from scipy import stats
#get_ipython().magic('matplotlib inline')
from ccg import correlograms
#from phy.stats.ccg import correlograms


# Simulation parameters
Ntrial = 1000*10
duration = 1000.*10#*3#0    # duration of the trial
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
sigmaI = 1.*mvolt       
muI = Vt-.5*mV
xmin = muI-.5*mV
xmax = muI+.5*mV
period = 50.#int(duration/10.)
print("background input time constant: ",tauI/ms,"(ms)","Input average amplitude: ",muI/mV,"(mV)","Input amplitude range:",.1*floor((xmax-xmin)/mV/.1),"(mV)","Input standard-deviation",sigmaI/mV,"(mV)","Interval duration: ",period,"(ms)")


# Monosynapse parameters
tauS = 3*ms                # synaptic conductance time constant
Esyn = 0*mV# muI-5*mV             # synapse reversal potential (mu+20*mV)
print("Synaptic reversal potential: ",Esyn/mV,"(mV)")
PSC = 25*pA     #25*pA          # post-synaptic current ammplitude: (high background noise: 100 pA | low noise regime: 15 pA)
g0 = PSC/(Esyn-muI)
print("synaptic weight: ",g0/nsiemens,"(siemens)")
latency = 1.5*ms
Nphase = 10.
phase = duration/Nphase
wmin = .5
wmax = 4.


# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model with monosynapse
eqs_ref = Equations('''
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
I : 1 (linked)
mu : volt
''')
eqs_ref0 = Equations('''
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
I : 1 (linked)
mu : volt (linked)
''')
eqs_refnoise = Equations('''
dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
''') 
eqs_targ = Equations('''
dV/dt = (-V+mu+sigmaI*I-g0/gm*gsyn*(V-Esyn))/tau : volt 
I : 1 (linked)
mu : volt (linked)
#-Monosynaptic input
dgsyn/dt = -gsyn/tauS : 1
''')
eqs_targnoise = Equations('''
dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
''')
#----
#--
#-----Specify the synapse-on model 
#--
#----
reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
target.mu = linked_var(reference,'mu')
ref_noise = NeuronGroup(Ntrial,model=eqs_refnoise,threshold='x>10**6',reset='x=0',method='euler')
targ_noise = NeuronGroup(Ntrial,model=eqs_targnoise,threshold='x>10**6',reset='x=0',method='euler')
reference.I = linked_var(ref_noise,'x')
target.I = linked_var(targ_noise,'x')
#-----Parameter initialization
reference.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
target.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
target.gsyn = 0
ref_noise.x = 2*rand(Ntrial)-1
targ_noise.x = 2*rand(Ntrial)-1
#--Synaptic connection
weight_value = np.random.permutation(linspace(wmin,wmax,Nphase))#wmin+(wmax-wmin)*np.random.rand(Nphase)
weight = TimedArray(weight_value,dt=phase*ms)
synaptic = Synapses(reference,target,
             '''w = weight(t) : 1''',
             on_pre='''
             gsyn += w
             ''')
synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
#synaptic.w = 1#linspace(.5,2,Ntrial)#
synaptic.delay = latency
#synaptic.run_regularly('''w = wmin+(wmax-wmin)*rand()''',dt=duration*ms/5.)
#----
#--
#----Specify the synpase-off model
#--
#----
reference0 = NeuronGroup(Ntrial,model=eqs_ref0,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
target0 = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt+0*mV',reset='V=Vr',refractory=refractory_period,method='euler')
reference0.mu = linked_var(reference,'mu')
target0.mu = linked_var(reference,'mu')
reference0.I = linked_var(ref_noise,'x')
target0.I = linked_var(targ_noise,'x')
#-----Parameter initialization
reference0.V = reference.V
target0.V = target.V
target0.gsyn = 0
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target)
#Mref = StateMonitor(reference,('V'),record=True) 
#Mtarg = StateMonitor(target,('V','gsyn'),record=True) 
Msyn = StateMonitor(synaptic,'w',record=0)
Sref0 = SpikeMonitor(reference0)
Starg0 = SpikeMonitor(target0)

run(duration*ms)

# Representing the basic recorded variables

# In[ ]:

#figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Reference cell')
#xlim([540,600])
#plot(Mtarg.t/ms,Vt*ones(len(Mtarg.t))/mV,'--r')
#plot(Mref.t/ms,Mref.V[0]/mV,'k')
#plot(Mref.t/ms,Mref.V[1]/mV,'b')
#plot(Mtarg.t/ms,Mtarg.V[0]/mV,'-k')
#plot(Mtarg.t/ms,Mtarg.V[1]/mV,'-b')
#figure()
#xlabel('Time (ms)')
#ylabel('Synaptic Conductance')
#title('Target cell')
#xlim([400,450])
#plot(Mtarg.t/ms,Mtarg.gsyn[0],'k')
#plot(Mtarg.t/ms,Mtarg.gsyn[1],'b')
FigW = figure()
xlabel('Time (ms)')
ylabel('Synaptic weight')
title('Target cell')
#xlim([400,450])
#plot(weight(t),'k')
plot(Msyn.t/ms,Msyn.w[0],'k')
#plot(Msyn.t/ms,Msyn.w[1],'--b')
#inputgraph = figure()
#title('Spike Raster')
#xlabel('Time (ms)')
#ylabel('Cell index')
#xlim(500,1000)
#plot(Sref.t/ms,Sref.i,'ok')
#plot(Starg.t/ms,Starg.i,'sc')
#show()

FigW.savefig('FigW.png')

# Basic firing parameters

# In[ ]:

print("WITH SYNAPSE")
print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))
print("WITHOUT SYNAPSE")
print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))


# Organize the spike times into two long spike trains

# In[ ]:

#--WITH SYNAPSE--
train_ref = sort(Sref.t/ms+floor(Sref.t/(ms*phase))*(-1+Ntrial)*phase+Sref.i*phase)
train_targ = sort(Starg.t/ms+floor(Starg.t/(ms*phase))*(-1+Ntrial)*phase+Starg.i*phase)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))
#--WITHOUT SYNAPSE--
train_ref0 = sort(Sref0.t/ms+floor(Sref0.t/(ms*phase))*(-1+Ntrial)*phase+Sref0.i*phase)
train_targ0 = sort(Starg0.t/ms+floor(Starg0.t/(ms*phase))*(-1+Ntrial)*phase+Starg0.i*phase)
train0 = append(train_ref0,train_targ0)
cell0 = int64(append(zeros(len(train_ref0)),ones(len(train_targ0))))


# Compute the CCG between the two neurons
lagmax = 100.                   #- in (ms)
bine = 1.                       #- in (ms)
#--WITH SYNAPSE--
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
#--WITHOUT SYNAPSE--
ind_sort = np.argsort(train0)
st = train0[ind_sort]*.001
sc = cell0[ind_sort]
Craw0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)


# Check whether the segmentation of the trains was correctly done
#--WITH SYNAPSE--
#ind_sort = np.argsort(train)
#st = train[ind_sort]
#sc = cell[ind_sort]
#c = bincount(int64(floor(st/(Ntrial*phase))),minlength=Nphase)
#c = append(0,c)
#c = cumsum(c)
#
#cref = bincount(int64(floor(train_ref/(Ntrial*phase))),minlength=Nphase)
#cref = append(0,cref)
#cref = cumsum(cref)
#
#ctarg = bincount(int64(floor(train_targ/(Ntrial*phase))),minlength=Nphase)
#ctarg = append(0,ctarg)
#ctarg = cumsum(ctarg)
#--WITHOUT SYNAPSE--
#ind_sort = np.argsort(train0)
#st0 = train0[ind_sort]
#sc0 = cell0[ind_sort]
#c0 = bincount(int64(floor(st0/(Ntrial*phase))),minlength=Nphase)
#c0 = append(0,c0)
#c0 = cumsum(c0)
#
#cref0 = bincount(int64(floor(train_ref0/(Ntrial*phase))),minlength=Nphase)
#cref0 = append(0,cref0)
#cref0 = cumsum(cref0)
#
#ctarg0 = bincount(int64(floor(train_targ0/(Ntrial*phase))),minlength=Nphase)
#ctarg0 = append(0,ctarg0)
#ctarg0 = cumsum(ctarg0)
#
#synch_width = 5.
#for k in range(int(Nphase)):
    #--WITH SYNAPSE--
#    stk = st[c[k]:c[k+1]]*.001
#    sck = sc[c[k]:c[k+1]]
#    Ccg = correlograms(stk,sck,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
#    Tref = synch_width*floor(train_ref[cref[k]:cref[k+1]]/synch_width)
#    Ttarg = synch_width*floor(train_targ[ctarg[k]:ctarg[k+1]]/synch_width)
#    Tsynch = array(list(set(Tref) & set(Ttarg)))
#    synch = len(Tsynch)
#    print(synch) 
    #--WITHOUT SYNAPSE--
#    stk = st0[c0[k]:c0[k+1]]*.001
#    sck = sc0[c0[k]:c0[k+1]]
#    Ccg0 = correlograms(stk,sck,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.) 
#    Tref0 = synch_width*floor(train_ref0[cref0[k]:cref0[k+1]]/synch_width)
#    Ttarg0 = synch_width*floor(train_targ0[ctarg0[k]:ctarg0[k+1]]/synch_width)
#    Tsynch0 = array(list(set(Tref0) & set(Ttarg0)))
#    synch0 = len(Tsynch0)
#    print(synch0)  
#    figure()
#    plot(lag,Ccg[0,1],'k')
#    plot(lag,Ccg0[0,1],'c') 
#    show()
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
plot(lag,Craw0[1,1],'.-c')
FigCCG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Raw count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
#plot(latency/ms*ones(2),[min(Craw[0,1]),max(Craw[0,1])],'--k')
plot(lag,Craw[0,1]/(len(train_ref)*bine*.001),'.-k')
#plot(lag,Craw0[0,1],'.-c')
#show()

# In[ ]:

FigACG.savefig('FigACG.png')
FigCCG.savefig('FigDB_CCG.eps')


# Measure the distribution of synchrony count before injection
synch_width = 1.*5
#--WITH SYNAPSE--
Tref = synch_width*floor(train_ref/synch_width)
lmax = lag[argmax(Craw[0,1])]
x = (train_targ-lmax)*(sign(train_targ-lmax)+1)/2.
x = x[nonzero(x)]
Ttarg = synch_width*floor(train_targ/synch_width)
Tsynch = array(list(set(Tref) & set(Ttarg)))
synch_count = bincount(int64(floor(Tsynch/(Ntrial*phase))),minlength=Nphase)
#--WITHOUT SYNAPSE--
Tref0 = synch_width*floor(train_ref0/synch_width)
lmax0 = lag[argmax(Craw0[0,1])]
x = (train_targ0-lmax0)*(sign(train_targ0-lmax0)+1)/2.
x = x[nonzero(x)]
Ttarg0 = synch_width*floor(x/synch_width)
Tsynch0 = array(list(set(Tref0) & set(Ttarg0)))
synch_count0 = bincount(int64(floor(Tsynch0/(Ntrial*phase))),minlength=Nphase)


# Check the optimal synchrony window 
#Nsynch = 100 
#synch_width_range = linspace(.1,10,Nsynch)
#s = zeros(Nsynch)
#s0 = zeros(Nsynch)
#lmax = lag[argmax(Craw[0,1])]
#print(latency/ms,lmax)
#lmax0 = lag[argmax(Craw0[0,1])]
#for k in range(Nsynch):
#    synch_width = synch_width_range[k]
    #--WITH SYNAPSE--
#    Tref = synch_width*floor(train_ref/synch_width)
#    x = (train_targ-lmax)*(sign(train_targ-lmax)+1)/2.
#    x = x[nonzero(x)]
#    Ttarg = synch_width*floor(x/synch_width)
#    Tsynch = array(list(set(Tref) & set(Ttarg)))
#    synch_count = bincount(int64(floor(Tsynch/(Ntrial*phase))),minlength=Nphase)
#    s[k] = mean(synch_count)
    #--WITHOUT SYNAPSE--
#    Tref0 = synch_width*floor(train_ref0/synch_width)
#    x = (train_targ0-lmax0)*(sign(train_targ0-lmax0)+1)/2.
#    x = x[nonzero(x)]
#    Ttarg0 = synch_width*floor(x/synch_width)
#    Tsynch0 = array(list(set(Tref0) & set(Ttarg0)))
#    synch_count0 = bincount(int64(floor(Tsynch0/(Ntrial*phase))),minlength=Nphase)
#    s0[k] = mean(synch_count0)
#figure()
#plot(synch_width_range,(max(Craw[0,1])-max(Craw0[0,1]))*ones(Nsynch),'--r')
#plot(synch_width_range,s-s0,'--k')
#figure()
#plot(synch_width_range,max(Craw[0,1])*ones(Nsynch),'--r')
#plot(synch_width_range,max(Craw0[0,1])*ones(Nsynch),'--g')
#plot(synch_width_range,s,'o-k')
#plot(synch_width_range,s0,'o-c')
#show()    
# In[ ]:

#print(synch_count,synch_count0)
#print(synch_count-synch_count0)


# Excess synchrony count unbiased estimation
delta = period
Ndelta = int(Ntrial*duration/delta)
count_ref = bincount(int64(floor(train_ref/delta)),minlength=Ndelta)
count_targ = bincount(int64(floor(train_targ/delta)),minlength=Ndelta)
count_synch = bincount(int64(floor(Tsynch/delta)),minlength=Ndelta)
Ndelta_phase = int(Ntrial*phase/delta)
RS_prod = sum(reshape(count_ref*count_synch,(Nphase,Ndelta_phase)),axis=1)
alpha = RS_prod/(delta*synch_count)  
RT_prod = sum(reshape(count_ref*count_targ,(Nphase,Ndelta_phase)),axis=1)
alphaN = alpha[~np.isnan(alpha)]
synch_countN = synch_count[~np.isnan(alpha)]
RT_prodN = RT_prod[~np.isnan(alpha)]
estimate = (synch_countN-RT_prodN/delta)/(1-alphaN)


# Check the result
x = g0/gm*weight_value
FigE = figure()
y = estimate
gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
print(r_value,p_value)
plot(x,y,'ok')
plot(x,gradient*x+intercept,'-r')
#figure()
#plot(x,synch_count-synch_count0,'ob')
#figure()
#plot(synch_count-synch_count0,estimate,'ok')
show()

FigE.savefig('FigDB_prediction-conductance.eps')