import numpy as np
from scipy import stats
#get_ipython().magic('matplotlib inline')
from ccg import correlograms
import matplotlib.pyplot as plt


wsyn = np.load('wsyn.npy')
train_ref = np.load('train_ref.npy')
train_targ = np.load('train_targ.npy')
train_ref0 = np.load('train_ref0.npy')
train_targ0 = np.load('train_targ0.npy')

# Simulation parameters
Ntrial = 1000#*10
duration = 1000.#*10#*3#0    # duration of the trial
time_step = 0.1            #-in (ms)
Fs = 1/(time_step*.001)
period = 50.#int(duration/10.)
latency = 1.5
Vt = -50.
muI = Vt-1. 
Esyn = 0.
PSC = 25.*4
g0 = PSC/(Esyn-muI)
sigmaW = .5


# Excess synchrony count unbiased estimation USING A SLIDING WINDOW
synch_width = 1.*5
delta = period
Ndelta = int(Ntrial*duration/delta)
window_width = Ntrial*duration*3/4.#10*period
Ninterval_window = int(window_width/delta)
Nwindow = Ndelta-Ninterval_window
Ndt_window = int(window_width/time_step)
estimate = np.array([])
true = np.array([])
Tr = synch_width*np.unique(np.floor(train_ref/synch_width))
Tt = synch_width*np.unique(np.floor((train_targ-latency)/synch_width))
train_synch = np.array(list(set(Tr) & set(Tt)))
Tr0 = synch_width*np.unique(np.floor(train_ref0/synch_width))
Tt0 = synch_width*np.unique(np.floor((train_targ0-latency)/synch_width))
train_synch0 = np.array(list(set(Tr0) & set(Tt0)))
print('basic check',len(train_ref),len(train_synch))
print('fancier check',len(train_ref0),len(train_synch0))
#--
#----Same Noise/No Synapse Condition
#--
estimate_true = np.array([])
#
#--
#
i_old = 0
iA = 0
iB = 0
j_old = 0
jA = 0
jB = 0
i0_old = 0
i0A = 0
i0B = 0
j0_old = 0
j0A = 0
j0B = 0
t0 = 0
while t0 <= Ntrial*duration-window_width:
    #--Reference train
    Tref = np.array([])
    iA = i_old
    if iA < len(train_ref) and train_ref[iA] < t0:
        while iA < len(train_ref) and train_ref[iA] < t0:
            iA += 1
        i_old = iA-1
    if iA < len(train_ref) and train_ref[iA] < t0+window_width-latency:
        iB = iA
        while iB < len(train_ref) and train_ref[iB] < t0+window_width-latency: 
            iB += 1
        Tref = train_ref[iA:iB] 
    #--Target train
    Ttarg = np.array([])
    jA = j_old
    if jA < len(train_targ) and train_targ[jA] < t0+latency:
        while jA < len(train_targ) and train_targ[jA] < t0+latency:
            jA += 1
        j_old = jA-1
    if jA < len(train_targ) and train_targ[jA] < t0+window_width:
        jB = jA
        while jB < len(train_targ) and train_targ[jB] < t0+window_width: 
            jB += 1
        Ttarg = train_targ[jA:jB] 
    #--Compute injected synchrony count estimate
    if len(Tref) >= 1 and len(Ttarg) >= 1:
        Tref = synch_width*np.unique(np.floor((Tref-t0)/synch_width))
        Ttarg = synch_width*np.unique(np.floor((Ttarg-t0-latency)/synch_width))
        Tsynch = np.array(list(set(Tref) & set(Ttarg)))
        count_ref = np.bincount(np.int64(np.floor(Tref/delta)),minlength=Ninterval_window)
        count_targ = np.bincount(np.int64(np.floor(Ttarg/delta)),minlength=Ninterval_window)
        count_synch = np.bincount(np.int64(np.floor(Tsynch/delta)),minlength=Ninterval_window)
        RS_prod = np.sum(count_ref*count_synch)
        RT_prod = np.sum(count_ref*count_targ)
        synch_count = len(Tsynch)
        estimate = np.append(estimate,(delta*synch_count-RT_prod)/(delta*synch_count-RS_prod)*synch_count)
    else:
        estimate = np.append(estimate,0)
    kA = int(t0/time_step)
    kB = int((t0+window_width)/time_step)        
    true = np.append(true,(1+sigmaW*np.mean(wsyn[kA:kB]))*(np.sign(1+sigmaW*np.mean(wsyn[kA:kB]))+1)/2.)#*g0)
        #--
        #----
        #--
    #--Reference train/Synapse off
    Tref0 = np.array([])
    i0A = i0_old
    if i0A < len(train_ref0) and train_ref0[i0A] < t0:
        while i0A < len(train_ref0) and train_ref0[i0A] < t0:
            i0A += 1
        i0_old = i0A-1
    if i0A < len(train_ref0) and train_ref0[i0A] < t0+window_width-latency:
        i0B = i0A
        while i0B < len(train_ref0) and train_ref0[i0B] < t0+window_width-latency: 
            i0B += 1
        Tre0f = train_ref0[i0A:i0B] 
    #--Target train
    Ttarg0 = np.array([])
    j0A = j0_old
    if j0A < len(train_targ0) and train_targ0[j0A] < t0+latency:
        while j0A < len(train_targ0) and train_targ0[j0A] < t0+latency:
            j0A += 1
        j0_old = j0A-1
    if j0A < len(train_targ0) and train_targ0[j0A] < t0+window_width:
        j0B = j0A
        while j0B < len(train_targ0) and train_targ0[j0B] < t0+window_width: 
            j0B += 1
        Ttarg0 = train_targ0[j0A:j0B]        
    #--Compute injected synchrony count estimate
    Tref0 = synch_width*np.unique(np.floor((Tref0-t0)/synch_width))
    Ttarg0 = synch_width*np.unique(np.floor((Ttarg0-t0-latency)/synch_width))
    Tsynch0 = np.array(list(set(Tref0) & set(Ttarg0)))
    synch_count0 = len(Tsynch0)
    #if (synch_count > 0)or(synch_count0 > 0): 
    #    print(t0,synch_count,synch_count0,synch_count-synch_count0)
    estimate_true = np.append(estimate_true,synch_count-synch_count0)
    #--
    #----
    #--
    t0 += delta
    #
    #lagmax = 100.                  
    #bine = 1.                    
    #train = np.append(Tref,Ttarg)
    #cell = np.int64(np.append(np.zeros(len(Tref)),np.ones(len(Ttarg))))
    #ind_sort = np.argsort(train)
    #st = train[ind_sort]*.001
    #sc = cell[ind_sort]
    #C = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
    #lag = (np.arange(len(C[0,1]))-len(C[0,1])/2)*bine
    #train0 = np.append(Tref0,Ttarg0)
    #cell0 = np.int64(np.append(np.zeros(len(Tref0)),np.ones(len(Ttarg0))))    
    #ind_sort = np.argsort(train0)
    #st = train0[ind_sort]*.001
    #sc = cell0[ind_sort]
    #C0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
    #def close_event():
    #    plt.close()
    #fig = plt.figure()
    #timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)
    #plt.plot(lag,C[0,1]/(len(train_ref)*bine*.001),'.-k')
    #plt.plot(lag,C[0,1]/(len(train_ref)*bine*.001),'.-c')
    #timer.start()
    #plt.show()    
    
    
# Check the result
x = true
y = estimate
z = estimate_true
time_win = np.arange(0,len(true),1)*(Ntrial*duration-window_width)/(len(true)-1.)+window_width/2.
time = np.arange(0,len(wsyn),1)*Ntrial*duration/(len(wsyn)-1.)
gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
print(r_value,p_value)
plt.figure()
plt.plot(y,z,'ok')
plt.figure()
plt.plot(time,(1+sigmaW*wsyn)*(np.sign(1+sigmaW*wsyn)+1)/2.,'k')#*g0,'k')
plt.plot(time_win,true,'o-r')
plt.figure()
#plt.plot(time_win,(true-min(true))/(max(true)-min(true)),'o-k')
plt.plot(time_win,estimate,'s-r',markeredgecolor='r')#(estimate-min(estimate))/(max(estimate)-min(estimate)),'s-r')
plt.plot(time_win,estimate_true,'s-m',markeredgecolor='m')#(estimate_true-min(estimate_true))/(max(estimate_true)-min(estimate_true)),'s-m')
plt.figure()
plt.hist(estimate)
plt.figure()
plt.hist(estimate_true)
FigE = plt.figure()
plt.plot(x,y,'ok')
plt.plot(x,gradient*x+intercept,'-r')
#figure()
#plot(x,synch_count-synch_count0,'ob')
#figure()
#plot(synch_count-synch_count0,estimate,'ok')
plt.show()

FigE.savefig('FigDC_tracking-conductance.eps')





#----------------------------------------------------------------------------------

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
