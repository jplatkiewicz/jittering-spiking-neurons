import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats,optimize#special,signal,integrate#,interpolate
#from scipy.special import erf,erfc
#%matplotlib inline
#from phy.stats.ccg import correlograms

Ntrial = 1000*5
duration = 1000.*10       
time_step = 0.1

x = np.load('muI.npy')
y = np.load('sigmaI.npy')
Ttarg_coll = np.load('Ttarg.npy')

# Generate the reference train
def PoissonProcess(rate,duration):
    count = np.random.poisson(lam=rate*duration)
    u = np.random.uniform(0,1,count)
    T = duration*np.sort(u)
    return T

Nref = int(10*len(Ttarg_coll[1])/Ntrial)
rate_ref = Nref/duration
Tref = PoissonProcess(rate_ref,duration)
Tref = time_step*np.floor(Tref/time_step)
synch_width = 1.
train_ref = np.tile(Tref,Ntrial)+np.sort(np.tile(np.arange(Ntrial),len(Tref)))*duration

Nsample = len(x)
ind_s = np.array([])
ind_ns = np.array([])
for k in range(Nsample):
    train_targ = np.array(Ttarg_coll[k])
    # Measure the synchrony count for each trial
    Tref_s = synch_width*np.floor(train_ref/synch_width)
    Ttarg_s = synch_width*np.floor(train_targ/synch_width)
    Tsynch = np.array(list(set(Tref_s) & set(Ttarg_s)))
    synch_count = np.bincount(np.int64(np.floor(Tsynch/duration)),minlength=Ntrial)

    # Conditional uniformity test
    interval = int(duration)
    Njitter = 150
    Tjitter = np.tile(train_targ,Njitter)+np.sort(np.tile(np.arange(Njitter),len(train_targ)))*Ntrial*duration
    Tjitter = interval*np.floor(Tjitter/interval)+np.random.uniform(0,interval,len(Tjitter))
    Tjitter = synch_width*np.floor(Tjitter/synch_width)
    Tref_jitter = np.tile(train_ref,Njitter)+np.sort(np.tile(np.arange(Njitter),len(train_ref)))*Ntrial*duration
    Tref_jitter = synch_width*np.floor(Tref_jitter/synch_width)
    Tsynch_jitter = np.array(list(set(Tref_jitter) & set(Tjitter)))
    jitter_synchrony = np.bincount(np.int64(np.floor(Tsynch_jitter/duration)),minlength=Ntrial*Njitter)
    observed_synchrony = np.tile(synch_count,Njitter)
    comparison = np.reshape(np.sign(np.sign(jitter_synchrony-observed_synchrony)+1),(Njitter,Ntrial))
    pvalue = (1+np.sum(comparison,axis=0))/(Njitter+1.)

    # Compute the p-value cumulative distribution
    bins = np.arange(0,1.1,.1)
    count,base = np.histogram(pvalue,bins=bins,density=1)
    X = (base[:-1]+base[1:])/2.
    dX = X[1]-X[0]
    CDF = np.hstack((0,np.cumsum(count)*dX))
    
    # Assess whether the p-value distribution is subuniform or not
    ind = np.where(CDF-base > 0.01)[0]
    if len(ind) == 0:
        ind_s = np.append(ind_s,k)
    else:
        ind_ns = np.append(ind_ns,k)
        
    # Represent the result
    #print(k)
    #plt.figure(1)
    #plt.plot([0,1],[0,1],'--r',linewidth=2)
    #plt.plot(base,CDF,'o-k',markeredgecolor='k')
    #plt.show()    
        
ind_s = np.int64(ind_s)
ind_ns = np.int64(ind_ns)
print(len(ind_s),len(ind_ns))

Fig_musigma = plt.figure()
plt.title('$P$-value Subuniformity',fontsize=18)
plt.xlabel('Mean input (mV)',fontsize=18)
plt.ylabel('Standard-deviation of the input (mV)',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#xlim(-52,-50)
plt.scatter(x[ind_ns],y[ind_ns],color='w',marker='o',edgecolor='k')
plt.scatter(x[ind_s],y[ind_s],color='k',marker='o',edgecolor='k')
plt.show()

Fig_musigma.savefig('Fig2A.png')
Fig_musigma.savefig('Fig2A.eps')