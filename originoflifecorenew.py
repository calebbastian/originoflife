import numpy as np
from collections import Counter, defaultdict, OrderedDict, namedtuple
from numpy.random import RandomState
import copy
from itertools import groupby, product
from numpy.random import RandomState
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stat
import scipy.spatial as spa, scipy.cluster.hierarchy as hc
import sys
import seaborn as sns; sns.set(color_codes=True)
from scipy.integrate import odeint
import numpy.linalg as LA
import pandas as pd
import operator
from itertools import imap

ne = operator.ne

rand = RandomState()

pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
nucleotides = ['A','U','G','C']
comp = lambda x: "".join(pairs[a] for a in x)
hamming = lambda x,y: np.sum(c1 != c2 for c1, c2 in zip(x, y))*1.0

def replicate(x,p=1.0):
	z = ""
	for y in x:
	    ind = nucleotides.index(pairs[y])
	    pp = [(1-p)/3]*4
	    pp[ind] = p
	    z += rand.choice(nucleotides,p=pp)
	return z


class IndependentHDMR(object):

	def __init__(self,X,Y,varlst):

		if X.shape[1] != len(varlst):
			raise ValueError('Varlst must have the same number of columns as X')

		self.X = X 
		self.Y = Y
		self.varlst = varlst 

		self.fu = {}
		self.T = None
		self.d1 = self.d2 = None
		self.N = None
		self.Cx = None
		self.b = None
		self.binds = {}
		self.xmu = None 
		self.xstd = None 
		self.ymu = None
		self.ystd = None
		self.qrs1 = None 
		self.qrs2 = None

	def bases1(self,X):
		return {(i,): np.vstack(list(X[:,i]**j for j in range(1,self.d1+1))).T for i in range(self.Cx)}

	def bases1subspace(self,Xi):
		return np.vstack(list(Xi**j for j in range(1,self.d1+1))).T

	def bases2(self,X):
		return {(i,k): np.vstack(list(X[:,i]**j*X[:,k]**l for j in range(1,self.d2+1) for l in range(1,self.d2+1))).T for i in range(self.Cx) for k in range(self.Cx) if i < k}

	def bases2subspace(self,Xik):
		return np.vstack(list(Xik[:,0]**j*Xik[:,1]**l for j in range(1,self.d2+1) for l in range(1,self.d2+1))).T

	def QRs1(self,X):
		return {k: LA.qr(np.hstack([np.ones((self.N,1)),v])) for k,v in self.bases1(X).iteritems()}

	def QRs2(self,X):
		return {k: LA.qr(np.hstack([np.ones((self.N,1)),v])) for k,v in self.bases2(X).iteritems()}

	def hdmr(self,T=1,d1=1,d2=1):

		self.T, self.d1, self.d2 = T, d1, d2

		self.xmu = np.mean(self.X,axis=0)
		self.xstd = np.std(self.X,axis=0)
		self.ymu = np.mean(self.Y,axis=0)
		self.ystd = np.std(self.Y,axis=0)

		Xstd = (self.X - self.xmu) / self.xstd
		Ystd = (self.Y - self.ymu)# / self.ystd

		self.N, self.Cx = Xstd.shape

		self.qrs1 = self.QRs1(Xstd)
		self.qrs2 = self.QRs2(Xstd)

		if T == 1:
			Bases = np.hstack(list(q[:,1:] for q,r in self.qrs1.values()))
			i = 0
			for k,(q,r) in self.qrs1.iteritems():
				self.binds[k] = range(i,i+q[:,1:].shape[1])
				i += q[:,1:].shape[1]
				#binds = dict(k: q[:,1:].shape[1] for k,(q,r) in self.qrs1.iteritems())
		elif T == 2:
			Bases = np.hstack(list(q[:,1:] for q,r in self.qrs1.values()) + list(q[:,1:] for q,r in self.qrs2.values()))
			i = 0
			for k,(q,r) in self.qrs1.iteritems():
				self.binds[k] = range(i,i+q[:,1:].shape[1])
				i += q[:,1:].shape[1]
			for k,(q,r) in self.qrs2.iteritems():
				self.binds[k] = range(i,i+q[:,1:].shape[1])
				i += q[:,1:].shape[1]

		#reg = np.logspace(start=0.0,stop=8.0,num=10)
		#for r in reg:
		b = LA.solve(np.dot(Bases.T,Bases)+10.0,np.dot(Bases.T,self.Y))
			#bi = LA.pinv(np.dot(Bases.T,Bases))
			#self.b = np.dot(bi,np.dot(Bases.T,Ystd)) 
		self.b = b

		for k,(q,r) in self.qrs1.iteritems():
			self.fu[k] = np.dot(q[:,1:],self.b[self.binds[k]])
		for k,(q,r) in self.qrs2.iteritems():
			self.fu[k] = np.dot(q[:,1:],self.b[self.binds[k]])
			#b = LA.pinv(np.dot(Bases.T,Bases)+r,np.dot(Bases.T,self.Y))
		err = np.mean((np.dot(Bases,self.b) - Ystd)**2)
		r2 = 1- err/self.ystd**2
		print "r2", r2
			#print r, err

		return self

	def truth_plot(self):
		df = pd.DataFrame(self.Y,index=range(self.Y.shape[0]),columns=['Ytrue'])
		if self.T == 1:
			Bases = np.hstack(list(q[:,1:] for q,r in self.qrs1.values()))
		elif self.T == 2:
			Bases = np.hstack(list(q[:,1:] for q,r in self.qrs1.values()) + list(q[:,1:] for q,r in self.qrs2.values()))
		df['Ypred'] = np.dot(Bases,self.b) + self.ymu.item()
		df['C'] = np.ones(self.Y.shape[0])
		#print df
		df.plot.hexbin(x='Ytrue',y='Ypred',C='C',reduce_C_function=np.sum,gridsize=50)
		return self

	def sensitivity_indices(self):
		d = {}
		for k in self.binds.keys():
			#fu = self.evaluate_cmp_fxn(k)
			#print k, np.mean(fu)
			fu = self.fu[k]
			Fu = 1.0/self.N*np.dot(fu.T,fu).item()
			#print Fu, self.ystd**2
			d[k] = Fu/self.ystd**2
		s1 = np.sum(v for k,v in d.iteritems() if len(k) == 1)
		s2 = np.sum(v for k,v in d.iteritems() if len(k) == 2)
		print "S1", s1, "S2", s2
		for k in self.binds.keys():
			print ','.join(list(self.varlst[i] for i in k)), d[k]
		return d

	def evaluate_cmp_fxn(self,Xu,subspace):

		XU = (Xu - self.xmu[list(subspace)]) / self.xstd[list(subspace)]
		Xstd = (self.X - self.xmu) / self.xstd
		N = XU.shape[0]
 
		if len(subspace) == 1:
			r = self.qrs1[subspace][1]
			r1 = LA.pinv(r)
			Bases = self.bases1subspace(XU[:,0])
			basis = np.hstack([np.ones((N,1)),Bases])
			q = np.dot(basis,r1)
			fu = np.dot(q[:,1:],self.b[self.binds[subspace]])
		elif len(subspace) == 2:
			r = self.qrs2[subspace][1]
			r1 = LA.pinv(r)
			#print r1
			Bases = self.bases2subspace(XU)
			basis = np.hstack([np.ones((N,1)),Bases])
			q = np.dot(basis,r1)
			fu = np.dot(q[:,1:],self.b[self.binds[subspace]])

		return fu


class System(object):

	def __init__(self,L,n,population,R,rds,rss,reprate,rdecay,clay_rep,clay_olig,pclay,hitting='R',hitting_V=None,power_fitness=6,power_similarity=3,power_rep=1,normalize_reprate=None,normalize_clay_reprate=None,linear=False,string=''):

		self.L = L
		self.n = n
		self.count = Counter(population) # master counter
		self.counts = [copy.copy(self.count)] # copies of counters
		self.R = R 
		self.rds = rds 
		self.rss = rss 
		self.reprate = reprate 
		self.rdecay = rdecay
		self.clay_rep = clay_rep
		self.clay_olig = clay_olig
		self.pclay = pclay
		self.hitting = hitting 
		self.hitting_V = hitting_V 
		self.power_fitness = -np.log(power_fitness) / L
		self.power_similarity = -np.log(power_similarity) / L
		self.power_rep = -np.log(power_rep) / L
		self.s = 0
		self.T = [0]
		self.population = population
		self.string = string

		self.hamming_R = lambda x: np.sum(s not in y for s,y in zip(x,self.R))*1.0
		self.cache_h = {}
		self.cache_hh = {}
		self.cache_f = {}
		self.cache_fidel = {}
		self.cache_s = {}
		self.cache_ss = {}
		self.sym_similarity = lambda x,y: min(self.Hamming(x,y),self.Hamming(x,comp(y)),self.Hamming(comp(x),y),self.Hamming(comp(x),comp(y)))
		
		if not linear:
			self.power_fitness = -np.log(power_fitness) / L
			self.power_similarity = -np.log(power_similarity) / L
			self.power_rep = -np.log(power_rep) / L
			self.fitness_rep = lambda x: np.exp(-self.power_fitness*self.Hamming_R(x))
			self.fitness_fidelity = lambda x: np.exp(-self.power_rep*self.Hamming_R(x))
			self.similarity = lambda x,y: np.exp(-self.power_similarity*self.Sym_Similarity(x,y))
		else:
			self.power_fitness = power_fitness
			self.power_similarity = power_similarity
			self.power_rep = power_rep
			self.fitness_rep = lambda x: 1 + (1-self.power_fitness)/(-1*L)*self.Hamming_R(x)
			self.fitness_fidelity = lambda x: 1 + (1-self.power_rep)/(-1*L)*self.Hamming_R(x)
			self.similarity = lambda x,y: 1 + (1-self.power_similarity)/(-1*L)*self.Sym_Similarity(x,y)
		
		self.Pmf = {}

		self.random_genome = lambda : ''.join(rand.choice(nucleotides,size=self.L))
		self.get = lambda x: next(iter(x))

		# double strand formation rates
		self.dsrates = {
		    (k,frozenset([comp(self.get(k))])): np.abs(self.rds*self.get_count(k)*self.get_count(frozenset([comp(self.get(k))])))
		    for k in self.count.keys() if len(k) == 1
		}
		# double strand dissociation rates
		self.dissrates = {
		    k: np.abs(self.rss*self.get_count(k))
		    for k in self.count.keys() if len(k) == 2
		}
		# replication rates
		self.reprates = {
		    (kx,ky): np.abs(self.reprate*self.Fitness_rep(self.get(kx))*self.Similarity(self.get(kx),self.get(ky))*self.get_count(kx)*(self.get_count(ky)-(1 if kx==ky else 0))) 
		    for kx in self.count.keys() for ky in self.count.keys()
		    if (len(kx) == 1) and (len(ky) == 1)
		}
		# decay rates
		self.decayrates = {
		    k: np.abs(self.rdecay*self.get_count(k))
		    for k in self.count.keys() if len(k) == 1
		}
		# clay replication rates
		self.clayreprates = {
			k: np.abs(self.clay_rep*self.get_count(k))
			for k in self.count.keys() if len(k) == 1
		}

		if normalize_reprate is not None:
			self.normalize_reprates(normalize_reprate)

		if normalize_clay_reprate is not None:
			self.normalize_clay_reprates(normalize_clay_reprate)

		ds_total = np.sum(self.dsrates.values())/2
		dis_total = np.sum(self.dissrates.values())
		rep_total = np.sum(self.reprates.values())
		decay_total = np.sum(self.decayrates.values())
		clay_rep_total = np.sum(self.clayreprates.values())
		clay_olig_total = self.clay_olig

		# get the rates and time increment 
		r = ds_total + dis_total + rep_total + decay_total + clay_rep_total + clay_olig_total

		# get the probabilities of the reactions
		p_dsrates = ds_total/r 
		p_dissrates = dis_total/r 
		p_reprates = rep_total/r
		p_decayrates = decay_total/r
		p_clayrates = clay_rep_total/r
		p_clay_olig_rates = clay_olig_total/r
		self.probs = [[p_dsrates,p_dissrates,p_reprates,p_decayrates,p_clayrates,p_clay_olig_rates]]

	def normalize_reprates(self,val):
		self.reprate = val * self.reprate / np.sum(self.reprates.values())
		self.reprates = {
		    (kx,ky): np.abs(self.reprate*self.Fitness_rep(self.get(kx))*self.Similarity(self.get(kx),self.get(ky))*self.get_count(kx)*(self.get_count(ky)-(1 if kx==ky else 0))) 
		    for kx in self.count.keys() for ky in self.count.keys()
		    if (len(kx) == 1) and (len(ky) == 1)
		}
		return self

	def normalize_clay_reprates(self,val):
		self.clay_rep = val * self.clay_rep / np.sum(self.clayreprates.values())
		self.clayreprates = {
		    k: np.abs(self.clay_rep*self.get_count(k)) 
		    for k in self.count.keys() if len(k) == 1
		}
		return self

	def Fitness_rep(self,x):
	    if x in self.cache_f:
	        return self.cache_f[x]
	    else:
	        self.cache_f[x] = self.fitness_rep(x) 
	        return self.cache_f[x]

	def Fitness_fidelity(self,x):
	    if x in self.cache_fidel:
	        return self.cache_fidel[x]
	    else:
	        self.cache_fidel[x] = self.fitness_fidelity(x) 
	        return self.cache_fidel[x]

	def Hamming_R(self,x):
		if x in self.cache_h:
		    return self.cache_h[x]
		else:
		    self.cache_h[x] = self.hamming_R(x) 
		    return self.cache_h[x]

	def Sym_Similarity(self,x,y):
		if (x,y) in self.cache_ss:
		    return self.cache_ss[(x,y)]
		else:
		    self.cache_ss[(x,y)] = self.cache_ss[(y,x)] = self.sym_similarity(x,y) 
		    return self.cache_ss[(x,y)]

	def Similarity(self,x,y):
		if (x,y) in self.cache_s:
		    return self.cache_s[(x,y)]
		else:
		    self.cache_s[(x,y)] = self.cache_s[(y,x)] = self.similarity(x,y) 
		    return self.cache_s[(x,y)]

	def Hamming(self,x,y):
		if (x,y) in self.cache_hh:
		    return self.cache_hh[(x,y)]
		else:
		    self.cache_hh[(x,y)] = self.cache_hh[(y,x)] = hamming(x,y) 
		    return self.cache_hh[(x,y)] 

	def get_count(self,x): 
		return self.count[x] if x in self.count else 0

	def get_count_i(self,x,i):
		return self.counts[i][x] if x in self.counts[i] else 0

	def hitting_Vol(self):
		total = np.sum(len(k)*v for k,v in self.count.iteritems())*1.0
		Rcontent = np.sum(v for k,v in self.count.iteritems() if len(k) == 1 and min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k)))) == 0)*1.0
		Rcontent += np.sum(v for k,v in self.count.iteritems() if len(k) == 2 and (self.Hamming_R(list(k)[0]) == 0 or self.Hamming_R(list(k)[1]) == 0))*1.0
		if total > 0 and Rcontent / total >= self.hitting_V:
		    return True
		else:
		    return False

	@classmethod
	def hitting_Vol_population(cls,population,R,hitting_V):
		Hamming_R = lambda x: np.sum(s not in y for s,y in zip(x,R))*1.0
		get = lambda x: next(iter(x))
		count = Counter(population)
		total = np.sum(len(k)*v for k,v in count.iteritems())*1.0
		Rcontent = np.sum(v for k,v in count.iteritems() if len(k) == 1 and min(Hamming_R(get(k)),Hamming_R(comp(get(k)))) == 0)*1.0
		Rcontent += np.sum(v for k,v in count.iteritems() if len(k) == 2 and (Hamming_R(list(k)[0]) == 0 or Hamming_R(list(k)[1]) == 0))*1.0
		if total > 0 and Rcontent / total >= hitting_V:
		    return True
		else:
		    return False

	@classmethod
	def hitting_R_population(cls,population,R):
		Hamming_R = lambda x: np.sum(s not in y for s,y in zip(x,R))*1.0
		get = lambda x: next(iter(x))
		count = Counter(population)
		Rcontent = np.sum(v for k,v in count.iteritems() if len(k) == 1 and min(Hamming_R(get(k)),Hamming_R(comp(get(k)))) == 0)*1.0
		Rcontent += np.sum(v for k,v in count.iteritems() if len(k) == 2 and (Hamming_R(list(k)[0]) == 0 or Hamming_R(list(k)[1]) == 0))*1.0
		if Rcontent > 0:
		    return True
		else:
		    return False

	def hitting_R(self):
		return any(min(self.Hamming_R(self.get(x)),self.Hamming_R(comp(self.get(x))))==0 for x in self.count.keys())

	def simulate_reactions(self,N):

	    if self.hitting == 'R':
	        hitting = self.hitting_R 
	    elif self.hitting == 'V':
	        hitting = self.hitting_Vol

	    if hitting():
	    	return self
	    else:
	    	for i in range(1,N+1):
	    		if hitting(): break
	    		print "Reaction", i
	    		self.next_reaction()
	    	return self

	def plot_concentration_curves(self):

		# get counts for genomes by distance across time		
		get_count_series = lambda counters,i: np.asarray(list(np.sum(v for k,v in counter.iteritems() if len(k)==1 and min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k))))==i)*1.0 + 
		np.sum(v for k,v in counter.iteritems() if len(k) == 2 and (self.Hamming_R(list(k)[0]) == i or self.Hamming_R(list(k)[1]) == i))*1.0 
		for counter in counters))

		plt.figure()
		plt.title('Growth curves by distance')
		Series = []
		for i in range(self.L+1):
			series = get_count_series(self.counts,i)
			Series.append(series)
			plt.plot(np.log(1+np.asarray(self.T)),np.log(1+series),label=str(i))
		plt.legend()
		plt.xlabel('Log(1+time)')
		plt.ylabel('Log(1+count)')

		Series = np.asarray(Series)
		Series /= np.sum(Series,axis=0)
		plt.figure()
		plt.title('Concentration curves by distance')
		for i,series in enumerate(Series):
			plt.plot(np.log(1+np.asarray(self.T)),series,label=str(i))
		plt.legend()
		plt.xlabel('Log(1+time)')
		plt.ylabel('Concentration')

		plt.figure()
		reactions = ['RNA-DS-FORMATION','RNA-DS-DISSOC','RNA-REP','RNA-DECAY','CLAY-REP','CLAY-OLIG']
		Series = np.asarray(self.probs).T
		for i,series in enumerate(Series):
			plt.plot(np.log(1+np.asarray(self.T)),np.log(1+series),label=reactions[i])
		plt.legend()
		plt.xlabel('Log(1+time)')
		plt.ylabel('Log(1+Probability)')

		return self

	def plot_distances(self,percentile=1.0):

		"""
		plot dendrogram for symmetric distances
		"""

		ind = max(int(percentile*len(self.counts))-1,0)
		genomes = np.asarray(list(self.get(k) for k,v in self.counts[ind].iteritems() for i in range(v)))

		# genome distances (symmetric)
		distances = np.asarray(list(list(self.Sym_Similarity(g1,g2) for g1 in genomes) for g2 in genomes))
		linkage = hc.linkage(spa.distance.squareform(distances),method='average')
		cg = sns.clustermap(distances, row_linkage=linkage, col_linkage=linkage)
		cg.ax_heatmap.set_yticklabels(labels = genomes)
		cg.ax_heatmap.set_xticklabels(labels = genomes)
		plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
		plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
		#plt.figure()
		#hc.dendrogram(linkage)
		return self

	def generate_odes(self):

		single_genomes = list(set([frozenset([''.join(x)]) for x in product(nucleotides,repeat=self.L)]))
		double_genomes = list(set([frozenset([self.get(g),comp(self.get(g))]) for g in single_genomes]))

		# create variables
		for s in single_genomes:
			string = self.get(s) + ' = 0'
			exec(string) in globals(), locals()
			#exec(self.get(s)) in globals(), locals()
			#print string
		for s in double_genomes:
			string = list(s)[0]+list(s)[1] + ' = 0'
			exec(string) in globals(), locals()

		counter = Counter(self.population)
		for k,v in counter.iteritems():
			string = self.get(k) + ' = ' + str(v)
			exec(string) in globals(), locals()
			#print string

		# create transition kernel for RNA polymerization
		Q_rep = np.zeros(((4**self.L)**2,4**self.L))
		inds = list(0.01*i*(4**self.L)**2 for i in range(1,101))
		dum = 1
		for i,(k1,k2) in enumerate(product(single_genomes,single_genomes)):
			p = self.Fitness_fidelity(self.get(k1))
			ck2 = comp(self.get(k2))
			if p < 1:
				distances = list(self.Hamming(ck2,self.get(g)) for g in single_genomes)
				counter = Counter(distances)
				Q_rep[i,:] = list(1.0/counter[d]*self.pmf(d,self.L,1-p) for d in distances)
			else:
				Q_rep[i,:] = list(1 if self.get(g) == ck2 else 0 for g in single_genomes)

			if i == int(dum*0.01*(4**self.L)**2):
				print "Q_rep", i*1.0/(4**self.L)**2
				dum += 1

		# generate nu_rep
		def get_nu_rep(**kwargs):
			nu = np.zeros((1,len(single_genomes)**2))
			inds = list(0.01*i*len(single_genomes)**2 for i in range(1,101))
			dum = 1
			for i,(k1,k2) in enumerate(product(single_genomes,single_genomes)):
				nu[0,i] = self.reprate*self.Fitness_rep(self.get(k1))*self.Similarity(self.get(k1),self.get(k2))*kwargs[self.get(k1)]*(kwargs[self.get(k2)]-(1 if k1==k2 else 0))
				#if i == int(dum*0.01*(len(single_genomes)**2)):
				#	print "nu_rep", i*1.0/(len(single_genomes)**2)
				#	dum += 1
			T = nu.sum()
			nu /= T
			return T, nu


		def derivative(X,t):
			exec(','.join([self.get(k) for k in single_genomes]+[list(k)[0]+list(k)[1] for k in double_genomes]) + ' = X') in globals(), locals()

			#for s in double_genomes:
			#	string = list(s)[0]+list(s)[1] + ' = 0'
			#	exec(string) in globals(), locals()

			# get nu_rep
			kwargs = {}
			for k in single_genomes:
				exec("kwargs['"+self.get(k)+"'] = "+self.get(k)) in globals(), locals()
			T, nu_rep = get_nu_rep(**kwargs)


			# get rep probabilities
			prod_rep = np.dot(nu_rep,Q_rep)
			Tdis = T*prod_rep

			# update kwargs with double stranded genomes
			for k in double_genomes:
				string = list(k)[0]+list(k)[1]
				exec("kwargs['"+string+"'] = "+string) in globals(), locals()

			# ds dissoc and assoc
			def Kwargs(k):
				try: 
					return kwargs[self.get(k)+comp(self.get(k))] 
				except: 
					return kwargs[comp(self.get(k))+self.get(k)]
			dsdissoc = np.asarray(list(self.rss*Kwargs(k) for k in single_genomes))
			dsform = np.asarray(list(self.rds*kwargs[self.get(k)]*kwargs[comp(self.get(k))] for k in single_genomes))

			single_diff = Tdis + dsdissoc - dsform

			# ds dissoc
			dsdissoc = np.asarray(list(self.rss*kwargs[list(k)[0]+list(k)[1]] for k in double_genomes))
			dsform = np.asarray(list(self.rds*kwargs[list(k)[0]]*kwargs[list(k)[1]] for k in double_genomes))

			double_diff = dsform - dsdissoc

			return np.hstack((single_diff.flatten(),double_diff))

		X = []
		for g in single_genomes:
			exec('X.append('+self.get(g)+')') in globals(), locals()
		for g in double_genomes:
			exec('X.append('+list(g)[0]+list(g)[1]+')') in globals(), locals()
		#print derivative(X,t=0)
		c = Counter(self.population)
		initial = np.asarray(list(c[g] for g in single_genomes+double_genomes))
		times = np.linspace(0,1,1000)
		z = odeint(derivative, initial, times)
		plt.figure()
		for a in z.T:
			plt.plot(times,np.log(1+a))

	def next_reaction(self):

		"""
		simulates one reaction
		"""

		ds_total = np.sum(self.dsrates.values())/2
		dis_total = np.sum(self.dissrates.values())
		rep_total = np.sum(self.reprates.values())
		decay_total = np.sum(self.decayrates.values())
		clay_rep_total = np.sum(self.clayreprates.values())
		clay_olig_total = self.clay_olig

		# get the rates and time increment 
		r = ds_total + dis_total + rep_total + decay_total + clay_rep_total + clay_olig_total
		self.s += rand.exponential(scale=1/r)
		self.T.append(self.s)

		# get the probabilities of the reactions
		p_dsrates = ds_total/r 
		p_dissrates = dis_total/r 
		p_reprates = rep_total/r
		p_decayrates = decay_total/r
		p_clayrates = clay_rep_total/r
		p_clay_olig_rates = clay_olig_total/r

		self.probs.append([p_dsrates,p_dissrates,p_reprates,p_decayrates,p_clayrates,p_clay_olig_rates])

		print self.string
		print self.s
		print "p_ds=", p_dsrates, "p_diss=", p_dissrates, "p_rep=", p_reprates, "p_decay=", p_decayrates, "p_clayrate=", p_clayrates, "p_clay_olig=", p_clay_olig_rates
		# choose the reaction type
		reaction = rand.choice([1,2,3,4,5,6],p=[p_dsrates,p_dissrates,p_reprates,p_decayrates,p_clayrates,p_clay_olig_rates])

		if reaction == 1:

			print "RNA-DS-FORMATION"
			p_reactions = [v/2/ds_total for v in self.dsrates.values()]
			ind = rand.choice(np.arange(len(self.dsrates)),p=p_reactions)
			key1,key2 = self.dsrates.keys()[ind]
			self._rxn_ds(key1,key2)

		elif reaction == 2:

			print "RNA-DS-DISSOC"
			p_reactions = [v/dis_total for v in self.dissrates.values()]
			ind = rand.choice(np.arange(len(self.dissrates)),p=p_reactions)
			key = self.dissrates.keys()[ind]
			self._rxn_ss(key)

		elif reaction == 3:

			print "RNA-REP"
			p_reactions = [v/rep_total for v in self.reprates.values()]
			ind = rand.choice(np.arange(len(self.reprates)),p=p_reactions)
			key1,key2 = self.reprates.keys()[ind]
			self._rxn_rnarep(key1,key2)

		elif reaction == 4:

			print "RNA-DECAY"
			p_reactions = [v/decay_total for v in self.decayrates.values()]
			ind = rand.choice(np.arange(len(p_reactions)),p=p_reactions)
			keynew = self.decayrates.keys()[ind]
			self._rxn_decay(keynew)

		elif reaction == 5:

			print "CLAY-REP"
			p_reactions = [v/clay_rep_total for v in self.clayreprates.values()]
			ind = rand.choice(np.arange(len(p_reactions)),p=p_reactions)
			key = self.clayreprates.keys()[ind]
			self._rxn_clay_rep(key)

		elif reaction == 6:

			print "CLAY-OLIG"
			key = frozenset([self.random_genome()])
			self._rxn_clay_olig(key)


		self.counts.append(copy.copy(self.count)) # for each reaction type, get the reaction

		total = np.sum(len(k)*v for k,v in self.count.iteritems())*1.0
		Rcontent = np.sum(v for k,v in self.count.iteritems() if len(k) == 1 and min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k))))== 0)*1.0
		Rcontent += np.sum(v for k,v in self.count.iteritems() if len(k) == 2 and (self.Hamming_R(list(k)[0]) == 0 or self.Hamming_R(list(k)[1]) == 0))*1.0

		lst = []
		for k,v in self.count.iteritems():
			if len(k) == 1 and v > 0:
			    for i in range(v):
			        lst.append(min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k)))))
			elif len(k) == 2 and v > 0:
			    k1,k2 = list(k)
			    for i in range(v):
			        lst.append(min(self.Hamming_R(k1),self.Hamming_R(k2)))

		mindis = Counter(lst)

		print mindis, Rcontent, total, Rcontent/total if total > 0 else 0
		return self

	def pmf(self,d,L,p):
		if (d,L,p) in self.Pmf:
		    return self.Pmf[(d,L,p)]
		else:
		    self.Pmf[(d,L,p)] = stat.binom.pmf(d,L,1-p)
		    return self.Pmf[(d,L,p)]

	def get_time(self,percentile=1.0):
		ind = max(int(percentile*len(self.counts))-1,0)
		return self.T[ind]

	def measures(self,percentile=1.0):

		"""
		Generates the measure-kernel-functions nu, Q, and mu
		"""

		ind = max(int(percentile*len(self.counts))-1,0)
		#keys = self.count.keys()
		keys = self.counts[ind].keys()
		nu = np.zeros((1,len(keys)**2))
		inds = list(0.01*i*len(keys)**2 for i in range(1,101))
		dum = 1
		for i,(k1,k2) in enumerate(product(keys,keys)):
			nu[0,i] = self.reprate*self.Fitness_rep(self.get(k1))*self.Similarity(self.get(k1),self.get(k2))*self.get_count_i(k1,ind)*(self.get_count_i(k2,ind)-(1 if k1==k2 else 0))
			if i == int(dum*0.01*(len(keys)**2)):
				print "nu", i*1.0/(len(keys)**2)
				dum += 1
		nu /= nu.sum()

		genomes = list(''.join(x) for x in product(nucleotides,repeat=self.L))

		Q = np.zeros((nu.size,4**self.L))
		inds = list(0.01*i*nu.size**2 for i in range(1,101))
		dum = 1
		for i,(k1,k2) in enumerate(product(keys,keys)):
			p = self.Fitness_fidelity(self.get(k1))
			ck2 = comp(self.get(k2))
			if p < 1:
				distances = list(self.Hamming(ck2,g) for g in genomes)
				counter = Counter(distances)
				Q[i,:] = list(1.0/counter[d]*self.pmf(d,self.L,1-p) for d in distances)
			else:
				Q[i,:] = list(1 if g == ck2 else 0 for g in genomes)

			if i == int(dum*0.01*nu.size):
				print "Q", i*1.0/nu.size
				dum += 1

		# get probabilities
		prod = np.dot(nu,Q)

		distances = np.asarray(list(min(self.Hamming_R(g),self.Hamming_R(comp(g))) for g in genomes))
		distanceset = sorted(list(set(distances.tolist())))
		probs = []
		for j in distanceset:
			#inds = list(i for i,x in enumerate(distances) if x == j)
			inds = np.where(distances == j)
			probs.append(prod[0,inds].sum())
		print distanceset, probs
		return probs

	def _rxn_ds(self,key1,key2):
		# create new key
		keynew = frozenset([self.get(key1),self.get(key2)])
		# update counter
		self.count.update({key1:-1,key2:-1,keynew:1})

		# update data structures
		# four updates in ds-formation rates
		key1comp = frozenset([comp(self.get(key1))])
		key2comp = frozenset([comp(self.get(key2))])
		self.dsrates[(key1,key1comp)] = self.dsrates[(key1comp,key1)] = self.rds*self.get_count(key1)*self.get_count(key1comp)
		self.dsrates[(key2,key2comp)] = self.dsrates[(key2comp,key2)] = self.rds*self.get_count(key2)*self.get_count(key2comp)
		# one update in diss rates
		self.dissrates[keynew] = self.rss*self.get_count(keynew)
		# 4 * len(keys) updates to reprates
		for key in self.count.keys():
		    if len(key) == 1:
		        self.reprates[(key1,key)] = self.reprate*self.Fitness_rep(self.get(key1))*self.Similarity(self.get(key1),self.get(key))*self.get_count(key1)*(self.get_count(key)-(1 if key1==key else 0))
		        self.reprates[(key2,key)] = self.reprate*self.Fitness_rep(self.get(key2))*self.Similarity(self.get(key2),self.get(key))*self.get_count(key2)*(self.get_count(key)-(1 if key2==key else 0))
		        self.reprates[(key,key1)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(key1))*self.get_count(key)*(self.get_count(key1)-(1 if key1==key else 0))
		        self.reprates[(key,key2)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(key2))*self.get_count(key)*(self.get_count(key2)-(1 if key2==key else 0))
		# two updates to decay rates
		self.decayrates[key1] = self.rdecay*self.get_count(key1)
		self.decayrates[key2] = self.rdecay*self.get_count(key2)
		# two updates to clayrep
		self.clayreprates[key1] = self.clay_rep*self.get_count(key1)
		self.clayreprates[key2] = self.clay_rep*self.get_count(key2)
		
		return self

	def _rxn_ss(self,key):
		frozens = list(frozenset([a]) for a in key)
		key1 = frozens[0]
		key2 = frozens[1]
		self.count.update({key:-1,key1:1,key2:1})

		# four updates in ds-formation rates
		key1comp = frozenset([comp(self.get(key1))])
		key2comp = frozenset([comp(self.get(key2))])
		self.dsrates[(key1,key1comp)] = self.dsrates[(key1comp,key1)] = self.rds*self.get_count(key1)*self.get_count(key1comp)
		self.dsrates[(key2,key2comp)] = self.dsrates[(key2comp,key2)] = self.rds*self.get_count(key2)*self.get_count(key2comp)
		# one update in diss rates
		self.dissrates[key] = self.rss*self.get_count(key)
		# 4 * len(keys) updates to reprates
		for key in self.count.keys():
		    if len(key) == 1:
		        self.reprates[(key1,key)] = self.reprate*self.Fitness_rep(self.get(key1))*self.Similarity(self.get(key1),self.get(key))*self.get_count(key1)*(self.get_count(key)-(1 if key1==key else 0))
		        self.reprates[(key2,key)] = self.reprate*self.Fitness_rep(self.get(key2))*self.Similarity(self.get(key2),self.get(key))*self.get_count(key2)*(self.get_count(key)-(1 if key2==key else 0))
		        self.reprates[(key,key1)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(key1))*self.get_count(key)*(self.get_count(key1)-(1 if key1==key else 0))
		        self.reprates[(key,key2)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(key2))*self.get_count(key)*(self.get_count(key2)-(1 if key2==key else 0))
		# two updates to decay rates
		self.decayrates[key1] = self.rdecay*self.get_count(key1)
		self.decayrates[key2] = self.rdecay*self.get_count(key2)
		# two updates to clayrep
		self.clayreprates[key1] = self.clay_rep*self.get_count(key1)
		self.clayreprates[key2] = self.clay_rep*self.get_count(key2)

		return self

	def _rxn_rnarep(self,key1,key2):

		prob = self.Fitness_fidelity(self.get(key1))#1-hamming(reference,next(iter(key1)))/len(next(iter(key1)))
		rep = frozenset([replicate(self.get(key2),p=prob)])
		self.count.update({rep:1})		

		# 2 updates in ds-formation rates
		repcomp = frozenset([comp(self.get(rep))])
		self.dsrates[(rep,repcomp)] = self.dsrates[(repcomp,rep)] = self.rds*self.get_count(rep)*self.get_count(repcomp)
		# 0 updates in diss rates
		# 2 * len(keys) updates to reprates
		fitrep = self.Fitness_rep(self.get(rep))
		countrep = self.get_count(rep)
		for key in self.count.keys():
			if len(key) == 1:
				countkey = self.get_count(key)
				sim = self.Similarity(self.get(rep),self.get(key))
				self.reprates[(rep,key)] = self.reprate*fitrep*sim*countrep*(countkey-(1 if rep==key else 0))
				self.reprates[(key,rep)] = self.reprate*self.Fitness_rep(self.get(key))*sim*countkey*(countrep-(1 if rep==key else 0))
		# 1 update to decay rates
		self.decayrates[rep] = self.rdecay*self.get_count(rep)
		# 1 update to clayrep
		self.clayreprates[rep] = self.clay_rep*self.get_count(rep)

		return self

	def _rxn_decay(self,keynew):

		self.count.update({keynew:-1})

		# 2 updates in ds-formation rates
		keycomp = frozenset([comp(self.get(keynew))])
		self.dsrates[(keynew,keycomp)] = self.dsrates[(keycomp,keynew)] = self.rdecay*self.get_count(keynew)*self.get_count(keycomp)
		# 0 updates in diss rates
		# 4 * len(keys) updates to reprates
		for key in self.count.keys():
		    if len(key) == 1:
		        self.reprates[(keynew,key)] = self.reprate*self.Fitness_rep(self.get(keynew))*self.Similarity(self.get(keynew),self.get(key))*self.get_count(keynew)*(self.get_count(key)-(1 if keynew==key else 0))
		        self.reprates[(key,keynew)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(keynew))*self.get_count(key)*(self.get_count(keynew)-(1 if keynew==key else 0))
		# 1 update to decay rates
		self.decayrates[keynew] = self.rdecay*self.get_count(keynew)
		# one update to clayrep
		self.clayreprates[keynew] = self.clay_rep*self.get_count(keynew)

		return self

	def _rxn_clay_rep(self,key):

		rep = frozenset([replicate(self.get(key),p=self.pclay)])
		self.count.update({rep:1})

		# 2 updates in ds-formation rates
		repcomp = frozenset([comp(self.get(rep))])
		self.dsrates[(rep,repcomp)] = self.dsrates[(repcomp,rep)] = self.rds*self.get_count(rep)*self.get_count(repcomp)
		# 0 updates in diss rates
		# 4 * len(keys) updates to reprates
		for key in self.count.keys():
		    if len(key) == 1:
		        self.reprates[(rep,key)] = self.reprate*self.Fitness_rep(self.get(rep))*self.Similarity(self.get(rep),self.get(key))*self.get_count(rep)*(self.get_count(key)-(1 if rep==key else 0))
		        self.reprates[(key,rep)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(rep))*self.get_count(key)*(self.get_count(rep)-(1 if rep==key else 0))
		# 1 update to decay rates
		self.decayrates[rep] = self.rdecay*self.get_count(rep)
		# 1 update to clayrep
		self.clayreprates[rep] = self.clay_rep*self.get_count(rep)

		return self

	def _rxn_clay_olig(self,rep):

		self.count.update({rep:1})

		# 2 updates in ds-formation rates
		repcomp = frozenset([comp(self.get(rep))])
		self.dsrates[(rep,repcomp)] = self.dsrates[(repcomp,rep)] = self.rds*self.get_count(rep)*self.get_count(repcomp)
		# 0 updates in diss rates
		# 4 * len(keys) updates to reprates
		for key in self.count.keys():
		    if len(key) == 1:
		        self.reprates[(rep,key)] = self.reprate*self.Fitness_rep(self.get(rep))*self.Similarity(self.get(rep),self.get(key))*self.get_count(rep)*(self.get_count(key)-(1 if rep==key else 0))
		        self.reprates[(key,rep)] = self.reprate*self.Fitness_rep(self.get(key))*self.Similarity(self.get(key),self.get(rep))*self.get_count(key)*(self.get_count(rep)-(1 if rep==key else 0))
		# 1 update to decay rates
		self.decayrates[rep] = self.rdecay*self.get_count(rep)
		# 1 update to clayrep
		self.clayreprates[rep] = self.clay_rep*self.get_count(rep)

		return self

class SystemRNARep(object):

	def __init__(self,L,n,population,R,reprate,hitting='R',hitting_V=None,power_fitness=6,power_similarity=3,power_rep=1,normalize_reprate=None,linear=False,string=''):

		self.L = L
		self.n = n
		self.count = Counter(population) # master counter
		self.counts = [copy.copy(self.count)] # copies of counters
		self.R = R 
		self.reprate = reprate 
		self.hitting = hitting 
		self.hitting_V = hitting_V 
		self.power_fitness = -np.log(power_fitness) / L
		self.power_similarity = -np.log(power_similarity) / L
		self.power_rep = -np.log(power_rep) / L
		self.s = 0
		self.T = [0]
		self.population = population
		self.string = string

		self.hamming_R = lambda x: np.sum(s not in y for s,y in zip(x,self.R))*1.0
		self.cache_h = {}
		self.cache_hh = {}
		self.cache_f = {}
		self.cache_fidel = {}
		self.cache_s = {}
		self.cache_ss = {}
		self.sym_similarity = lambda x,y: min(self.Hamming(x,y),self.Hamming(x,comp(y)),self.Hamming(comp(x),y),self.Hamming(comp(x),comp(y)))
		
		if not linear:
			self.power_fitness = -np.log(power_fitness) / L
			self.power_similarity = -np.log(power_similarity) / L
			self.power_rep = -np.log(power_rep) / L
			self.fitness_rep = lambda x: np.exp(-self.power_fitness*self.Hamming_R(x))
			self.fitness_fidelity = lambda x: np.exp(-self.power_rep*self.Hamming_R(x))
			self.similarity = lambda x,y: np.exp(-self.power_similarity*self.Sym_Similarity(x,y))
		else:
			self.power_fitness = power_fitness
			self.power_similarity = power_similarity
			self.power_rep = power_rep
			self.fitness_rep = lambda x: 1 + (1-self.power_fitness)/(-1*L)*self.Hamming_R(x)
			self.fitness_fidelity = lambda x: 1 + (1-self.power_rep)/(-1*L)*self.Hamming_R(x)
			self.similarity = lambda x,y: 1 + (1-self.power_similarity)/(-1*L)*self.Sym_Similarity(x,y)
		
		self.Pmf = {}

		self.random_genome = lambda : ''.join(rand.choice(nucleotides,size=self.L))
		self.get = lambda x: next(iter(x))

		# replication rates
		self.reprates = {
		    (kx,ky): np.abs(self.reprate*self.Fitness_rep(self.get(kx))*self.Similarity(self.get(kx),self.get(ky))*self.get_count(kx)*(self.get_count(ky)-(1 if kx==ky else 0))) 
		    for kx in self.count.keys() for ky in self.count.keys()
		}

		if normalize_reprate is not None:
			self.normalize_reprates(normalize_reprate)

	def normalize_reprates(self,val):
		self.reprate = val * self.reprate / np.sum(self.reprates.values())
		self.reprates = {
		    (kx,ky): np.abs(self.reprate*self.Fitness_rep(self.get(kx))*self.Similarity(self.get(kx),self.get(ky))*self.get_count(kx)*(self.get_count(ky)-(1 if kx==ky else 0))) 
		    for kx in self.count.keys() for ky in self.count.keys()
		}
		return self

	def Fitness_rep(self,x):
	    if x in self.cache_f:
	        return self.cache_f[x]
	    else:
	        self.cache_f[x] = self.fitness_rep(x) 
	        return self.cache_f[x]

	def Fitness_fidelity(self,x):
	    if x in self.cache_fidel:
	        return self.cache_fidel[x]
	    else:
	        self.cache_fidel[x] = self.fitness_fidelity(x) 
	        return self.cache_fidel[x]

	def Hamming_R(self,x):
		if x in self.cache_h:
		    return self.cache_h[x]
		else:
		    self.cache_h[x] = self.hamming_R(x) 
		    return self.cache_h[x]

	def Sym_Similarity(self,x,y):
		if (x,y) in self.cache_ss:
		    return self.cache_ss[(x,y)]
		else:
		    self.cache_ss[(x,y)] = self.cache_ss[(y,x)] = self.sym_similarity(x,y) 
		    return self.cache_ss[(x,y)]

	def Similarity(self,x,y):
		if (x,y) in self.cache_s:
		    return self.cache_s[(x,y)]
		else:
		    self.cache_s[(x,y)] = self.cache_s[(y,x)] = self.similarity(x,y) 
		    return self.cache_s[(x,y)]

	def Hamming(self,x,y):
		if (x,y) in self.cache_hh:
		    return self.cache_hh[(x,y)]
		else:
		    self.cache_hh[(x,y)] = self.cache_hh[(y,x)] = hamming(x,y) 
		    return self.cache_hh[(x,y)] 

	def get_count(self,x): 
		return self.count[x] if x in self.count else 0

	def get_count_i(self,x,i):
		return self.counts[i][x] if x in self.counts[i] else 0

	def hitting_Vol(self):
		total = np.sum(len(k)*v for k,v in self.count.iteritems())*1.0
		Rcontent = np.sum(v for k,v in self.count.iteritems() if len(k) == 1 and min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k)))) == 0)*1.0
		Rcontent += np.sum(v for k,v in self.count.iteritems() if len(k) == 2 and (self.Hamming_R(list(k)[0]) == 0 or self.Hamming_R(list(k)[1]) == 0))*1.0
		if total > 0 and Rcontent / total >= self.hitting_V:
		    return True
		else:
		    return False

	@classmethod
	def hitting_Vol_population(cls,population,R,hitting_V):
		Hamming_R = lambda x: np.sum(s not in y for s,y in zip(x,R))*1.0
		get = lambda x: next(iter(x))
		count = Counter(population)
		total = np.sum(len(k)*v for k,v in count.iteritems())*1.0
		Rcontent = np.sum(v for k,v in count.iteritems() if len(k) == 1 and min(Hamming_R(get(k)),Hamming_R(comp(get(k)))) == 0)*1.0
		Rcontent += np.sum(v for k,v in count.iteritems() if len(k) == 2 and (Hamming_R(list(k)[0]) == 0 or Hamming_R(list(k)[1]) == 0))*1.0
		if total > 0 and Rcontent / total >= hitting_V:
		    return True
		else:
		    return False

	@classmethod
	def hitting_R_population(cls,population,R):
		Hamming_R = lambda x: np.sum(s not in y for s,y in zip(x,R))*1.0
		get = lambda x: next(iter(x))
		count = Counter(population)
		Rcontent = np.sum(v for k,v in count.iteritems() if len(k) == 1 and min(Hamming_R(get(k)),Hamming_R(comp(get(k)))) == 0)*1.0
		Rcontent += np.sum(v for k,v in count.iteritems() if len(k) == 2 and (Hamming_R(list(k)[0]) == 0 or Hamming_R(list(k)[1]) == 0))*1.0
		if Rcontent > 0:
		    return True
		else:
		    return False

	def hitting_R(self):
		return any(min(self.Hamming_R(self.get(x)),self.Hamming_R(comp(self.get(x))))==0 for x in self.count.keys())

	def simulate_reactions(self,N):

	    if self.hitting == 'R':
	        hitting = self.hitting_R 
	    elif self.hitting == 'V':
	        hitting = self.hitting_Vol

	    if hitting():
	    	return self
	    else:
	    	for i in range(1,N+1):
	    		if hitting(): break
	    		print "Reaction", i
	    		self.next_reaction()
	    	return self

	def plot_concentration_curves(self):

		# get counts for genomes by distance across time		
		get_count_series = lambda counters,i: np.asarray(list(np.sum(v for k,v in counter.iteritems() if min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k))))==i)*1.0 
		for counter in counters))

		plt.figure()
		plt.title('Growth curves by distance')
		Series = []
		for i in range(self.L+1):
			series = get_count_series(self.counts,i)
			Series.append(series)
			plt.plot(np.log(1+np.asarray(self.T)),np.log(1+series),label=str(i))
		plt.legend()
		plt.xlabel('Log(1+time)')
		plt.ylabel('Log(1+count)')

		Series = np.asarray(Series)
		Series /= np.sum(Series,axis=0)
		plt.figure()
		plt.title('Concentration curves by distance')
		for i,series in enumerate(Series):
			plt.plot(np.log(1+np.asarray(self.T)),series,label=str(i))
		plt.legend()
		plt.xlabel('Log(1+time)')
		plt.ylabel('Concentration')

		return self

	def plot_distances(self,percentile=1.0):

		"""
		plot dendrogram for symmetric distances
		"""

		ind = max(int(percentile*len(self.counts))-1,0)
		genomes = np.asarray(list(self.get(k) for k,v in self.counts[ind].iteritems() for i in range(v)))

		# genome distances (symmetric)
		distances = np.asarray(list(list(self.Sym_Similarity(g1,g2) for g1 in genomes) for g2 in genomes))
		linkage = hc.linkage(spa.distance.squareform(distances),method='average')
		cg = sns.clustermap(distances, row_linkage=linkage, col_linkage=linkage)
		cg.ax_heatmap.set_yticklabels(labels = genomes)
		cg.ax_heatmap.set_xticklabels(labels = genomes)
		plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
		plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
		#plt.figure()
		#hc.dendrogram(linkage)
		return self

	def next_reaction(self):

		"""
		simulates one reaction
		"""

		rep_total = np.sum(self.reprates.values())

		# get the rates and time increment 
		r = rep_total 
		self.s += rand.exponential(scale=1/r)
		self.T.append(self.s)

		print self.string
		print self.s

		print "RNA-REP"
		p_reactions = [v/rep_total for v in self.reprates.values()]
		ind = rand.choice(np.arange(len(self.reprates)),p=p_reactions)
		key1,key2 = self.reprates.keys()[ind]
		self._rxn_rnarep(key1,key2)

		self.counts.append(copy.copy(self.count)) # for each reaction type, get the reaction

		total = np.sum(len(k)*v for k,v in self.count.iteritems())*1.0
		Rcontent = np.sum(v for k,v in self.count.iteritems() if min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k))))== 0)*1.0

		lst = []
		for k,v in self.count.iteritems():
			if len(k) == 1 and v > 0:
			    for i in range(v):
			        lst.append(min(self.Hamming_R(self.get(k)),self.Hamming_R(comp(self.get(k)))))
			elif len(k) == 2 and v > 0:
			    k1,k2 = list(k)
			    for i in range(v):
			        lst.append(min(self.Hamming_R(k1),self.Hamming_R(k2)))

		mindis = Counter(lst)

		print mindis, Rcontent, total, Rcontent/total if total > 0 else 0
		return self

	def pmf(self,d,L,p):
		if (d,L,p) in self.Pmf:
		    return self.Pmf[(d,L,p)]
		else:
		    self.Pmf[(d,L,p)] = stat.binom.pmf(d,L,1-p)
		    return self.Pmf[(d,L,p)]

	def get_time(self,percentile=1.0):
		ind = max(int(percentile*len(self.counts))-1,0)
		return self.T[ind]

	def measures(self,percentile=1.0):

		"""
		Generates the measure-kernel-functions nu, Q, and mu
		"""

		ind = max(int(percentile*len(self.counts))-1,0)
		#keys = self.count.keys()
		keys = self.counts[ind].keys()
		nu = np.zeros((1,len(keys)**2))
		inds = list(0.01*i*len(keys)**2 for i in range(1,101))
		dum = 1
		for i,(k1,k2) in enumerate(product(keys,keys)):
			nu[0,i] = self.reprate*self.Fitness_rep(self.get(k1))*self.Similarity(self.get(k1),self.get(k2))*self.get_count_i(k1,ind)*(self.get_count_i(k2,ind)-(1 if k1==k2 else 0))
			if i == int(dum*0.01*(len(keys)**2)):
				print "nu", i*1.0/(len(keys)**2)
				dum += 1
		nu /= nu.sum()

		genomes = list(''.join(x) for x in product(nucleotides,repeat=self.L))

		Q = np.zeros((nu.size,4**self.L))
		inds = list(0.01*i*nu.size**2 for i in range(1,101))
		dum = 1
		for i,(k1,k2) in enumerate(product(keys,keys)):
			p = self.Fitness_fidelity(self.get(k1))
			ck2 = comp(self.get(k2))
			if p < 1:
				distances = list(self.Hamming(ck2,g) for g in genomes)
				counter = Counter(distances)
				Q[i,:] = list(1.0/counter[d]*self.pmf(d,self.L,1-p) for d in distances)
			else:
				Q[i,:] = list(1 if g == ck2 else 0 for g in genomes)

			if i == int(dum*0.01*nu.size):
				print "Q", i*1.0/nu.size
				dum += 1

		# get probabilities
		prod = np.dot(nu,Q)

		distances = np.asarray(list(min(self.Hamming_R(g),self.Hamming_R(comp(g))) for g in genomes))
		distanceset = sorted(list(set(distances.tolist())))
		probs = []
		for j in distanceset:
			#inds = list(i for i,x in enumerate(distances) if x == j)
			inds = np.where(distances == j)
			probs.append(prod[0,inds].sum())
		print distanceset, probs
		return probs

	def _rxn_rnarep(self,key1,key2):

		prob = self.Fitness_fidelity(self.get(key1))#1-hamming(reference,next(iter(key1)))/len(next(iter(key1)))
		rep = frozenset([replicate(self.get(key2),p=prob)])
		self.count.update({rep:1})		

		fitrep = self.Fitness_rep(self.get(rep))
		# 2 * len(keys) updates to reprates
		for key in self.count.keys():
			sim = self.Similarity(self.get(rep),self.get(key))
			self.reprates[(rep,key)] = self.reprate*fitrep*sim*self.get_count(rep)*(self.get_count(key)-(1 if rep==key else 0))
			self.reprates[(key,rep)] = self.reprate*self.Fitness_rep(self.get(key))*sim*self.get_count(key)*(self.get_count(rep)-(1 if rep==key else 0))

		return self

