from originoflifecorenew import System, nucleotides
from numpy.random import RandomState
import matplotlib.pyplot as plt
import sys
from sobol_goodies import safezip
import pickle
from itertools import product
import numpy as np

sys.setrecursionlimit(10000)
# python -m cProfile -o cprofile.dat originoflifeoo.py
# python -c 'import pstats; pstats.Stats("cprofile.dat").sort_stats("cumulative").print_stats(30)'


rand = RandomState()

reactions = 5000
initial = 10
#ppower_fitness = 6
#ppower_similarity = 6
#ppower_rep = 3

rds=1.0
rss=1.0
reprate=1.0
clay_olig=0.0
clay_rep=0.0
rdecay=0.0
pclay = 0.9
p_4 = 0.0
power_rep = 0.25

# define functions for intializing the initial population and high-fidelity manifold
random_genome = lambda L: ''.join(rand.choice(nucleotides,size=L))
randomR = lambda L,p_4: list(set(rand.choice(nucleotides,replace=False,size=rand.choice([1,4],p=[1-p_4,p_4]))) for i in range(L))


dimensions = [3,4,5,6,7,8,9,10]
power_fitnesses = power_similarities = list(10**(-i) for i in range(1,8+1))

#rdecays = [0.0,1.0]
#clay_reps = [0.0,1.0]
#pclays = [0.7,1.0]

def random_theta():
	dimension = rand.choice(dimensions)
	power_fitness = rand.choice(power_fitnesses)
	return (dimension, power_fitness)
#gen = product(dimensions,power_fitnesses,rdecays,clay_reps,pclays)

parameters = []
rxns = []
sims = 30
for j in range(1,sims+1):
	
	L, power_fitness = random_theta()
	
	R = randomR(L,p_4)
	Initial = list(frozenset([random_genome(L)]) for i in range(initial))
	#while System.hitting_Vol_population(Initial,R,0.10):
	#	Initial = list(frozenset([random_genome(L)]) for i in range(initial))
	while System.hitting_R_population(Initial,R):
		Initial = list(frozenset([random_genome(L)]) for i in range(initial))

	for i in range(1,10+1):
		system = System(
			L=L,
			n=reactions,
			population=Initial,
			R=R,
			rds=rds,
			rss=rss,
			reprate=reprate,
			rdecay=rdecay,
			clay_rep=clay_rep,
			clay_olig=clay_olig,
			pclay=pclay,
			hitting='R',
			#hitting_V=0.10,
			power_fitness=power_fitness,
			power_similarity=power_fitness,
			power_rep=power_rep,
			normalize_reprate=10.0,
			normalize_clay_reprate=None,
			string=str(j)+" " + str(i)
		)
		system.simulate_reactions(N=reactions)
		rxns.append(len(system.T)-1)
		T = system.T[-1]
		parameters.append((L,power_fitness,len(system.T),T,system.hitting_R()))

#system.generate_odes()
#print np.sum(rxns), np.mean(rxns), np.std(rxns)
# save data in dictionary

data = {}
data['parameters'] = parameters

# pickle the dictionary
with open('hittings11d.pickle', 'wb') as handle:
    pickle.dump(data, handle)

#system.simulate_reactions(N=reactions)
#system.plot_concentration_curves()
#system.plot_distances(percentile=0.1)
#system.plot_distances(percentile=0.5)
#system.measures(percentile=0.1)
#system.plot_distances(percentile=1.0)
#system.measures(percentile=1.0)

#plt.show(block=True)

