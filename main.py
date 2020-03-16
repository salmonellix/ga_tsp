
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]

init_pop = np.random.choice(range(10), 10, replace=False)
init_pop2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
P = 250
n = 0.8
pm = 0.2
Tmax = 1000


class Chromosome():
    def __init__(self,xlist,ylist,order):
        self.xlist = xlist
        self.ylist = ylist
        self.order = order
        self.distance = 0
        self.fitness = 0.0

    def cost(self):
        lenTour = len(self.order)
        new_x = []
        new_y = []
        for i in range(0,lenTour):
            o = self.order[i]
            new_x.append(self.xlist[o])
            new_y.append(self.ylist[o])
            j = i+1
            if j < lenTour:
                jo = self.order[j]
                self.distance = self.distance + np.sqrt(((self.xlist[o]-self.xlist[jo])**2)+((self.ylist[o]-self.ylist[jo])**2))

        # plt.plot(new_x,new_y,'-o')
        # order_list = np.arange(0, 10, 1).tolist()
        # for x, y, z in zip(new_x, new_y,order_list ):
        #     plt.text(x, y, str(z), color="red", fontsize=12)
        # plt.show()
        print(self.distance)
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.cost())
        return self.fitness





def createChromosome():
    route = np.random.choice(range(10), 10, replace=False)
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createChromosome())
    return population

def rankRoutes(x,y,population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Chromosome(x,y,population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

chrr = Chromosome(x,y, init_pop2)
chrr.cost()
ourPop = initialPopulation(P, init_pop2)
print(ourPop)
bestRoutes = rankRoutes(x,y,ourPop)

print (bestRoutes)
