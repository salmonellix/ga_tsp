
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


x = [0,3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]

init_pop = np.random.choice(range(10), 10, replace=False)
P = 250
n = 0.8
pm = 0.2
Tmax = 1000

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Chromosome():
    def __init__(self,xlist,ylist,order):
        self.xlist = xlist
        self.ylist = ylist
        self.order = order
        self.distance = 0

    def cost(self):
        lenTour = len(self.order)
        new_x = []
        new_y = []
        for i in range(0,lenTour):
            new_x.append(self.xlist[self.order[i]])
            new_y.append(self.ylist[self.order[i]])
            j=i+1
            if j<lenTour:
                self.distance = self.distance + np.sqrt(((self.xlist[i]-self.xlist[j])**2)+((self.ylist[i]-self.ylist[j])**2))

        plt.plot(new_x,new_y,'-o')
        order_list = np.arange(0, 10, 1).tolist()
        for x, y, z in zip(new_x, new_y,order_list ):
            plt.text(x, y, str(z), color="red", fontsize=12)
        plt.show()

        return self.distance

    def routeFitness(self):
            if self.fitness == 0:
                self.fitness = 1 - float(self.cost())
            return self.fitness





def createChromosome(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createChromosome(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Chromosome(population[i]).cost()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

chrr= Chromosome(x,y,order1)
chrr.cost()


class TourCost:
    def __init__(self, tour):
        self.tour = tour
        self.distance = 0
        self.fitness= 0.0

