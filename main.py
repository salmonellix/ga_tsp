
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]

init_pop = np.random.choice(range(10), 10, replace=False)
init_chr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
        self.new_x = []
        self.new_y =[]

    # def showTour(self):
    #     plt.plot(self.xlist,self.ylist,'-o')
    #     order_list = self.order
    #     print(order_list)
    #     for x, y, z in zip(self.new_x, self.new_y, order_list):
    #         plt.text(x, y, str(z), color="red", fontsize=40)
    #     plt.show()


    def xyOrder(self):
        lenTour = len(self.order)
        for i in range(0, lenTour):
            o = self.order[i]
            self.new_x.append(self.xlist[o])
            self.new_y.append(self.ylist[o])
        return self.new_x, self.new_y

    def cost(self):
        for i in range(0, len(self.order)):
                o = self.order[i]
                j = i + 1
                if j < len(self.order):
                    jo = self.order[j]
                    self.distance = self.distance + np.sqrt(((self.xlist[o]-self.xlist[jo])**2)+((self.ylist[o]-self.ylist[jo])**2))


        #print(self.distance)
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

def populationCumSum(x,y,population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Chromosome(x,y,population[i]).routeFitness()
    fitnessResults = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    df_fitResults = pd.DataFrame(fitnessResults, columns = ["Chr_nbr", "Fitness"])
    #df_fitResults = df_fitResults.transpose()
    cumDF = df_fitResults.cumsum()
    cumDF["Chr_nbr"] = df_fitResults["Chr_nbr"]
    return cumDF

def rouletteSele(df_cum, population):
    ts = df_cum["Fitness"].iloc[-1]
    r = random.uniform(0,ts)   #random number between 0 and cumulated fitness
    chr_id = 0
    for i in range(0, len(df_cum["Fitness"])):
        if df_cum["Fitness"][i] > r:
            chr_id = df_cum["Chr_nbr"][i]
    seleChrmsm = population[chr_id]
    return seleChrmsm


chrr = Chromosome(x,y, init_chr)
ourPop = initialPopulation(P, init_chr)
bestRoutes = populationCumSum(x,y,ourPop)
select_chroms = rouletteSele(bestRoutes,ourPop)
print (select_chroms)
