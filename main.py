
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]

init_pop = np.random.choice(range(10), 10, replace=False)
init_chr = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    fitnessResults = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)
    df_fitResults = pd.DataFrame(fitnessResults, columns = ["Chr_nbr", "Fitness"])
    #df_fitResults = df_fitResults.transpose()
    cumDF = df_fitResults.cumsum()
    cumDF["Chr_nbr"] = df_fitResults["Chr_nbr"]
    return cumDF #zwraca całą df

def rouletteSele(df_cum, population):
    ts = df_cum["Fitness"].iloc[-1]
    r = random.uniform(0,ts)   #random number between 0 and cumulated fitness
    # print(r)
    # print("............")
    chr_id = 0
    for i in range(0, len(df_cum["Fitness"])):
        if df_cum["Fitness"][i] > r:
            # print(df_cum["Fitness"][i] )
            chr_id = df_cum["Chr_nbr"][i]
            break
    seleChrmsm = population[chr_id]
    return seleChrmsm.tolist()    # return chromosome choosen by roulette wheal

# crossover

def crossover(P1,P2):
    O1 = [22 for i in range(0,len(P1))]
    O1[0] = P1[0]
    p2_ele = P2[0]
    position = P1.index(p2_ele)
    for i in range(1, len(P1)):
        O1[position] = p2_ele
        p2_ele = P2[position]
        position = P1.index(p2_ele)
        if p2_ele in O1:
            break
    for j in range(0, len(P1)):
        if O1[j] == 22:
            O1[j] = P2[j]
    return O1

def mutation(offspring):
    nbr_list = [i for i in range(0, 10)]
    nbr_list = random.sample(nbr_list, len(nbr_list))
    point1 = nbr_list[0]
    point2 = nbr_list[1]
    tmp = offspring[point1]
    offspring[point1] = offspring[point2]
    offspring[point2] = tmp
    return offspring

def showTour(x,y,tour):
    new_x = []
    new_y = []
    for i in tour:
        new_x.append(x[i])
        new_y.append(y[i])
    print(" new x : %s", new_x)
    print(" new y : %s", new_y)
    print(tour)
    plt.plot(new_x,new_y, '-o')
    for x, y, z in zip(new_x,new_y, tour):
        plt.text(x, y, str(z), color="red", fontsize=14)
    plt.show()


chrr = Chromosome(x,y, init_chr)
ourPop = initialPopulation(P, init_chr)
bestRoutes = populationCumSum(x,y,ourPop)
select_chroms = rouletteSele(bestRoutes,ourPop)
best_chr = Chromosome(x,y, select_chroms)

P1 = rouletteSele(bestRoutes,ourPop)
P2 = rouletteSele(bestRoutes, ourPop)
# print(P1)
# # print(P2)
# # print("|||||||||||||")
O1 = crossover(P1,P2)
print(O1)
# # O2 = crossover(P2,P1)
# #
# # print(O1)
print("-------------")
# # print(O2)

print(mutation(O1))
showTour(x,y,mutation(O1))