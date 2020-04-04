
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt, time, sys


start = time.time()

#  Get x, y cords from file
map_file = open(str(sys.argv[1]), "r")
lines = map_file.readlines()
x_start = lines[1].index("[")
x_end = lines[1].index("]")
x = list(map(int, lines[1][x_start+1:x_end].split(" ")))
y_start = lines[2].index("[")
y_end = lines[2].index("]")
y = list(map(int, lines[2][y_start+1:y_end].split(" ")))
# get other params
P = int(sys.argv[2])
n = float(sys.argv[3])
pm = float(sys.argv[4])
Tmax = int(sys.argv[5])

init_chr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# P = 500
# n = 0.9
# pm = 0.3
# Tmax = 10
# x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
# y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]


print("       Traveling Salesman Problem         ")
print("x cordinates: ", x)
print("y cordinates: ", y)
print("P: {}, n: {}, pm: {}, Tmax: {} \n".format(P, n, pm, Tmax))


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
        # print(self.distance)


        #print(self.distance)
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.cost())
        return self.fitness


# chm = Chromosome(x,y,ch1)
# chm.cost()


def createChromosome():
    route = np.random.choice(range(10), 10, replace=False)
    return route

def initialPopulation(popSize, cityList):   # create group of random chromosomes depends on P parameter
    population = []

    for i in range(0, popSize):
        population.append(createChromosome())
    return population

def populationCumSum(x,y,population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Chromosome(x,y,population[i]).routeFitness()
    fitnessResults = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)
    df_fitResults = pd.DataFrame(fitnessResults, columns = ["Chrms", "Fitness"])
    cumDF = df_fitResults.cumsum()
    cumDF["Chrms"] = df_fitResults["Chrms"]
    return cumDF #zwraca całą df

def rouletteSele(df_cum, population):
    ts = df_cum["Fitness"].iloc[-1]
    r = random.uniform(0,ts)   #random number between 0 and cumulated fitness
    chr_id = 0
    for i in range(0, len(df_cum["Fitness"])):
        if df_cum["Fitness"][i] > r:
            chr_id = df_cum["Chrms"][i]
            break
    seleChrmsm = population[chr_id]
    try:
        seleChrmsm = seleChrmsm.tolist()
    except:
        pass

    return seleChrmsm    # return chromosome choosen by roulette wheal

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
    nbr_list = [j for j in range(0, 10)]
    nbr_list = random.sample(nbr_list, len(nbr_list))
    point1 = nbr_list[0]
    point2 = nbr_list[1]
    tmp = offspring[point1]
    offspring[point1] = offspring[point2]
    offspring[point2] = tmp
    return offspring

def showTour(x,y,tour, distance):
    new_x = []
    new_y = []
    print("Cities order: {}".format(tour))
    for k in tour:
        new_x.append(x[k])
        new_y.append(y[k])
    plt.plot(new_x,new_y, '-o')
    for x, y, z in zip(new_x,new_y, tour):
        plt.text(x, y, str(z), color="red", fontsize=14)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Calculated path - distance: {}".format(distance))
    plt.show()


 #  MAIN LOOP
n_time = 0                                       # count loop start with 0
current_population = initialPopulation(P, init_chr)  # step 1 - create initial population (P chromosomes)
# print(current_population)
n_parents = n*P    # nbr of parents to create offsprings
n_parents  = np.int(np.floor(n_parents))
df_all = pd.DataFrame({"Fitness": [], "Chrms": []})
while n_time < Tmax:
    # print(n_time)
    nP_current = []
    offsprings = []
    pop_cum_fit = populationCumSum(x, y, current_population) # create data frame with cumulative fitness values
    for i in range(0, n_parents):           # Choose n*P parents from the current population
        sele_parent = rouletteSele(pop_cum_fit, current_population)
        nP_current.append(sele_parent)

    for i in range(0, n_parents):
        P1 = current_population[random.randrange(n_parents)] # Select randomly two parents to create offspring using crossover operator
        P2 = current_population[random.randrange(n_parents)]             # Repeat the Step 4 until n  P ospring are generated
        try:
            O1 = crossover(P1.tolist(),P2.tolist())
        except:
            O1 = crossover(P1, P2)
        offsprings.append(O1)

    for i in range(0,n_parents):
        random_O1 = random.choice(offsprings)               # Apply mutation operators for changes in randomly selected offspring
        position_O1 = offsprings.index(random_O1)
        offsprings[position_O1] = mutation(random_O1)    # podmienia wszystkie offspring --> lista z sekwencjami

    # create dataframe from the combined population of parents and offspring:plt.xlabel('time (s)')
    # plt.ylabel('volts (mV)')
    fit = []
    for i in range(0, len(offsprings)):
        fit.append(Chromosome(x,y,offsprings[i]).routeFitness())
    pop_list=[]
    pop_fit_list = pop_cum_fit["Fitness"].tolist()
    for i in range(0, len(current_population)):
        nbr  = [int(pop_cum_fit["Chrms"][i])][0]
        population_ele= current_population[nbr]
        try:
            pop_list.append(population_ele.tolist())
        except:
            pop_list.append(population_ele)
    pop_list.extend(offsprings)
    pop_fit_list.extend(list(fit))
    df_all = pd.DataFrame({ "Chrms": pop_list, "Fitness": pop_fit_list})
    df_all=df_all.sort_values(by=['Fitness'],ascending=False)
    for i in range(0, len(current_population)):
        current_population[i] = df_all["Chrms"][i]
    n_time+=1
end = time.time()

# print info:
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
print("Calculation Time: {}\n".format(end - start))
best_route = current_population[-1]
chm = Chromosome(x,y,best_route)
print("Distance:{}".format(chm.cost()))
showTour(x,y,best_route, chm.cost())



