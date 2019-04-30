# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:28:18 2019

@author: UX430
"""



import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import random
 
PV = np.array([0, 0, 0, 0, 0, 0, 0, 0, 6, 10, 15, 20, 30, 40, 40, 20, 15, 10, 8, 2, 0, 0, 0, 0])
WT = np.array([51, 51, 58, 51, 64, 51, 44, 51, 44, 51, 51, 46, 81, 74, 65, 65, 65, 51, 39, 63, 38, 66, 74, 74])
DM = np.array([67, 67, 90, 114, 120, 130, 150, 190, 200, 206, 227, 227, 250, 250, 200, 180, 160, 160, 190, 150, 100, 50, 20, 20])
P_dem = DM - PV - WT
# Batería
SOC_ini = 140
P_BT_max = 120
SOC_BT_max = 280
SOC_BT_min = 70
eta_c = 0.9
eta_d = 0.9
# DE
P_DE_min = 5
P_DE_max = 80
# Mt
P_MT_min = 10 #20
P_MT_max = 140 #210
# Robusto
sigma = 3
 
def crea_individuo():
# Creación de individuos iniciales
    individuo = np.zeros(48)
    for i in range(0,24):
        individuo[i] = random.uniform(P_DE_min,min(P_dem[i],P_DE_max))
        if P_dem[i] < 0:
            individuo[i] = 0
        individuo[24+i] = P_dem[i] - individuo[i]
        if individuo[24+i] < P_MT_min:
            individuo[24+i] = 0
        if individuo[24+i] > P_MT_max:
            individuo[24+i] = P_MT_max
    return individuo
   
def mutacion (individuo, indpb):
# Mutación
    for j, i in enumerate(individuo):
        if random.random() < indpb[0]:
            individuo[j] = random.gauss(individuo[j],30)
        if random.random() < indpb[1]:
            individuo[j] = 0
        if random.random() < indpb[2]:
            individuo[j] = 0.05
    return individuo,
 
def fitness_function_single_cost(individual):
    # Datos, Coste y penalización
    coste = 0
    penalizacion = 10000000
    
    P_BT = P_dem - individual[:24] - individual[24:] # P_dem - P_WT - P_PV - P_MT - P_DE
    
    # Penalizaciones
    # Si P_BT > 0 la batería entrega potencia a la red. Si negativo se carga.
    if any(P_BT < -P_BT_max):
        return penalizacion+1,
    if any(P_BT > P_BT_max):
        return penalizacion+2,
    # Si MT o DE da menos potencia que la potencia mínima y no es cero 
    if any(individual[np.nonzero(individual[:24])[0]] < P_DE_min):
        return penalizacion+3,
    if any(individual[np.nonzero(individual[24:])[0]] < P_MT_min):
        return penalizacion+4,
    # Si MT o DE da más potencia que la potencia máxima y no es cero   
    if any(individual[np.nonzero(individual[:24])[0]] > P_DE_max):
        return penalizacion+5,
    if any(individual[np.nonzero(individual[24:])[0]] > P_MT_max):
        return penalizacion+6,
    # Se comprueba que el estado de carga de la batería este dentro de los límites
    SOC = Battery_evolution(P_BT, SOC_ini)
    if any(SOC > SOC_BT_max):
        return penalizacion+7,
    if any(SOC < SOC_BT_min):
        return penalizacion+8,
        
    # Coste de emisiones
    for i in range(0,47):
        if i < 24:
            coste += enviro_DE(individual[i])
        else:
            coste += enviro_MT(individual[i])
    coste += sum(individual[:24]*0.01258 + individual[24:]*0.00587)
        
    # Se devuelve el coste total
    return coste,  

def enviro_DE(P):
    alpha_DE = [0.0275, 1.9475, 8.2625]
    beta_DE = [0.6495, 0.2059, 9.8883]
    cost = 0
    for i in range(0,3):
        cost += alpha_DE[i]*beta_DE[i]*P
    return cost
    
def enviro_MT(P):
    alpha_MT = [0.0275, 1.9475, 8.2625]
    beta_MT = [0.7239, 0.0036, 0.1995]
    cost = 0
    for i in range(0,3):
        cost += alpha_MT[i]*beta_MT[i]*P
    return cost
 
def Battery_evolution(BT, SOC_ini):
    SOC = np.zeros(24)
    SOC[0] = SOC_ini
    for i in range(1,SOC.size):
        if BT[i] > 0:
            SOC[i] = SOC[i-1] - BT[i]/eta_d
        else:
            SOC[i] = SOC[i-1] - BT[i]*eta_c  
    return SOC
 

   
def configura(eta):       
    # paso1: creación del problema
    creator.create("Problema1", base.Fitness, weights=(-1,))
    # paso2: creación del individuo
    creator.create("Individual", np.ndarray, fitness = creator.Problema1)
 
    toolbox = base.Toolbox() # creamos la caja de herramientas
    # Registramos nuevas funciones
    toolbox.register("individual", tools.initIterate, creator.Individual, crea_individuo)
    toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individual)
 
    # Operaciones genéticas
    toolbox.register("evaluate", fitness_function_single_cost)
           
    toolbox.register("mate", tools.cxSimulatedBinary, eta= eta)
    toolbox.register("mutate", mutacion, indpb=(0.05,0.05,0.05))
    toolbox.register("select", tools.selTournament, tournsize = 3)
 
    return toolbox
 
def unico_objetivo_ga(c, m, i, toolbox):
    """ los parámetros de entrada son la probabilidad de cruce, la probabilidad
    de mutación y el número iteración
    """
    NGEN = 1000
    MU = 3000 # aumentar
    LAMBDA = MU # aumentar
    CXPB = c
    MUTPB = m
    random.seed(i) # actualizamos la semilla cada vez que hacemos una simulación
   
    pop = toolbox.ini_poblacion(n = MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
 
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    logbook = tools.Logbook()
   
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats= stats, halloffame=hof, verbose = False)
   
    return pop, hof, logbook
 
def plot(log, ind, PV, WT, DM):
    width = 0.7
    DE = ind[:24]
    MT = ind[24:]
    BT = DM - PV - WT - DE - MT
    BTpos = (BT>0)*BT
    BTneg = (BT<0)*BT
   
    # Resultados

    p1 = plt.bar(np.arange(24), PV, width, color = 'r')
    p2 = plt.bar(np.arange(24), WT, width, bottom = PV, color = 'b')
    p3 = plt.bar(np.arange(24), DE, width, bottom = PV+WT, color = 'k')
    p4 = plt.bar(np.arange(24), MT, width, bottom = PV+WT+DE, color = 'c')
    p5 = plt.plot(np.arange(24), DM, '*--')
    p6 = plt.bar(np.arange(24), BTpos, width, bottom = PV+WT+DE+MT, color = 'y')
    plt.bar(np.arange(24), BTneg, width, color = 'y')
    plt.grid()
    plt.ylabel('Energy (kWh)')
    plt.title('Unit commitment')
    plt.legend((p1[0], p2[0], p3[0], p4[0], p6[0], p5[0]), ('PV', 'WT', 'DE', 'MT', 'Battery', 'Demand'))
    plt.show()
    plt.savefig("solution.eps", dpi = 100)
       
    # Evolución de la batería
   
    fig, ax1 = plt.subplots()
   
    SOC = Battery_evolution(BT, SOC_ini)
    p1 = ax1.plot(np.arange(24), SOC_BT_min*np.ones(24), 'k')
    p2 = ax1.plot(np.arange(24), SOC_BT_max*np.ones(24), 'k')
    p3 = ax1.bar(np.arange(24), SOC - SOC_BT_min, width, bottom = SOC_BT_min)
    ax2 = ax1.twinx()
    ax1.set_ylim(bottom = SOC_BT_min - 10, top = SOC_BT_max + 10)
    ax2.set_ylim(bottom = -P_BT_max, top = P_BT_max)
    p4 = ax2.plot(np.arange(24), BT, 'r*--')
    plt.grid(True)
    plt.ylabel('Energy (kWh)')
    plt.title('Battery management')
    plt.show()
    fig.savefig("Battery.eps", dpi = 100)
   
    # Evolución del algoritmo
   
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")
 
    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([min(fit_mins), 700])
    plt.grid(True)
    plt.show()
    fig.savefig('plot.eps')
 
    
    
if __name__ == "__main__":   
    parameters= [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)] # probabilidades que quiero probar
    for c, m in parameters:
        eta = 10
        for i in range(0, 30):
            toolbox = configura(eta)
            res_individuos = open("individuos.txt", "a")
            res_fitness = open("fitness.txt", "a")
            pop_new, pareto_new, log = unico_objetivo_ga(c, m, int(i), toolbox)
            for ide, ind in enumerate(pareto_new):
                res_individuos.write(str(i))
                res_individuos.write(",")
                res_individuos.write(str(ind))
                res_individuos.write("\n")
                res_fitness.write(str(i))
                res_fitness.write(",")
                res_fitness.write(str(eta))
                res_fitness.write(",")
                res_fitness.write(str(c))
                res_fitness.write(",")
                res_fitness.write(str(m))
                res_fitness.write(",")
                res_fitness.write(str(ind.fitness.values[0]))
                print(ind.fitness.values[0])
                res_fitness.write("\n")
            del(pop_new)
            del(pareto_new)
            res_fitness.close()
            res_individuos.close()