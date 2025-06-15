#from typing import override
import copy 

from jmetal.config import StoppingByEvaluations
from jmetal.core.problem import IntegerProblem, IntegerSolution
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.operator.selection import (
    BinaryTournamentSelection,
    RankingAndCrowdingDistanceSelection,
)
from jmetal.operator.mutation import Mutation
from jmetal.core.operator import Crossover
from jmetal.algorithm.multiobjective import NSGAII
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sys import argv
import random

from scipy.io import loadmat
import json

from KerasExecutor import KerasExecutor
from algorithm import eval_keras, Individual
from Operators import complete_mutation, complete_crossover


###############################################################################

class Configuration(object):
    def __init__(self, j):
        self.__dict__ = json.load(j)


###########################
def cifar10G_data_builder():

    cifar10G_path = "./cifar10G32.mat"
    cifar10G_raw = loadmat(cifar10G_path)
    dataset = {
        "data": cifar10G_raw["data"],
        "target": cifar10G_raw["target"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "",
    }

    return dataset
#############################

#############################
def fashion_mnist_data_builder():

    fashion_mnist_path = "./fashion_mnist.mat"
    fashion_mnist_raw = loadmat(fashion_mnist_path)
    dataset = {
        "data": fashion_mnist_raw["data"],
        "target": fashion_mnist_raw["target"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "",
    }

    return dataset
#############################

###############################################################################


def plot_pareto(front, fname="nas_nsga2_pareto.png"):
    fig, ax = plt.subplots()
    first_obj = [(-1 * f.objectives[0]) for f in front]
    second_obj = [f.objectives[1] for f in front]
    df = pd.DataFrame.from_dict({"accuracy": first_obj, "complexity": second_obj})
    sns.scatterplot(data=df, x="accuracy", y="complexity", ax=ax, s=50)
    sns.lineplot(data=df, x="accuracy", y="complexity", ax=ax)
    p = plt.gcf()
    p.savefig(fname)
    return


class NASProblem(IntegerProblem):

    def __init__(self, ke: KerasExecutor, config: Configuration):
        super(IntegerProblem, self).__init__()
        self.ke = ke
        self.config = config
        self.number_of_objectives = 2
        self.obj_labels = ["Accuracy", "Complexity"]
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]

    def name(self):
        return "NASProblem"

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return 2

    def number_of_variables(self) -> int:
        return super().number_of_variables()

    def evaluate(self, solution: IntegerSolution) -> list:
        fitness = eval_keras(solution.variables, self.ke)
        solution.objectives[0] = -1.0 * fitness[0] #Precisión en validación
        solution.objectives[1] = fitness[4] #Nº de parámetros

    def create_solution(self) -> IntegerSolution:
        new_ind = Individual(self.config, copy.deepcopy(self.ke.n_in), copy.deepcopy(self.ke.n_out))
        new_solution = IntegerSolution(
            lower_bound=np.zeros(3),
            upper_bound=np.ones(3),
            number_of_objectives=2
        )
        new_solution.variables = new_ind
        return new_solution


class mutationNAS(Mutation):
    def __init__(self, probability: float, config: Configuration):
        super().__init__(probability)
        self.config = config
        

    def get_name(self) -> str:
        return "MutationNAS"

    def execute(self, solution: IntegerSolution):

        if np.random.rand() > self.probability:
            return solution

        solution.variables = complete_mutation(solution.variables, self.probability, 0.3, self.config)
        return solution


class CrossoverNAS(Crossover):
    def __init__(self, probability: float, config: Configuration):
        super(CrossoverNAS, self).__init__(probability)
        self.config = config

    def execute(self, parents: list[IntegerSolution]):
        # Aplico la probabilidad
        if np.random.rand() > self.probability:
            return parents

        # Obtengo los padres
        p1 = parents[0].variables
        p2 = parents[1].variables
        offsprings = copy.deepcopy(parents)

        o1, o2 = complete_crossover(p1, p2, self.probability, self.config)

        offsprings[0].variables = o1
        offsprings[1].variables = o2

        return offsprings

    def get_name():
        return "CrossoverNAS"

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2


def main():
    print("Hello from ejemplo-nsga2!")

    if len(argv) == 1:
        seed = 56
    else:
        seed = int(argv[1])

    print(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    #Argumentos para el KerasExecutor
    EXPERIMENT = 3
    if EXPERIMENT == 3:
        dataset = cifar10G_data_builder()
        print("DATASET: CIFAR10-G")
    else:
        dataset = fashion_mnist_data_builder()
        print("DATASET: FASHIONMNIST")

    test_size = 0.2
    metrics = ["accuracy"]
    early_stopping_patience = 100
    loss = "categorical_crossentropy"
    ke = KerasExecutor(dataset, test_size, metrics, early_stopping_patience, loss)

    with open("parametersGenetic.json", 'r') as f:
        configuration = Configuration(f)


    problem = NASProblem(ke=ke, config=configuration)

    algorithm = NSGAII(
        # El problema nuestro NAS
        problem=problem,
        # Tamaño de población
        population_size=10,
        # Offsprting
        offspring_population_size=20,
        # Para con 10 iteraciones (se le pasa las evaluaciones)
        termination_criterion=StoppingByEvaluations(210),
        # Nuestro operador de mutacion
        mutation=mutationNAS(probability=0.3, config=configuration),
        # Nuestro operador de cruce
        crossover=CrossoverNAS(probability=1.0, config=configuration),
        # Selección del NAS
        # selection=RankingAndCrowdingDistanceSelection(max_population_size=10),
        selection=BinaryTournamentSelection(
             comparator=RankingAndCrowdingDistanceComparator()
        )
    )

    algorithm.run()
    front = algorithm.result()

    for f in front:
        print(f.objectives)

    print(front[0])
    plot_pareto(front)


if __name__ == "__main__":
    main()
