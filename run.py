# coding=utf-8
import argparse
import json
import random
import urllib.request
from scipy.io import loadmat
import time
from greenery.fsm import fsm

import multiprocessing
import os
import itertools
import numpy
import yaml
from copy import deepcopy
from deap import creator, base
from deap import tools
from KerasExecutor import KerasExecutor
from Operators import complete_crossover, complete_mutation
from algorithm import eval_keras, compare_individuals, Individual, eaMuPlusLambdaModified, dummy_eval, sel_accvalParams

import scipy
#from PIL import Image
import keras as keras
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

MAX_GENERATIONS_NO_CHANGES = 5

"""
EXPERIMENTS:
- 0: MNIST
- 1: HORSEHUMAN-G
- 2: VANGOGH-G
- 3: CIFAR10-G
- 4: FASHION_MNIST
"""

EXPERIMENT = 3

#############################################
# EXTRA CLASSES DEFINITION
#############################################
class Configuration(object):
    def __init__(self, j):
        self.__dict__ = json.load(j)


last_time = 0
counter_generations = 0
metrics = ["accuracy"]
early_stopping_patience = 100
loss = "categorical_crossentropy"


def mnist_data_builder():
    # Loading MNIST dataset
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"

    with urllib.request.urlopen(mnist_alternative_url) as url:
        content = url.read()
        # I'm guessing this would output the html source code ?
        with open(mnist_path, "wb") as f:
            print("Downloading data")
            f.write(content)

    # if not os.path.isfile(mnist_path):
    #     response = urllib.urlopen(mnist_alternative_url)
    #     with open(mnist_path, "wb") as f:
    #         print("Downloading data")
    #         content = response.read()
    #         f.write(content)
    mnist_raw = loadmat(mnist_path)
    dataset = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }

    return dataset


###########################
def horsehumanG_data_builder():

    horsehumanG_path = "./horsehumanG.mat"
    horsehumanG_raw = loadmat(horsehumanG_path)
    dataset = {
        "data": horsehumanG_raw["data"],
        "target": horsehumanG_raw["target"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "",
    }

    return dataset
###########################


###########################
def vangoghG_data_builder():

    vangoghG_path = "./vangoghG.mat"
    vangoghG_raw = loadmat(vangoghG_path)
    dataset = {
        "data": vangoghG_raw["data"],
        "target": vangoghG_raw["target"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "",
    }

    return dataset
###########################


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


def timing(_):
    global last_time
    if last_time == 0:
        last_time = time.time()
        return 0
    else:

        global counter_generations

        counter_generations += 1
        result = time.time() - last_time
        last_time = time.time()
        return round(result, 2)


def get_string_parameters():

    output_string = ""
    output_string += "Dataset: " + "CIFAR10-G" + "\n"
    output_string += "Test size: " + str(test_size) + "\n"
    output_string += "Early stopping patience: " + str(early_stopping_patience) + "\n"
    output_string += "Loss: " + loss + "\n"
    output_string += "Population size: " + str(args.lamb) + "\n"
    output_string += "LAMBDA: " + str(args.lamb) + "\n"
    output_string += "MU: " + str(args.mu) + "\n"
    output_string += "Execution id: " + str(execution_id) + "\n"
    output_string += "Num generations: " + str(args.ngen) + "\n"
    output_string += "Prob cross: " + str(args.cxpb) + "\n"
    output_string += "Prob mut: " + str(args.mutpb) + "\n"
    output_string += "prob_add_layer: " + str(args.newpb) + "\n"
    output_string += "MAX_GENERATIONS_NO_CHANGES: " + str(MAX_GENERATIONS_NO_CHANGES) + "\n"
    return output_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute a mu,lambda evolutionary algorithm')
    parser.add_argument('paramFile', type=argparse.FileType('r'))
    parser.add_argument('--mu', dest='mu', default=5, type=int)
    parser.add_argument('--lambda', dest='lamb', default=10, type=int)
    parser.add_argument('--cxpb', dest='cxpb', default=0.3, type=float,
                        help='Crossover probability')
    parser.add_argument('--mutpb', dest='mutpb', default=0.3, type=float,
                        help='Mutation probability')
    parser.add_argument('--newpb', dest='newpb', default=0.3, type=float,
                        help='Probability of adding a new layer')
    parser.add_argument('--ngen', dest='ngen', default=10, type=int,
                        help='Number of generations')
    parser.add_argument('--seed', dest='seed', default=1652, type=int,
                        help='Random seed')
    parser.add_argument('--test-size', dest='test_size', default=0.2, type=float,
                        help='Ratio of observations for testing')
    parser.add_argument('--parallel', action='store_true',
                        help='Multi-threaded execution')
    parser.add_argument('--threshold', dest='threshold', default=0.0, type=float,
                        help='Threshold for the fitness function')
    args = parser.parse_args()

    if args.seed != 1652:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    dataset = None
    if EXPERIMENT == 0:
        print( "EXPERIMENT WITH MNIST DATASET")
        dataset = mnist_data_builder()
    elif EXPERIMENT == 1:
        print ("EXPERIMENT WITH HORSEHUMAN-G DATASET")
        dataset = horsehumanG_data_builder()
    elif EXPERIMENT == 2:
        print ("EXPERIMENT WITH VANGOGH-G DATASET")
        dataset = vangoghG_data_builder()
    elif EXPERIMENT == 3:
        print ("EXPERIMENT WITH CIFAR10-G DATASET")
        dataset = cifar10G_data_builder()
    elif EXPERIMENT == 4:
        print ("EXPERIMENT WITH FASHION_MNIST DATASET")
        dataset = fashion_mnist_data_builder()
    else:
        print("EXPERIMENT NOT IMPLEMENTED!")

    execution_id = time.time()

    print("EXECUTION ID: " + str(execution_id))
    print("SEED: " + str(args.seed))

    # Loading parameters file
    configuration = Configuration(args.paramFile)
    test_size = args.test_size

    # Starting keras module
    ke = KerasExecutor(dataset, test_size, metrics, early_stopping_patience, loss)

    # Creating fitness function and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", Individual, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()


    #####################

    # Defining genetic search
    toolbox.register("individual", creator.Individual, config=configuration,
                     n_global_in=deepcopy(ke.n_in),
                     n_global_out=deepcopy(ke.n_out))

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Defining operators...
    #toolbox.register("evaluate", dummy_eval)
    toolbox.register("evaluate", eval_keras, ke=deepcopy(ke))

    toolbox.register("mate", complete_crossover, indpb=0.5, config=configuration)
    toolbox.register("mutate", complete_mutation, indpb=0.5,
                     prob_add_remove_layer=args.newpb, config=configuration)
    ###toolbox.register("select", tools.selBest)
    toolbox.register("select", sel_accvalParams)

    print("Building population...")
    population = toolbox.population(n=args.lamb)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    stats.register("time", timing)
    hof = tools.HallOfFame(args.lamb, similar=compare_individuals)


    # Parallelization (individuals are evaluated in parallel)
    if args.parallel:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    # Running genetic algorithm
    pop, logbook, my_logbook, contadoresNF = eaMuPlusLambdaModified(population=population, toolbox=toolbox, mu=args.mu,
                                          lambda_=args.lamb, cxpb=args.cxpb, mutpb=args.mutpb, ngen=args.ngen, 
                                          threshold=args.threshold, stats=stats, halloffame=hof)
    print('-------> TERMINADO')
    csv_accuracy = "accuracy_training, accuracy_validation, accuracy_test\n"
    csv_accuracy = "accuracy_training, accuracy_validation, accuracy_test\n"
    file_individuals = ""
    file_individuals += get_string_parameters()

    # Generating results
    for index, individual in enumerate(hof):
        accuracy_validation, number_layers, accuracy_training, accuracy_test, _ = individual.my_fitness

        file_individuals += "New individual: " + str(index) + "\n"
        file_individuals += "ACC_TRAINING {} ACC_VALIDATION {} ACC_TEST {} NUMBER_LAYERS {}\n\n".format(
                accuracy_training, accuracy_validation, accuracy_test, number_layers)
        file_individuals += individual.toString() + "\n"
        csv_accuracy += str(accuracy_training) + "," + str(accuracy_validation) + "," + str(accuracy_test) + "\n"

    file_output_csv_accuracy = open("accuracy_list_" + str(execution_id) + ".csv", "w")
    file_output_csv_accuracy.write(csv_accuracy)
    file_output_csv_accuracy.close()

    file_output_individuals = open("individuals_list_" + str(execution_id) + ".txt", "w")
    file_output_individuals.write(file_individuals)
    file_output_individuals.close()

    if args.parallel:
        pool.close()

    with open('individual' + str(execution_id) + '.yml', 'w') as outfile:
        yaml.dump(hof, outfile)

    with open('statistics' + str(execution_id) + '.txt', 'w') as outfile:
        outfile.write(str(logbook))
    
    with open('my_statistics' + str(execution_id) + '.csv', 'w') as outfile:
        outfile.write(str(my_logbook))
        outfile.write("\nNº selecciones por precision: " + str(contadoresNF[0]))
        outfile.write("\nNº selecciones por precision y nº parametros: " + str(contadoresNF[1]))

    print("EXECUTION FINISHED, ID: " + str(execution_id))
    print("SEED: " + str(args.seed))
    print("MAX_GENERATIONS_NO_CHANGES: " + str(MAX_GENERATIONS_NO_CHANGES))
    print("THRESHOLD: " + str(args.threshold))
    print("MU :" + str(args.mu))
