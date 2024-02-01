import torch
from multiprocessing import Pool


def function_execution(args):
    # process the argument
    id_param, param, experiment_function = args
    print(id_param, flush=True)

    with torch.no_grad():
        exp_result = experiment_function(*param)

    return id_param, exp_result


def parallel_execution(parameters_enumeration, experiment_function, nb_processes):
    # multiprocess the computations
    argument_list = [(id_p, p, experiment_function) for id_p, p in parameters_enumeration]

    with Pool(processes=nb_processes) as pool:
        solution = pool.map(function_execution, argument_list)

    return solution
