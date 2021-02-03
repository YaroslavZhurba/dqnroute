import multiprocessing
import time
import psutil

from typing import *

from RunWrapper import run
from dqnroute.verification.exception import MarabouException

ARGS_MUKHUTDINOV = ["conveyor_topology_mukhutdinov",
                    "original_example_graph",
                    "original_example_settings_energy_test"]
ARGS_TARAU = ["conveyor_topology_tarau",
              "tarau2010_graph_original",
              "tarau2010_settings_regular"]
MARABOU_MEMORY_LIMIT = "--linux_marabou_memory_limit_mb 12288"

def embedding_verification_mukhutdinov(bound: float, epsilon: float, source: int, sink: int):
    run("embedding_adversarial_verification", *ARGS_MUKHUTDINOV, bound,
        f"--skip_graphviz --single_source {source} --single_sink {sink} --input_eps_l_inf {epsilon} "
        f"{MARABOU_MEMORY_LIMIT}")

def embedding_verification_tarau(bound: float, epsilon: float, source: int, sink: int):
    run("embedding_adversarial_verification", *ARGS_TARAU,
        bound, f"--skip_graphviz --single_source {source} --single_sink {sink} --input_eps_l_inf {epsilon} "
        f"{MARABOU_MEMORY_LIMIT}")
    
def lipschitz_verification_mukhutdinov(bound: float):
    run("q_adversarial_verification", *ARGS_MUKHUTDINOV, bound,
        "--skip_graphviz --single_source 1 --single_sink 3 --input_max_delta_q 20 "\
        "--learning_step_indices 1,2,3,8")

def lipschitz_verification_tarau(bound: float):
    run("q_adversarial_verification", *ARGS_TARAU, bound,
        "--skip_graphviz --single_source 0 --single_sink 1 --input_max_delta_q 10 "\
        "--learning_step_indices 17,22,23,24")

def killall(name: str):
    for proc in psutil.process_iter():
        if proc.name() == name:
            proc.kill()
    
def run_with_timeout(fun: Callable, args: List, timeout_sec: int):
    print()
    print("****************************************")
    print("*************** NEW RUN ****************")
    print("****************************************")
    print(f"Call: {fun.__name__}{args}")
    print()
    
    def f():
        fun(*args)
    
    try:
        # https://stackoverflow.com/questions/492519/timeout-on-a-function-call
        p = multiprocessing.Process(target=f)
        p.start()
        p.join(timeout_sec)
        if p.is_alive():
            p.terminate()
            #p.kill() #- will work for sure, no chance for process to finish nicely however
            print("TIMEOUT")
            time.sleep(1.0)
        p.join()
    finally:
        killall("Marabou")
    
if __name__ == "__main__":
    TIMEOUT = 60 * 120
    
    # original, 1 -> 3
    #for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8, 1.6]:
    for eps in [0.01]:
        #for c0 in [81.0, 44.0, 43.5, 43.12, 43.1]:
        for c0 in [81.0, 44.0, 43.5, 43.12, 43.1]:
            run_with_timeout(embedding_verification_mukhutdinov, [c0, eps, 1, 3], TIMEOUT)
    
    # original, 1 -> 2
    #for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8, 1.6]:
    for eps in [0.01]:
        #for c0 in [72.8, 54.12, 53.11, 53.1, 53.0]:
        for c0 in [72.8, 54.12, 53.11, 53.1, 53.0]:
            run_with_timeout(embedding_verification_mukhutdinov, [c0, eps, 1, 2], TIMEOUT)
    
    # tarau, 0 -> 1
    #for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8, 1.6]:
    for eps in [0.01]:
        #for c0 in [825.0, 820.0, 819.0, 818.6, 818.0]:
        for c0 in [825.0, 820.0, 819.0, 818.6, 818.0]:
            run_with_timeout(embedding_verification_tarau, [c0, eps, 0, 1], TIMEOUT)
    
    # tarau, 2 -> 0
    #for eps in [0., 0.01, 0.1, 0.2, 0.4, 0.8, 1.6]:
    for eps in [0.01]:
        #for c0 in [830.0, 820.0, 819.0, 818.4, 818.0]:
        for c0 in [830.0, 820.0, 819.0, 818.4, 818.0]:
            run_with_timeout(embedding_verification_tarau, [c0, eps, 2, 0], TIMEOUT)
    
    #lipschitz_verification_mukhutdinov(43.563)
    #lipschitz_verification_mukhutdinov(65.616)
    #lipschitz_verification_tarau(62694.8)
    #lipschitz_verification_tarau(67000.0)
