import numpy as np
import os 
from ncmcm.data_loaders.matlab_dataset import Database
import time 
from .evaluator import MarkovQueryEvaluator
from .visualisation import Visualiser

class WormSimulationInterface: 
    def __init__(self, matlab_path: str = "WT_NoStim.mat", cognitive_dir: str = "datasets/generated/saved_embeddings", worm_id: int = 0): 
        """
        Loads data for the specified worm and initializes the evaluator.  
        - worm_id: id of the worm, must be between 0 and 4
        """
        self.matlab_path = matlab_path
        self.cognitive_dir = cognitive_dir
        self.evaluator = None
        self.worm_id = None
        
        if worm_id < 0 or worm_id > 4: 
            raise ValueError("Invalid worm_id. The worm_id must be between 0 and 4.")
            
        self.worm_id = worm_id
        worm_data = Database(data_path=self.matlab_path, dataset_no=worm_id)
        real_behavioural = worm_data.behaviour

        cognitive_filename = f"B__BunDLeNet_worm_{worm_id}"
        full_cog_path = os.path.join(self.cognitive_dir, cognitive_filename)

        if not os.path.exists(full_cog_path): 
            raise FileNotFoundError(f"Could not locate the saved embeddings file: {full_cog_path}.")

        real_cognitive = np.loadtxt(full_cog_path).astype(int)

        min_len = min(len(real_behavioural), len(real_cognitive))
        if len(real_behavioural) != len(real_cognitive):
            real_cognitive = real_cognitive[:min_len]
            real_behavioural = real_behavioural[:min_len]

        self.evaluator = MarkovQueryEvaluator(real_cognitive, real_behavioural)
        print(f"The evaluator was intitialized for Worm {worm_id}.")

    def print_worm_states(self): 
        """
        Helper method to allow the user to get all the possible states of the worm. 
        """
        print(self.evaluator.labels)
    
    def plot_parallel_benchmark(self, start_state: str, target_state: str, steps: int, num_runs: int, ci: float = 0.95, severed_edges: list[tuple[str, str]] = None, imported_contexts: list[list[float]] = None, num_workers: int = 4): 
        """ 
        calls normal query and parallel query, times how long each one takes to run. sends this data to visualizer.
        output is a graph that shows the different speedup
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_runs: total number of runs
        - ci: the confidence interval 
        - severed_edges: one or more pairs of states between which the connection will be set to 0 (only relevant for counterfactual runs)
        - imported contexts: context of the factual run (only relevant for counterfactual runs)
        - num_workers: the number of workers (only relevant for parallel runs) 
        """
        start_seq = time.time() 
        self.evaluator.evaluate_query(start_state, target_state, steps, num_runs, ci, severed_edges, imported_contexts)
        duration_seq = time.time() - start_seq

        start_par = time.time() 
        self.evaluator.parallel_evaluate_query(start_state, target_state, steps, num_runs, ci, severed_edges, imported_contexts, num_workers)
        duration_par = time.time() - start_par

        Visualiser.plot_parallel_benchmark(duration_seq, duration_par)  

    def plot_scalability_analysis(self, start_state: str = "1-0", target_state: str = "2-0", steps: int = 100, run_sizes: list[int] = [100, 500, 1000, 5000, 10000, 50000], num_workers: int = 4, ci: float = 0.95): 
        """
        Runs a scalability analysis. 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_workers: the number of workers
        """
        sequential_times = []
        parallel_times = []

        for num_runs in run_sizes: 
            start_seq = time.time() 
            self.evaluator.evaluate_query(start_state, target_state, steps, num_runs, ci)
            duration_seq = time.time() - start_seq 
            sequential_times.append(duration_seq)

            start_par = time.time() 
            self.evaluator.parallel_evaluate_query(start_state, target_state, steps, num_runs, ci, num_workers=num_workers)
            duration_par = time.time() - start_par
            parallel_times.append(duration_par)

        Visualiser.plot_scalability_analysis(run_sizes, sequential_times, parallel_times)


    def plot_sample_trajectory(self, query_data: dict, run_index: int = 0): 
        """
        Plots the trajectory of a specified run. 
        Parameters: 
        - query_data: output of a query evaluation
        - run_index: the index of the specified run
        """
        if run_index >= len(query_data["trajectories"]): 
            raise IndexError(f"Run index {run_index} out of bounds.")

        target_trajectory = query_data["trajectories"][run_index]
        start_state = target_trajectory[0]
        steps_taken = len(target_trajectory) - 1
        title = f"Trajectory Path for Worm {self.worm_id}, Run {run_index}, Start State: {start_state}, Steps Taken: {steps_taken}"
        Visualiser.plot_trajectory(target_trajectory, title)

    def evaluate_query(self, start_state: str, target_state: str, steps: int = 100, num_runs: int = 1000, ci: float = 0.95, run_parallel: bool = False, num_workers: int = 4): 
        """
        Runs a query evaluation. Outputs a plot that shows after how many steps target_state was reached across all performed runs. 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_runs: the number of runs that will be performed in the evaluation
        - ci: the confidence interval
        - run_parallel: decides if the runs should be executed in parallel or sequentially
        - num_workers: the number of workers (only relevant for parallel runs)
        """
        if run_parallel: 
            result = self.evaluator.evaluate_query(
                    start_state = start_state, 
                    target_state = target_state, 
                    steps = steps, 
                    num_runs = num_runs, 
                    ci = ci,  
                    num_workers = num_workers
                )

        else: 
            result = self.evaluator.evaluate_query(
                start_state = start_state, 
                target_state = target_state, 
                steps = steps, 
                num_runs = num_runs, 
                ci = ci, 
            )

        Visualiser.plot_query(result, target_state)

        return result

    def evaluate_counterfactual(self, start_state: str, target_state: str, steps: int = 100, num_runs: int = 1000, ci: float = 0.95, run_parallel: bool = False, num_workers: int = 4, severed_edges: list[tuple[str, str]] = None): 
        """
        Runs a counterfactual evaluation. Outputs a plot of the factual query that shows after how many iterations the target state was reached. Additionally outputs the same plot for the counterfactual query. 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored 
        - steps: maximal number of iterations
        - num_runs: the number of runs that will be performed in the evaluation
        - ci: the confidence interval
        - run_parallel: decides if the runs should be executed in parallel or sequentially
        - num_workers: the number of workers (only relevant for parallel runs)
        - severed_edges: one or more pairs of states between which the connection will be set to 0 (only relevant for counterfactual runs)
        """
        if run_parallel: 
            factual_results, counterfactual_results = self.evaluator.evaluate_counterfactual_query(
                start_state = start_state, 
                target_state = target_state, 
                steps = steps, 
                num_runs = num_runs, 
                ci = ci,  
                num_workers = num_workers,
                severed_edges = severed_edges
            )
        else:
            factual_results, counterfactual_results = self.evaluator.evaluate_counterfactual_query(
                start_state = start_state, 
                target_state = target_state, 
                steps = steps, 
                num_runs = num_runs, 
                ci = ci,  
                severed_edges = severed_edges
            )

        Visualiser.plot_query(factual_results, target_state)
        Visualiser.plot_query(counterfactual_results, target_state, is_counterfactual=True)

        return factual_results, counterfactual_results

    def plot_frequently_traversed(self, query_metrics: dict, max_states: int = 10): 
        """
        Plots the most frequently traversed states from most to least frequent, and how often each state was reached across all performed runs. 
        Parameters: 
        - query_metrics: the output of a query evaluation
        - max_states: the maximal number of states that will be considered
        """
        Visualiser.plot_traversed_states(query_metrics, max_states)

    def plot_network_topology(self, min_probability: float = 0.05, max_probability: float = 0.95, filename: str = None): 
        """
        Plots the topology of the entire network. 
        Parameters: 
        - min_probability: connections with a probability that is less than this value will not be considered
        - max_probability: connections with a probability that is higher than this value will not be considered
        - filename: if a filename is provided the graph will be saved in an external file
        """ 
        Visualiser.plot_network_topology(self.evaluator, min_probability, max_probability, filename)
            
            

    









        