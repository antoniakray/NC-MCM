import numpy as np 
from collections import Counter
from ncmcm.cognitive_graphs.calculations import adj_matrix_ncmcm
import math 
from statistics import NormalDist
from concurrent.futures import ProcessPoolExecutor

class MarkovQueryEvaluator: 
    def __init__(self, cognitive_series: np.ndarray, behavioral_series: np.ndarray): 
        matrix, state_labels = adj_matrix_ncmcm(cognitive_series, behavioral_series)
        self.transition_matrix = np.array(matrix, dtype=float)
        self.unique_states = np.unique(state_labels)
        self.labels = [str(state) for state in self.unique_states]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.num_states = len(self.labels)

        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)

        for idx in range(self.num_states): 
            if row_sums[idx] == 0: 
                self.transition_matrix[idx,idx] = 1.0 
                row_sums[idx] = 1.0

        self.transition_matrix = self.transition_matrix / row_sums

        #perform double normalization
        final_sums = self.transition_matrix.sum(axis=1, keepdims=True)

        for idx in range(self.num_states): 
            row = self.transition_matrix[idx]
            non_zero_ids = np.where(row > 0)[0]
            if len(non_zero_ids) > 0: 
                last_active_id = non_zero_ids[-1]
                row[last_active_id] += (1.0 - final_sums[idx].item())

    def simulate_trajectory(self, start_state: str, target_state: str,  steps: int, severed_edges: list[tuple[str, str]] = None, execution_context: list[float] = None) -> tuple[list[str], list[float]]: 
        """
        Simulates a random walk sequence. Returns a tuple: the trajectory and the execution context. 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - severed_edges: one or more pairs of states between which the connection will be set to 0 (only relevant for counterfactual runs)
        - execution_context: context of the factual run (only relevant for counterfactual runs)
        """
        if start_state not in self.label_to_idx:
            raise ValueError(f"State '{start_state}' is invalid.")

        if target_state != "all" and target_state not in self.label_to_idx: 
            raise ValueError(f"State '{target_state}' is invalid.")

        current_idx = self.label_to_idx[start_state]
        trajectory = [start_state]
        random_draws = []

        is_counterfactual = execution_context is not None 
        severed_edges = severed_edges or []

        for step_idx in range(steps): 
            probabilities = self.transition_matrix[current_idx].copy() 

            #sever the edge
            for u, v in severed_edges: 
                if u in self.label_to_idx and v in self.label_to_idx: 
                    if current_idx == self.label_to_idx[u]: 
                        probabilities[self.label_to_idx[v]] = 0.0

            #re-normalize in case previous step severed some of the edges 
            p_sum = np.sum(probabilities)
            if p_sum > 0: 
                probabilities /= p_sum
            else: 
                probabilities[current_idx] = 1.0 

            if is_counterfactual: 
                random_value = execution_context[step_idx]
            else: 
                random_value = np.random.rand() 
                random_draws.append(random_value)

            cumulative_probabilities = np.cumsum(probabilities)
            current_idx = np.searchsorted(cumulative_probabilities, random_value)

            next_state_label = self.labels[current_idx]
            trajectory.append(next_state_label)
                    
        return trajectory, random_draws 

    def evaluate_query (self, start_state: str, target_state: str, steps: int, num_runs: int, ci: float = 0.95, severed_edges: list[tuple[str, str]] = None, imported_contexts: list[list[float]] = None) -> dict | None: 
        """ 
        Runs one query multiple times, extracts metrics about those runs and returns this information in a dictionary. 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_runs: total number of runs 
        - ci: confindence interval, if not specified the default will be set to 0.95
        - severed_edges: one or more pairs of states between which the connection will be set to 0 (only relevant for counterfactual runs)
        - imported_contexts: context of the factual run (only relevant for counterfactual runs)
        """

        if ci > 1.0 or ci < 0.0: 
            raise ValueError("The confidence interval must be a float between 0.0 and 1.0")

        num_steps_across_all_runs = []
        successful_runs = 0 
        state_traversal_counts = {label: 0 for label in self.labels}
        export_contexts = []
        trajectories = []

        for run_idx in range(num_runs): 
            context = imported_contexts[run_idx] if imported_contexts is not None else None 

            #extract the trajectory and the context for this specific run
            trajectory, recorded_context = self.simulate_trajectory(start_state, target_state, steps, severed_edges=severed_edges, execution_context=context)
            trajectories.append(trajectory)

            if imported_contexts is None: 
                export_contexts.append(recorded_context)

            for state in set(trajectory): 
                state_traversal_counts[state] += 1

            if target_state in trajectory: 
                first_reach_idx = trajectory.index(target_state)
                successful_runs += 1
                num_steps_across_all_runs.append(first_reach_idx)

        if successful_runs <= 0: 
            return {
                "x_steps": list(range(0, steps + 1)),
                "frequencies": [0] * (steps + 1),
                "success_rate": f"0/{num_runs} (0.0%)", 
                "ci_level": f"{ci * 100:.0f}%", 
                "ci_bounds": (0.0, 0.0), 
                "state_traversal_counts": state_traversal_counts, 
                "contexts": export_contexts, 
                "successful_runs": successful_runs,
                "trajectories": trajectories
            }

        counts = Counter(num_steps_across_all_runs)
        plot_x_steps = list(range(0, steps + 1))
        frequencies = [counts.get(step, 0) for step in plot_x_steps]
        p = successful_runs / num_runs 
        successful_percentage = p * 100

        ci_lower, ci_upper = self._calculate_binomial_ci(p, num_runs, ci)


        return {
            "x_steps": plot_x_steps,
            "frequencies": frequencies,
            "success_rate": f"{successful_runs}/{num_runs} ({successful_percentage:.1f}%)", 
            "ci_level": f"{ci * 100:.0f}%", 
            "ci_bounds": (round(ci_lower * 100, 2), round(ci_upper * 100, 2)), 
            "state_traversal_counts": state_traversal_counts, 
            "contexts": export_contexts, 
            "successful_runs": successful_runs,
            "trajectories": trajectories
        }

    def evaluate_counterfactual_query(self, start_state: str, target_state: str, steps: int, num_runs: int, severed_edges: list[tuple[str, str]], ci: float = 0.95, run_parallel: bool = False, num_workers: int = 4) -> tuple[dict, dict]: 
        """
        Automatically runs the query and the counterfactual, returns two dicts with the metrics for both runs. 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_runs: total number of runs 
        - severed_edges: one or more pairs of states between which the connection will be set to 0 (only relevant for counterfactual runs)
        - ci: the confidence interval 
        - run_parallel: decides if the runs should be executed in parallel or sequentially
        - num_workers: number of workers (only relevant for parallel runs)
        """

        query_engine = self.parallel_evaluate_query if run_parallel else self.evaluate_query
        args = {"num_workers": num_workers} if run_parallel else {}

        #run the factual query
        factual_results = query_engine(
            start_state = start_state, 
            target_state = target_state, 
            steps = steps, 
            num_runs = num_runs, 
            ci = ci, 
            **args 
        )

        #run the counterfactual query
        counterfactual_results = query_engine(
            start_state = start_state, 
            target_state = target_state, 
            steps = steps, 
            num_runs = num_runs, 
            ci = ci, 
            severed_edges = severed_edges, 
            imported_contexts = factual_results["contexts"], 
            **args
        )

        return factual_results, counterfactual_results

    def _calculate_binomial_ci(self, p: float, n: int, confidence: float) -> tuple[float, float]: 
        """
        Private method to calculate the proportions for a binomial confidence interval. Uses standard normal distribution. 
        Returns tuple with the (lower, uppper) proportions. 
        Parameters: 
        - p: sample proportion
        - n: sample size
        - confidence: the confidence interval
        """
        standard_error = math.sqrt((p * (1.0 - p)) / n)
        alpha = 1.0 - confidence 
        target_percentile = 1.0 - (alpha / 2.0)
        z = NormalDist().inv_cdf(target_percentile)
        margin = z * standard_error 
        lower_bound = max(0.0, p - margin)
        upper_bound = min(1.0, p + margin)

        return lower_bound, upper_bound
        

    def _calculate_steps_to_reach_all_states(self, trajectory: list[str]) -> int: 
        """
        Private method that calculates how many steps it took to reach all states. Returns an integer with number of transitions.
        Pre-condition: Assumes that all states were reached within the given trajectory
        Parameters: 
        - trajectory: list of string labels representing the path
        """
        target_length = len(self.labels)
        seen_states = set()
        for idx, state in enumerate(trajectory): 
            seen_states.add(state)
            if len(seen_states) == target_length: 
                return idx

        return len(trajectory) - 1 

    def _check_if_all_states_were_reached(self, trajectory: list[str]) -> bool: 
        """
        Private method that checks whether all states were reached. Returns true or false. 
        Parameters: 
        - trajectory: list of string labels representing the path
        """
        target_set = set(self.labels)
        trajectory_set = set(trajectory)
        return target_set == trajectory_set

    def parallel_evaluate_query(self, start_state: str, target_state: str, steps: int, num_runs: int, ci: float = 0.95, severed_edges: list[tuple[str, str]] = None, imported_contexts: list[list[float]] = None, num_workers: int = 4) -> dict: 
        """ 
        PARALLEL version of evaluate_query 
        Parameters: 
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_runs: total number of runs 
        - ci: confindence interval, if not specified the default will be set to 0.95
        - severed_edges: one or more pairs of states between which the connection will be set to 0 (only relevant for counterfactual runs)
        - imported_contexts: context of the factual run (only relevant for counterfactual runs)
        - num_workers: the number of workers (only relevant for parallel runs)
        """
        runs_per_worker = num_runs // num_workers 
        remainder = num_runs % num_workers 
        worker_runs = [runs_per_worker + (1 if worker_id < remainder else 0) for worker_id in range(num_workers)]

        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = []
        current_offset = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for runs in worker_runs: 
                worker_contexts = None 
                if imported_contexts is not None: 
                    worker_contexts = imported_contexts[current_offset : current_offset + runs]
                    current_offset += runs
    
                futures.append(
                    executor.submit(
                        self.evaluate_query, start_state, target_state, steps, runs, ci, severed_edges, worker_contexts
                    )
                )
    
            results = [future.result() for future in futures]

        if not results: 
            return {
                "x_steps": list(range(0, steps + 1)),
                "frequencies": [0] * (steps + 1),
                "success_rate": f"0/{num_runs} (0.0%)", 
                "ci_level": f"{ci * 100:.0f}%", 
                "ci_bounds": (0.0, 0.0),
                "state_traversal_counts": {label: 0 for label in self.labels}, 
                "contexts": [],
                "successful_runs": 0
                
            }
        
        combined_frequencies = np.zeros(steps + 1, dtype=int)
        combined_traversal = {label: 0 for label in self.labels}
        total_successful_runs = 0
        all_contexts = []
        trajectories = []

        for result in results: 
            combined_frequencies += np.array(result["frequencies"])
            for label in self.labels: 
                combined_traversal[label] += result["state_traversal_counts"][label]
            total_successful_runs += result["successful_runs"]
            all_contexts.extend(result["contexts"])
            trajectories.extend(result["trajectories"])

        plot_x_steps = list(range(0, steps+1))
        p = total_successful_runs / num_runs 
        ci_lower, ci_upper = self._calculate_binomial_ci(p, num_runs, ci)

        return {
            "x_steps": plot_x_steps,
            "frequencies": combined_frequencies.tolist(),
            "success_rate": f"{total_successful_runs}/{num_runs} ({p*100:.1f}%)", 
            "ci_level": f"{ci * 100:.0f}%", 
            "ci_bounds": (round(ci_lower * 100, 2), round(ci_upper * 100, 2)), 
            "state_traversal_counts": combined_traversal, 
            "contexts": all_contexts,
            "successful_runs": total_successful_runs, 
            "trajectories": trajectories
        }

    def evaluate_sequence_query(self, start_state: str, sequence: list[str], steps: int, num_runs: int, ci: float = 0.95, severed_edges: list[tuple[str, str]] = None, imported_contexts: list[list[float]] = None) -> dict: 
        """
        askdjfas
        Parameters: 
        - 
        """
        if ci > 1.0 or ci < 0.0: 
            raise ValueError("The confidence interval must be a float between 0.0 and 1.0")

        if not sequence: 
            raise ValueError("Target sequence cannot be empty.")

        for state in sequence: 
            if state not in self.labels: 
                raise ValueError(f"State {state} is invalid.")
        
        num_steps_across_all_runs = []
        successful_runs = 0 
        state_traversal_counts = {label: 0 for label in self.labels}
        export_contexts = []
        trajectories = []
        sequence_start_indices = []

        for run_idx in range(num_runs): 
            context = imported_contexts[run_idx] if imported_contexts is not None else None 

            #extract the trajectory and the context for this specific run 
            trajectory, recorded_context = self.simulate_trajectory(start_state, sequence[-1], steps, severed_edges=severed_edges, execution_context=context)
            trajectories.append(trajectory)

            if imported_contexts is None: 
                export_contexts.append(recorded_context)

            for state in set(trajectory): 
                state_traversal_counts[state] += 1

            first_reach_id = self._find_sequence_in_trajectory(trajectory, sequence)
            sequence_start_indices.append(first_reach_id)

            if first_reach_id != -1: 
                successful_runs += 1
                num_steps_across_all_runs.append(first_reach_id)

        if successful_runs <= 0: 
            return {
                "x_steps": list(range(0, steps + 1)),
                "frequencies": [0] * (steps + 1),
                "success_rate": f"0/{num_runs} (0.0%)", 
                "ci_level": f"{ci * 100:.0f}%", 
                "ci_bounds": (0.0, 0.0), 
                "state_traversal_counts": state_traversal_counts, 
                "contexts": export_contexts, 
                "successful_runs": successful_runs,
                "trajectories": trajectories, 
                "sequence_start_indices": sequence_start_indices
            }

        counts = Counter(num_steps_across_all_runs)
        plot_x_steps = list(range(0, steps + 1))
        frequencies = [counts.get(step, 0) for step in plot_x_steps]
        p = successful_runs / num_runs 
        successful_percentage = p * 100

        ci_lower, ci_upper = self._calculate_binomial_ci(p, num_runs, ci)


        return {
            "x_steps": plot_x_steps,
            "frequencies": frequencies,
            "success_rate": f"{successful_runs}/{num_runs} ({successful_percentage:.1f}%)", 
            "ci_level": f"{ci * 100:.0f}%", 
            "ci_bounds": (round(ci_lower * 100, 2), round(ci_upper * 100, 2)), 
            "state_traversal_counts": state_traversal_counts, 
            "contexts": export_contexts, 
            "successful_runs": successful_runs,
            "trajectories": trajectories, 
            "sequence_start_indices": sequence_start_indices
        }

    def _find_sequence_in_trajectory(self, trajectory: list[str], sequence: list[str]) -> int: 
        n = len(trajectory)
        k = len(sequence)
        for i in range(n - k + 1): 
            if trajectory[i : i+k] == sequence: 
                return i
        return -1
        

        


        