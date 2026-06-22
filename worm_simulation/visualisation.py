import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Visualiser: 
    @staticmethod
    def plot_query(query_metrics: dict | None, target_state: str, is_counterfactual: bool = False): 
        """
        Outputs the success rate and the confidence interval of the query. Creates that shows statistics about after how many steps target_state was reached across all of the performed runs. 
        Parameters: 
        - query_metrics: dictionary returned by MarkovQueryEvaluator.evaluate_query or None (if no runs were successful)
        - target_state: label of the target_state used in the query
        """
        if query_metrics is None: 
            print("No runs were successful.")
            return 

        title = f"Simulation for {target_state}"
        if is_counterfactual: 
            title += " (Counterfactual)"

        x_steps = query_metrics["x_steps"]
        frequencies = query_metrics["frequencies"]
        success_rate = query_metrics["success_rate"]
        ci_level = query_metrics["ci_level"]
        ci_bounds = query_metrics["ci_bounds"]

        print(f"Success rate: {success_rate} | {ci_level} Confidence Interval: [{ci_bounds[0]}%, {ci_bounds[1]}%]")

        plt.figure(figsize=(10,6))
        plt.bar(x_steps, frequencies)
        plt.xlabel("Number of steps to reach goal state")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(linestyle="--", alpha=0.6)
        plt.show()

    @staticmethod 
    def plot_traversed_states(query_metrics: dict | None, max_states_to_show: int = 10): 
        """
        Plots the x and most frequented states (across all runs) and how often they were traversed. 
        Parameters: 
        - query_metrics: output of a query evaluation
        - max_states_to_show: int that decides how many of the most frequent states should be plotted
        """
        if query_metrics is None: 
            print("No metrics available to plot.")
            return 

        state_traversal_counts = query_metrics["state_traversal_counts"]
        sorted_states = sorted(state_traversal_counts.items(), key=lambda item: item[1], reverse=True)

        sorted_states = sorted_states[:max_states_to_show]

        states = [item[0] for item in sorted_states]
        run_counts = [item[1] for item in sorted_states]

        plt.figure(figsize=(10,6))
        plt.bar(states, run_counts)
        plt.xlabel(f"{max_states_to_show} most frequent states")
        plt.ylabel("Frequency")
        plt.grid(linestyle="--", alpha=0.6)
        plt.show()

    @staticmethod 
    def plot_network_topology(evaluator, min_probability: float = 0.05, max_probability: float = 1.0, filename: str = None): 
        """
        Plots the connections between the different states, where the probability of transition is >= min_probability
        Parameters: 
        - evaluator: an instance of the class MarkovQueryEvaluator
        - min_probability: connections with a probability less than value this will be filtered out (must be between 0.0 and 1.0)
        - max_probability: connections with a probability that is higher than value this will be filtered out (must be between 0.0 and 1.0)
        - filename: if a filename is provided the graph will be saved in an external file 
        """
        if min_probability < 0.0 or min_probability > 1.0: 
            raise ValueError("min_probability must be between 0.0 and 1.0")

        if max_probability < 0.0 or max_probability > 1.0: 
            raise ValueError("max probability must be between 0.0 and 1.0")
        
        G = nx.DiGraph()

        for from_label in evaluator.labels: 
            from_id = evaluator.label_to_idx[from_label]
            for to_label in evaluator.labels: 
                to_id = evaluator.label_to_idx[to_label]
                probability = evaluator.transition_matrix[from_id, to_id]

                if probability >= min_probability and probability <= max_probability and from_label != to_label: 
                    G.add_edge(from_label, to_label, weight=probability)

        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)

        if len(G.edges) <= 0: 
            print(f"No connections found with a probability of at least {min_probability * 100}%.")
            return

        figure = plt.figure(figsize=(16,12))
        ax = figure.add_subplot(111)
        position = nx.circular_layout(G, scale=30.0)
        nx.draw_networkx_nodes(G, position, node_size=700, ax=ax)
        nx.draw_networkx_labels(G, position, font_weight="bold", ax=ax)
        active_edges = G.edges
        weights = [G[u][v]['weight'] for u, v in active_edges]
        nx.draw_networkx_edges(
            G, 
            position,
            edgelist=active_edges,
            node_size=700, 
            arrowstyle="-|>", 
            arrowsize=18, 
            #connectionstyle="arc3, rad=0.2", 
            ax=ax
        )
        edge_labels = {(u, v): f"{G[u][v]['weight']*100:.1f}%" for u, v in active_edges}
        nx.draw_networkx_edge_labels(G, position, edge_labels=edge_labels, label_pos=0.4, rotate=True, bbox=dict(boxstyle="round, pad=0.15", fc="white", ec="none", alpha=0.95), ax=ax)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-35.5, 35.5)
        ax.set_ylim(-35.5, 35.5)
        ax.axis("off")

        if filename is not None: 
            plt.savefig(filename, dpi=300)
            plt.close(figure)
            print(f"Graph saved to {filename}.")
        else: 
            plt.show()

    @staticmethod
    def plot_parallel_benchmark(sequential_time: float, parallel_time: float): 
        """
        Plots a bar chart, compares the sequential runtime against the parallel runtime.
        Parameters: 
        - sequential_time: the runtime of the sequential run
        - parallel_time: the runtime of the parallel run
        """
        modes = ["Sequential (Single-Core)", "Parallel (Multi-Core)"]
        time_data = [sequential_time, parallel_time]

        plt.figure(figsize=(10,6))
        plt.bar(modes, time_data)
        plt.ylabel("Execution Time (Seconds)")
        plt.title("Performance Benchmark: Sequential vs. Parallel Simulation")
        plt.grid(linestyle="--", alpha=0.5)
        plt.show()

    @staticmethod
    def plot_scalability_analysis(run_sizes: list[int], sequential_times: list[float], parallel_times: list[float]): 
        """
        Plots scalability analysis. 
        Parameters: 
        - run_sizes: for each value in this list, a sequential and a parallel evaluation will be run, with the given value as the number of runs that are executed in each evalualtion  
        - sequential_times: the runtimes of all the sequential runs
        - parallel_times: the runtimes of all the parallel runs
        """
        x_indices = np.arange(len(run_sizes))
        bar_width = 0.35

        plt.figure(figsize=(10,6))
        bars_seq = plt.bar(x_indices - bar_width/2, sequential_times, width=bar_width, label="Sequential (Single-Core)")
        bars_par = plt.bar(x_indices + bar_width/2, parallel_times, width=bar_width, label="Parallel (Multi-Core)")
        plt.xticks(x_indices, [f"{n:,}" for n in run_sizes])
        plt.xlabel("Total Monte Carlo Runs")
        plt.ylabel("Execution Time (Seconds)")
        plt.title("System Scalability: Sequential  vs. Parallel Simulation")
        plt.grid(linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", frameon=True)
        plt.show()

    @staticmethod 
    def plot_trajectory(trajectory: list[str], title: str = "State Trajectory"): 
        """
        Plots the trajectory of a specific run. 
        Parameters: 
        - trajectory: a list of states that were traversed in the run
        - title: the title of the graph
        """
        if not trajectory: 
            raise ValueError("Empty trajectory, nothing to plot.")

        G = nx.DiGraph()

        edge_labels = {}

        for i in range(len(trajectory) - 1): 
            u = trajectory[i]
            v = trajectory[i+1]

            if not G.has_edge(u,v) and u != v: 
                G.add_edge(u,v)
                edge_labels[(u,v)] = f"{i+1}"

        plt.figure(figsize=(10,6))
        pos = nx.circular_layout(G)
        start_node = trajectory[0]
        end_node = trajectory[-1]
        other_nodes = [node for node in G.nodes() if node != start_node and node != end_node]

        nx.draw_networkx_nodes(
            G, 
            pos, 
            nodelist = [start_node], 
            node_size = 1000,
            node_color = "#34A87D"
        )

        nx.draw_networkx_nodes(
            G, 
            pos, 
            nodelist = [end_node], 
            node_size = 1000, 
            node_color = "#A62121"
        )

        if other_nodes: 
            nx.draw_networkx_nodes(
                G,
                pos, 
                nodelist = other_nodes, 
                node_size = 1000,
                node_color = "#A69D9D"
            )

        nx.draw_networkx_edges(
            G, 
            pos, 
            arrowstyle = "-|>", 
            node_size = 1000
        )

        nx.draw_networkx_labels(
            G, 
            pos, 
            font_size = 10
        )

        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels = edge_labels, 
            rotate = False, 
            label_pos = 0.5
        )

        plt.title(title)
        plt.axis('off')
        plt.show()






    