import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from worm_simulation.evaluator import MarkovQueryEvaluator
from pyvis.network import Network
import math
import pandas as pd 
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Visualiser: 
    @staticmethod
    def plot_query(query_metrics: dict | None, target_state: str, is_counterfactual: bool = False): 
        """
        Outputs the success rate and the confidence interval of the query. Creates a plot that shows statistics about after how many steps target_state was reached across all of the performed runs. 
        Parameters: 
        - query_metrics: dictionary returned by MarkovQueryEvaluator.evaluate_query or None (if no runs were successful)
        - target_state: label of the target_state used in the query
        - is_counterfactual: should be set to true, for a counterfactual run
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
        nx.draw_networkx_nodes(
            G, 
            position, 
            node_size=700, 
            ax=ax
        )
        nx.draw_networkx_labels(
            G, 
            position, 
            font_weight="bold", 
            ax=ax
        )
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
        nx.draw_networkx_edge_labels(
            G, 
            position, 
            edge_labels=edge_labels, 
            label_pos=0.5, 
            rotate=True, 
            #bbox=dict(boxstyle="round, pad=0.15", fc="white", ec="none", alpha=0.95), 
            ax=ax
        )
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
    def plot_trajectory(trajectory: list[str], title: str = "State Trajectory", filename: str = "trajectory.html"): 
        """
        Plots the trajectory of a specific run. 
        Parameters: 
        - trajectory: a list of states that were traversed in the run
        - title: the title of the graph
        - filename: optional filename, if provided the plot will be saved to a file of that name
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
            node_size = 1000
        )

        nx.draw_networkx_nodes(
            G, 
            pos, 
            nodelist = [end_node], 
            node_size = 1000 
        )

        if other_nodes: 
            nx.draw_networkx_nodes(
                G,
                pos, 
                nodelist = other_nodes, 
                node_size = 1000
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

    def plot_interactive_topology(evaluator: MarkovQueryEvaluator, min_probability: float = 0.05, max_probability: float = 0.95, filename: str = "topology.html"): 
        """
        Plots the connections between the different states, where the probability of transition is >= min_probability
        Parameters: 
        - evaluator: an instance of the class MarkovQueryEvaluator
        - min_probability: connections with a probability less than this value will be filtered out (must be between 0.0 and 1.0)
        - max_probability: connections with a probability more than this value will be filtered out (must be between 0.0 and 1.0)
        - filename: optional filename, if provided the plot will be saved to a file of that name
        """
        G = nx.DiGraph()

        for from_label in evaluator.labels:
            from_id = evaluator.label_to_idx[from_label]
            for to_label in evaluator.labels: 
                to_id = evaluator.label_to_idx[to_label]
                probability = evaluator.transition_matrix[from_id, to_id]

                if min_probability <= probability <= max_probability and from_label != to_label: 
                    G.add_edge(from_label, to_label, weight=probability)

        if len(G.edges) <= 0: 
            print(f"No connections with a probability between {min_probability*100}% and {max_probability*100}% found.")
            return 

        net = Network(
            height = "530px", 
            width = "100%", 
            directed = True, 
            notebook = True, 
            cdn_resources = "remote"
        )

        net.set_options("""
            {
                "interaction": {
                    "hover": true, 
                    "zoomView": true, 
                    "dragView": true, 
                    "hoverConnectedEdges": true, 
                    "selectConnectedEdges": true 
                },  
                "physics": {
                    "enabled": true, 
                    "hierarchicalRepulsion": {
                        "nodeSpacing": 100, 
                        "avoidOverlap": 1.0
                    }
                }, 
                "edges": {
                    "color": {
                        "color": "#3395A6",
                        "inherit": false,
                        "highlight": "#F0861A", 
                        "hover": "#F0861A" 
                    }
                }
            }
        """)

        try: 
            position = nx.kamada_kawai_layout(G)
        except: 
            position.nx_spring_layout(G, k=1.0, seed=42)

        for node in G.nodes():
            raw_x = position[node][0]
            raw_y = position[node][1]

            scaled_x = raw_x * 500
            scaled_y = raw_y * 200
            
            net.add_node(
                node, 
                label = str(node), 
                shape = "circle",
                size = 35, 
                x = scaled_x, 
                y = scaled_y,
                font = {
                    "size": 18, 
                    "face": "arial", 
                }, 
                color = {
                    "background": "#5EC5D1",
                    "border": "#2E8594",
                    "highlight": {
                        "background": "#F0861A",
                        "border": "#C76606"
                    },
                    "hover": {
                        "background": "#F0861A",
                        "border": "#C76606"
                    }
                }
                
            )

        for u, v, data in G.edges(data=True): 
            probability = data['weight'] * 100
            scaled_probability = math.log10(1.0 + data['weight'] * 9.0)
            line_thickness = 1.0 + scaled_probability * 2.0

            net.add_edge(
                u,
                v, 
                value = line_thickness, 
                title = f"Probability: {probability:.1f}%", 
                arrowStrikethrough = False
            )
            
        return net.show(filename)

    def plot_interactive_trajectory(trajectory: list[str], title: str = "Interactive State Trajectory", filename: str = "trajectory.html"): 
        """
        Plots the trajectory of a specific run. 
        Parameters: 
        - trajectory: a list of states that were traversed in the run
        - title: optional title of the plot
        - filename: optional filename, if provided the plot will be saved to a file of that name
        """
        if not trajectory: 
            raise ValueError("Empty trajectory, nothing to plot.")

        G = nx.DiGraph()
        edge_labels = {}

        for i in range(len(trajectory) - 1): 
            u = trajectory[i]
            v = trajectory[i+1]

            if u != v: 
                if not G.has_edge(u,v): 
                    G.add_edge(u,v)
                    edge_labels[(u,v)] = []
                edge_labels[(u,v)].append(f"{i+1}")

        net = Network(
            height = "530px", 
            width = "100%", 
            directed = True, 
            notebook = True, 
            cdn_resources = "remote"
        )

        net.set_options("""
            {
                "interaction": {
                    "hover": true, 
                    "zoomView": true, 
                    "dragView": true, 
                    "hoverConnectedEdges": false, 
                    "selectConnectedEdges": false
                }, 
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -3500, 
                        "centralGravity": 1.2, 
                        "springLength": 100, 
                        "springConstant": 0.05, 
                        "damping": 0.3, 
                        "avoidOverlap": 1.0
                    }, 
                    "stabilization": {
                        "enabled": true, 
                        "iterations": 1000, 
                        "fit": true
                    }
                }, 
                "edges": {
                    "color": {
                        "color": "#3395A6",
                        "inherit": false,
                        "highlight": "#F0861A", 
                        "hover": "#F0861A" 
                    }
                }
            }
        """)

        start_node = trajectory[0]

        for node in G.nodes(): 
            if node == start_node: 
                background = "#F55151"
                border = "#A63333"
            else: 
                background = "#5EC5D1"
                border = "#2E8594"
            
            net.add_node(
                node, 
                label = str(node), 
                shape = "circle", 
                size = 25, 
                font = {
                    "size": 14, 
                    "face": "arial"
                },
                color = {
                    "background": background,
                    "border": border,
                    "highlight": {
                        "background": background,
                        "border": border
                    },
                    "hover": {
                        "background": background,
                        "border": border
                    }
                } 
            )

        for u, v in G.edges():
            steps_taken = f"Steps: {', '.join(edge_labels[(u,v)])}"
            net.add_edge(
                u, v, 
                width = 3, 
                label = edge_labels[(u,v)][0],
                title = steps_taken, 
                color = {
                    "color": "#3395A6",
                    "inherit": False,
                    "highlight": "#F0861A", 
                    "hover": "#F0861A" 
                }, 
                arrowStrikethrough = False
            )


        with open(filename, "w", encoding="utf-8") as f: 
            f.write(net.generate_html(local=False).replace("border: 1px solid lightgray;", "border: none;"))

        return net.show(filename)


    def plot_interactive_query(query_metrics: dict | None, start_state: str, target_state: str, steps: int, num_runs: int, is_counterfactual: bool = False, is_sequence: bool = False, sequence: list[str] = None, filename: str = "query_distribution.html"): 
        """
        Outputs the success rate and the confidence interval of the query. Creates a plot that shows after how many steps target_state was reached across all of the performed runs.
        Parameters: 
        - query_metrics: output of a query evaluation
        - start_state: label of the initial node
        - target_state: label of the final node OR "all" to check if entire state space was explored
        - steps: maximal number of iterations
        - num_runs: total number of runs
        - is_counterfactual: should be set to true, for a counterfactual run 
        - is_sequence: should be set to true, for a sequential run 
        - sequence: a list of nodes
        - filename: optional filename, if provided the plot will be saved to a file of that name
        """
        if query_metrics is None: 
            print("No runs were successful.")
            return 

        if is_sequence and sequence: 
            target_label = " -> ".join(sequence)
            title = f"Sequence Simulation from {start_state} to {sequence} in {steps} steps across {num_runs} runs"
            xlabel = "Steps To Reach Sequence Start"
        else: 
            title = f"Simulation from {start_state} to {target_state} in {steps} steps across {num_runs} runs"
            xlabel = "Steps To Reach Goal"

        subtitle = (
            f"Success Rate: {query_metrics["success_rate"]} &nbsp;|&nbsp; {query_metrics["ci_level"]} Confidence Interval: [{query_metrics["ci_bounds"][0]:.2f}%, {query_metrics["ci_bounds"][1]:.2f}%]"
        )
        
        if is_counterfactual: 
            title += " (Counterfactual)"

        html_title = (
            f"<b>{title}</b><br>"
            f"<span style='font_size: 13px; color=#666666, font-weight=normal;'>"
            f"{subtitle}"
            f"</span>"
        )

        df = pd.DataFrame({
            "Steps to Reach Goal": query_metrics["x_steps"], 
            "Frequency": query_metrics["frequencies"]
        })

        fig = px.bar(
            df, 
            x = "Steps to Reach Goal", 
            y = "Frequency", 
            title = html_title, 
            template = "plotly_white", 
            color_discrete_sequence = ["#3395A6"]
        )

        fig.update_traces(
            hovertemplate = (
                "<b>Step</b> %{x}<br>" + 
                "<b>Frequency</b> %{y:.0f}<br>" + 
                "<extra></extra>"
            )
        )

        fig.update_layout(
            title_font = dict(size=18, family="Arial", color="#222222"), 
            xaxis = dict(
                title = xlabel, 
                title_font = dict(size=12, family="Arial")
            ), 
            yaxis = dict(
                title = "Frequency", 
                title_font = dict(size=12, family="Arial"), 
                gridcolor = "#EBEBEB"
            ), 
            hoverlabel = dict(
                bgcolor = "#FFFFFF", 
                font_size = 13, 
                font_family = "Arial"
            ), 
            margin = dict(l=50, r=50, t=100, b=50)
        )

        fig.write_html(filename)
        pio.renderers.default = "iframe"
        fig.show()
        return fig

    def plot_interactive_scalability_analysis(run_sizes: list[int], sequential_times: list[float], parallel_times: list[float], filename: str = "scalability_analysis.html"):
        """
        Plots scalability analysis. 
        Parameters: 
        - run_sizes: for each value in this list, a sequential and a parallel evaluation will be run, with the given value as the number of runs that are executed in each evalualtion  
        - sequential_times: the runtimes of all the sequential runs
        - parallel_times: the runtimes of all the parallel runs
        - filename: optional filename, if provided the plot will be saved to a file of that name
        """
        speedups = [seq / par if par > 0 else 1.0 for seq, par in zip(sequential_times, parallel_times)]
        time_saved = [seq - par for seq, par in zip(sequential_times, parallel_times)]
        labels = [f"{n:,}" for n in run_sizes]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x = labels, 
                y = sequential_times, 
                name = "Sequential (Single-Core)", 
                marker_color = "#96A1A3", 
                customdata = list(zip(sequential_times, speedups, time_saved)), 
                hovertemplate = (
                    "<b>Runs:</b> %{x}<br>" + 
                    "<b>Execution Mode:</b> Sequential<br>" + 
                    "<b>Runtime</b> %{y:.3f}s<br>" + 
                    "<extra></extra>"
                )
            ), 
            secondary_y = False
        )

        fig.add_trace(
            go.Bar(
                x = labels, 
                y = parallel_times, 
                name = "Parallel (Multi-Core)", 
                marker_color = "#3395A6", 
                customdata = list(zip(parallel_times, speedups, time_saved)), 
                hovertemplate = (
                    "<b>Runs:</b> %{x}<br>" + 
                    "<b>Execution Mode:</b> Parallel<br>" + 
                    "<b>Runtime</b> %{y:.3f}s<br>" + 
                    "<b>Speedup:</b> %{customdata[1]:.2f}x faster<br>" + 
                    "<b>Time Saved:</b> %{customdata[2]:.3f}s<br>" + 
                    "<extra></extra>"
                )
            ), 
            secondary_y = False
        )

        fig.add_trace(
            go.Scatter(
                x = labels, 
                y = speedups, 
                name = "Parallel Speedup", 
                mode = "lines+markers", 
                line = dict(
                    color = "#F0861A", 
                    width = 3, 
                    dash = "dash"
                ), 
                marker = dict(
                    size = 8, 
                    symbol = "diamond"
                ), 
                hovertemplate = (
                    "<b>Runs:</b> %{x}<br>" + 
                    "<b>Speedup:</b> %{y:.2f}x<br>" + 
                    "<extra></extra>"
                )
            ),
            secondary_y = True
        )

        title = "<b>System Scalability Analysis</b>"

        fig.update_layout(
            title = title, 
            title_font = dict(
                size = 18, 
                family = "Arial", 
                color = "#222222"
            ), 
            template = "plotly_white", 
            barmode = "group", 
            hoverlabel = dict(
                bgcolor = "#FFFFFF", 
                font_size = 13, 
                font_family = "Arial"
            ), 
            margin = dict(l=50, r=50, t=100, b=50), 
            legend = dict(
                orientation = "h", 
                yanchor = "bottom", 
                y = 1.02, 
                xanchor = "right", 
                x = 1
            )
        )

        fig.update_xaxes(
            title_text = "Total Monte Carlo Runs", 
            title_font = dict(
                size = 13,
                family = "Arial"
            )
        )

        fig.update_yaxes(
            title_text = "Execution Time (Seconds)", 
            gridcolor = "#EBEBEB", 
            secondary_y = False
        )

        fig.update_yaxes(
            title_text = "Speedup Multiplier (x)", 
            showgrid = False, 
            secondary_y = True
        )

        fig.write_html(filename)
        pio.renderers.default = "iframe"
        fig.show()
        return fig

    def plot_interactive_frequent_states(query_metrics: dict | None, max_states_to_show: int = 10, filename: str = "frequent_states_chart.html"): 
        """
        Plots the x most frequented states (across all runs) and how often they were traversed. 
        Parameters: 
        - query_metrics: output of a query evaluation
        - max_states_to_show: int that decides how many of the most frequent states should be plotted 
        - filename: optional filename, if provided the plot will be saved to a file of that name
        """
        if query_metrics == None: 
            print("No metrics available to plot.")
            return 

        state_traversal_counts = query_metrics["state_traversal_counts"]
        sorted_states = sorted(state_traversal_counts.items(), key=lambda item: item[1], reverse=True)
        sorted_states = sorted_states[:max_states_to_show]

        states = [item[0] for item in sorted_states]
        run_counts = [item[1] for item in sorted_states]

        df = pd.DataFrame({
            "State Label": states, 
            "Frequency": run_counts
        })

        title = f"<b>Top {max_states_to_show} Most Frequented States Across all Runs</b>"

        fig = px.bar(
            df, 
            x = "Frequency", 
            y = "State Label", 
            orientation = "h", 
            title = title, 
            template = "plotly_white", 
            color_discrete_sequence = ["#3395A6"]
        )

        fig.update_traces(
            hovertemplate = (
                "<b>State:</b> %{y}<br>" + 
                "<b>Total Traversals:</b> %{x:.0f}<br>" + 
                "<extra></extra>"
            )
        )

        fig.update_layout(
            title_font = dict(
                size = 18, 
                family = "Arial", 
                color = "#222222"
            ), 
            xaxis = dict(
                title = "Total Traversals", 
                title_font = dict(
                    size = 13, 
                    family = "Arial"
                ), 
                gridcolor = "#EBEBEB"
            ),
            yaxis = dict(
                title = "States", 
                title_font = dict(
                    size = 13, 
                    family = "Arial"
                ), 
                gridcolor = "#EBEBEB", 
                autorange = "reversed", 
                ticksuffix = "   "
            ), 
            hoverlabel = dict(
                bgcolor = "#FFFFFF", 
                font_size = 13, 
                font_family = "Arial"
            ), 
            margin = dict(l=80, r=50, t=100, b=20)
        )

        fig.write_html(filename)
        pio.renderers.default = "iframe"
        fig.show()

        return fig

        

        
        

        
        
        
        
        






    