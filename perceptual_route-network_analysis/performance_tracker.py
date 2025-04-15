import time
import json
from datetime import datetime
import functools
import networkx as nx
import numpy as np

class PerformanceTracker:
    def __init__(self, output_file='performance_data.json'):
        self.output_file = output_file

        # Load existing data if file exists
        try:
            with open(output_file, 'r') as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = []

    def log_execution(self, function_name, execution_time, metrics=None,
                      params=None, instance_id=None):
        """
        Log function execution with multiple metrics

        Args:
            function_name: Name of the function
            execution_time: Time taken to execute (seconds)
            metrics: Dictionary of metrics (e.g., {'nodes': 100, 'edges': 250})
            params: Any parameters to log
            instance_id: Unique identifier for this run
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'execution_time': execution_time,
            'instance_id': instance_id
        }

        # Add metrics dictionary to log entry
        if metrics:
            log_entry.update(metrics)

        # Add parameters if provided
        if params:
            log_entry['params'] = params

        self.data.append(log_entry)

    def save(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f)


def track_performance(tracker, metrics_funcs=None):
    """
    Decorator for tracking performance with multiple metrics

    Args:
        tracker: PerformanceTracker instance
        metrics_funcs: Dictionary mapping metric names to functions that extract those metrics
                      from the first argument of the decorated function
                      Example: {'nodes': get_node_count, 'edges': get_edge_count}
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Collect metrics
            metrics = {}
            if metrics_funcs and args:
                for metric_name, metric_func in metrics_funcs.items():
                    try:
                        metrics[metric_name] = metric_func(args[0])
                    except Exception as e:
                        print(f"Warning: Failed to compute metric '{metric_name}': {e}")

            # Generate instance ID based on timestamp
            instance_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Time the function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log performance data
            tracker.log_execution(
                function_name=func.__name__,
                execution_time=execution_time,
                metrics=metrics,
                instance_id=instance_id
            )

            return result

        return wrapper

    return decorator

def get_node_count(graph):
    return len(graph.nodes)

def get_edge_count(graph):
    return len(graph.edges)

def get_avg_degree(graph):
    degrees = [d for _, d in graph.degree()]
    return sum(degrees) / len(degrees) if degrees else 0

def get_density(graph):
    return nx.density(graph)

def get_route_n_nodes(route):
    return len(route.nodes)