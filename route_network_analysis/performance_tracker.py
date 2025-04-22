import time
import json
from datetime import datetime
import functools
import networkx as nx


class PerformanceTracker:
    def __init__(self, output_file='performance_data.json'):
        self.output_file = output_file
        self.data = []

        try:
            with open(output_file, 'r') as f:
                self.data = json.load(f)
                #print(f"Loaded {len(self.data)} existing entries from {output_file}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = []
            #print(f"No existing data found. Will create new file: {output_file}")

    def log_execution(self, function_name, execution_time, metrics=None, instance_id=None, params=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'execution_time': execution_time,
            'instance_id': instance_id
        }

        if metrics:
            for key, value in metrics.items():
                log_entry[key] = value

        if params:
            log_entry['params'] = params

        self.data.append(log_entry)

        self.save()


    def save(self):
        try:
            import os
            abs_path = os.path.abspath(self.output_file)

            directory = os.path.dirname(abs_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(abs_path, 'w') as f:
                json.dump(self.data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            #print(f"Data saved to {abs_path} ({len(self.data)} entries)")
        except Exception as e:
            print(f"ERROR saving data: {str(e)}")
            import os
            #print(f"Attempted to save to: {os.path.abspath(self.output_file)}")
            #print(f"Current working directory: {os.getcwd()}")


def track_performance(tracker, metrics_funcs=None):
    """
    Create a decorator that tracks performance metrics with robust args handling
    """
    # Validate inputs first
    if not isinstance(tracker, PerformanceTracker):
        raise TypeError("tracker must be a PerformanceTracker instance")

    if metrics_funcs is not None and not isinstance(metrics_funcs, dict):
        raise TypeError("metrics_funcs must be a dictionary")

    # Create the actual decorator
    def actual_decorator(func):
        #print(f"Creating decorator for {func.__name__} with metrics: {list(metrics_funcs.keys()) if metrics_funcs else 'None'}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Detailed debugging of arguments
            #print(f"Function {func.__name__} called with:")
            #print(f"  - Positional args: {len(args)} {[type(a).__name__ for a in args]}")
            #print(f"  - Keyword args: {len(kwargs)} {list(kwargs.keys())}")

            computed_metrics = {}

            # More robust handling of graph argument

            graph_arg = None
            array_type_arg = None
            # First check positional args

            if "city_name" in list(metrics_funcs.keys()):
                if len(args) > 0:
                    graph_arg = args[0]
                    #print(f"Graph type in decorator: {type(graph_arg).__name__}")
                    #print(f"Is directed: {graph_arg.is_directed() if hasattr(graph_arg, 'is_directed') else 'Unknown'}")
                elif "G" in kwargs:
                    graph_arg = kwargs["G"]
                    #print(f"Graph type in decorator (from kwargs): {type(graph_arg).__name__}")
                    #print(f"Is directed: {graph_arg.is_directed() if hasattr(graph_arg, 'is_directed') else 'Unknown'}")

            if "n_count" in list(metrics_funcs.keys()):
                if len(args) > 1:
                    array_type_arg = args[1]
                elif "route_edges" in kwargs:
                    array_type_arg = kwargs["route_edges"]

            # Check if we found a graph and have metrics to compute
            if graph_arg is not None and metrics_funcs:
                #print(f"Computing metrics using graph found")

                # Compute metrics using the identified graph
                for metric_name, metric_func in metrics_funcs.items():
                    if metric_name == "n_count":
                        metric_value = len(array_type_arg)
                        computed_metrics[metric_name] = metric_value
                        #print(f"Computed metric: {metric_name} = {metric_value}")
                        continue
                        metric_value = metric_func(graph_arg)
                        computed_metrics[metric_name] = metric_value
                        #print(f"Computed metric: {metric_name} = {metric_value}")
            else:
                reasons = []
                if not graph_arg:
                    reasons.append("no graph found")
                if not metrics_funcs:
                    reasons.append("no metrics provided")
                #print(f"No metrics computed: {', '.join(reasons)}")

            # Generate ID and time execution
            run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            start = time.time()

            # Execute the original function
            result = func(*args, **kwargs)

            # Calculate execution time
            elapsed = time.time() - start

            # Log the execution with metrics
            #print(f"Logging metrics: {computed_metrics}")
            tracker.log_execution(
                function_name=func.__name__,
                execution_time=elapsed,
                metrics=computed_metrics,
                instance_id=run_id
            )

            return result

        return wrapper

    # Return the decorator function
    return actual_decorator
