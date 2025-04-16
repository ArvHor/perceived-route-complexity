import time
import json
from datetime import datetime
import functools
import networkx as nx


class PerformanceTracker:
    def __init__(self, output_file='performance_data.json'):
        self.output_file = output_file
        self.data = []

        # Try to load existing data
        try:
            with open(output_file, 'r') as f:
                self.data = json.load(f)
                print(f"Loaded {len(self.data)} existing entries from {output_file}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = []
            print(f"No existing data found. Will create new file: {output_file}")

    def log_execution(self, function_name, execution_time, metrics=None, instance_id=None, params=None):
        """
        Log function execution details and save to file
        """
        # Create the base log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'execution_time': execution_time,
            'instance_id': instance_id
        }

        # Important: Add metrics as TOP-LEVEL fields in the log entry
        # This is likely where your issue is
        if metrics:
            for key, value in metrics.items():
                log_entry[key] = value

        # Add any parameters if provided
        if params:
            log_entry['params'] = params

        # Add to the data list
        self.data.append(log_entry)

        # Save after each execution
        self.save()

        # Print confirmation for debugging
        print(f"Logged execution of '{function_name}' with metrics: {metrics}")

    def save(self):
        """Save data to JSON file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"Data saved to {self.output_file} ({len(self.data)} entries)")
        except Exception as e:
            print(f"ERROR saving data: {e}")


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
        print(
            f"Creating decorator for {func.__name__} with metrics: {list(metrics_funcs.keys()) if metrics_funcs else 'None'}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Detailed debugging of arguments
            print(f"Function {func.__name__} called with:")
            print(f"  - Positional args: {len(args)} {[type(a).__name__ for a in args]}")
            print(f"  - Keyword args: {len(kwargs)} {list(kwargs.keys())}")

            # metrics_funcs is in scope through closure
            computed_metrics = {}

            # First check positional args
            for metric_name, metric_func in metrics_funcs.items():
                if metric_name == 'n_count':
                    arg = args[1]
                    computed_metrics[metric_name] = metric_func(arg)
                else:
                    arg = args[0]
                    computed_metrics[metric_name] = metric_func(arg)

            # Generate ID and time execution
            run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            start = time.time()

            # Execute the original function
            result = func(*args, **kwargs)

            # Calculate execution time
            elapsed = time.time() - start

            # Log the execution with metrics
            print(f"Logging metrics: {computed_metrics}")
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

