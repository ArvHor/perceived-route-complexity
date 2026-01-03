
## Todo

- Figure out why the cardinal direction of OD-pairs does not work correctly.
- Only get street-network-based properties for OD-pairs and routes that have
been added to the underlying graph. E.g., The instruction equivalence of route
nodes should only be acquired if the instruction equivalence weights have been
added to the graph. Otherwise, confusing zeroes are added to the data. 
