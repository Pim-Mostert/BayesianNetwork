# Bayesian Network

Todo:
- [x] Better logger for optimizer 
- [x] "Average" log-likelihood
  - [x] Check whether it's actually useful in experiment
  - [x] Probably make it a constructor configuration, 
    - [x] Also adjust v1, v2 and tests
- [ ] Unit test for BatchOptimizer (on evaluation set it should always increase)
  - [x] Implement Evaluator class, passed into constructor of optimizer. It takes the eval_data and setting to run evaluation every X iterations. Also requires an inference machine, naturally.
- [ ] Override `Evidence` for Lazy MNIST loader
