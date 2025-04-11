# Bayesian Network

Todo:
- [x] Better logger for optimizer 
- [x] "Average" log-likelihood
  - [ ] Check whether it's actually useful in experiment
  - [ ] Probably make it a constructor configuration, rather than property on `log_likelihood` method
- [ ] Unit test for BatchOptimizer (on evaluation set it should always increase)
- [ ] Override `Evidence` for Lazy MNIST loader
