# Performance test

May be used to profile the inference machine

## Requirements

- cProfile
- snakeviz

## How to run

From the current folder, run:
```
python3 -m cProfile -o results.prof performance_test_mnist_inference_machine_3.py
snakeviz results.prof
```