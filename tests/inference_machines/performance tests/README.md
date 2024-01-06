# Performance test

May be used to profile the inference machine

## Requirements

- cProfile
- snakeviz

## How to run

From the root folder, run:
```
python3 -m cProfile \
    -o results.prof \
    ./tests/inference_machines/performance\ tests/performance_test_mnist_inference_machine.py
snakeviz results.prof
```