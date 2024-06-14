# Tensor network algorithm for optimizing control-enhanced quantum metrology
This repository contains the Python code and data accompanying the article "Efficient tensor networks for control-enhanced quantum metrology"([arXiv:2403.09519](https://arxiv.org/abs/2403.09519)). We implement the tensor network algorithm to efficiently maximize the quantum Fisher information (QFI) of the output state obtained by several types of control strategies. Our approach covers a general and practical scenario where the experimenter applies $N−1$ interleaved control operations between $N$ queries of the channel to estimate and uses no or bounded ancilla. 
## Requirements
The code for tensor contraction requires the Python package [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/). The code for SDP requires the Python package [CVXPY](https://www.cvxpy.org) with the optimizer [MOSEK](https://www.mosek.com). The code for variational circuit optimization requires the Python package [PennyLane](https://pennylane.ai/).
## Description
* `functions.py` contains useful functions for the setup.
* `arbitrary_control.py` constructs the problem of QFI optimization with arbitrary completely positive trace-preserving (CPTP) control operations.
* `identical_control.py` constructs the problem of QFI optimization with identical CPTP control operations.
* `unitary_control.py` constructs the problem of QFI optimization with variational unitary control operations.
* `identical_unitary_control.py` constructs the problem of QFI optimization with identical variational unitary control operations.
* `solve.py` provides two examples for applying the optimization algorithm.
