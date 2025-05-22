# MPI4pyDynamoDB

An implementation of DynamoDB in MPI4py

## Setup

This package requires the `poetry` dependency manager for python, an installation of python 3.10/3.11, and `open-mpi`

To install poetry:

> pip install poetry

If you don't have python 3.10/3.11 you can install them easily using https://github.com/pyenv/pyenv

Install open-mpi using your preferred package manager or directly here: https://www.open-mpi.org/

Once installed, run:

> cd dynamodb/

> poetry install

> poetry env activate

Your environment should now be setup to run the program

## Running

Update the value of `LOGGING_DIR` in `dynamodb/src/db_test.py` to point to the proper directory

Then, to test distributed system heartbeating, you can run:

> ./test_script.sh heartbeat

The output of the previous run for each node will appear in your `dynamodb/logs/` directory
