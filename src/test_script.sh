#!/bin/bash

# Default to running all tests if no argument is provided
TEST_TYPE=${1:-all}

# Define the number of processes and Python script argument based on test type
case $TEST_TYPE in
  heartbeat)
    NUM_PROCESSES=6   # As originally specified
    PYTHON_ARG=$TEST_TYPE
    ;;
  db_ops)
    NUM_PROCESSES=6
    PYTHON_ARG=$TEST_TYPE
    ;;
  db_ops_w_fail)
    NUM_PROCESSES=6 
    PYTHON_ARG=$TEST_TYPE
    ;;
  *)
    echo "Unknown test type: $TEST_TYPE"
    echo "Valid types are: all, heartbeat, db_ops, fault_tolerance"
    exit 1
    ;;
esac


echo "Running $TEST_TYPE test with $NUM_PROCESSES processes..."

# The $PYTHON_ARG will be empty for 'all', causing db_test.py to run all its internal tests.
# Otherwise, it passes the specific test name.
mpiexec -n $NUM_PROCESSES python3 db_test.py $PYTHON_ARG