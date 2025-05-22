#!/bin/bash

# Check for test type argument
TEST_TYPE=${1:-basic}

# Define the number of processes based on test type
case $TEST_TYPE in 
  heartbeat)
    NUM_PROCESSES=6
    ;;
    # add more as needed
  *)
    echo "Unknown test type: $TEST_TYPE"
    echo "Valid types are: basic, recovery, multiple"
    exit 1
    ;;
esac

# Run the test with mpiexec
echo "Running $TEST_TYPE test with $NUM_PROCESSES processes..."
mpiexec -n $NUM_PROCESSES python3 db_test.py $TEST_TYPE