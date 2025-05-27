from mpi4py import MPI
import time
import sys
import os
import random

# Assuming db.py is in the same directory or Python path
from db import DynamoDB
# Import constants that might be dynamically set in DynamoDB.__init__
# These are global in your db.py. If they were instance variables, you'd access via db_instance.variable
from db import REPLICATION_FACTOR, WRITE_QUORUM_W, READ_QUORUM_R, DEAD_TIMEOUT_S, CULL_TIMEOUT_S

LOGGING_DIR = './logs_test' # Ensure this directory exists or is created by the logger

# Helper to ensure log directory exists
if MPI.COMM_WORLD.Get_rank() == 0:
    os.makedirs(LOGGING_DIR, exist_ok=True)
MPI.COMM_WORLD.Barrier()


LOGGING_DIR = '/Users/rileypeters/CSC569/dynamodb/logs'

def test_heartbeat():
    """Basic test to verify heartbeat functionality"""
    db = DynamoDB(LOGGING_DIR)
    rank = db.rank
    size = db.size
    
    comm = MPI.COMM_WORLD
    
    # Wait to give time for heartbeats to exchange
    time.sleep(5)

    # For server nodes: check the heartbeat table 
    live_nodes = db.heartbeat_table.get_live_nodes()
    db.logger.info(f"Node {rank}: Neighbors {db.neighbors}")
    db.logger.info(f"Node {rank}: Live nodes after 5s: {live_nodes}")
    
    # Print detailed heartbeat table info
    db.logger.info(f"Node {rank}: Heartbeat table details:")
    for idx, entry in enumerate(db.heartbeat_table.get_table_snapshot()):
        if entry is not None:
            db.logger.info(f"  Node {idx}: Counter={entry.counter}, Alive={entry.alive}, Time={entry.ticks}")
    
    # Barrier to make sure all nodes complete this part together
    comm.Barrier()
    
    # Simulate a node failure (only for testing, node 2 stops gossiping)
    if rank == 2 and size > 2:
        db.logger.info(f"Node {rank}: Simulating node failure (stopping heartbeat)")
        db.stop()
    
    # Wait longer for failure detection to happen
    time.sleep(10)
    
    # Check which nodes are now seen as alive
    if rank != 2:  # Still active servers check their tables
        live_nodes = db.heartbeat_table.get_live_nodes()
        db.logger.info(f"Node {rank}: Live nodes after failure: {live_nodes}")
        
        # If node 2 was correctly marked as dead, it shouldn't be in the live nodes list
        if size > 2:
            if 2 in live_nodes:
                db.logger.info(f"Node {rank}: FAIL - Node 2 still considered alive when it should be dead")
            else:
                db.logger.info(f"Node {rank}: PASS - Node 2 correctly identified as dead")

    # Print detailed heartbeat table info
    db.logger.info(f"Node {rank}: Heartbeat table details:")
    for idx, entry in enumerate(db.heartbeat_table.get_table_snapshot()):
        if entry is not None:
            db.logger.info(f"  Node {idx}: Counter={entry.counter}, Alive={entry.alive}, Time={entry.ticks}")
    
    # Clean up
    if rank != 2:  # Only active servers need cleanup
        db.stop()
    
    # Final barrier to ensure all tests complete before exiting
    comm.Barrier()
    return


def test_database_operations():
    """Test basic PUT and GET operations with replication."""
    db = DynamoDB(LOGGING_DIR)
    rank = db.rank
    size = db.size
    comm = MPI.COMM_WORLD

    if rank == 0: print("Starting database operations test.")
    db.logger.info(f"Node {rank}: Starting database operations test (N={REPLICATION_FACTOR}, W={WRITE_QUORUM_W}, R={READ_QUORUM_R}).")

    # Wait for all nodes to initialize and heartbeats to stabilize
    time.sleep(3)
    comm.Barrier()

    if rank == 0: print("Testing PUTs.")

    test_key_1 = f"testKey{rank}_1"
    test_value_1 = f"valueForNode{rank}" # Each node tries to write, coordinator should win

    test_key_2 = f"testKey{rank}_2"
    test_value_2 = [rank, rank+1, rank+2] # Each node tries to write, coordinator should win

    db.logger.info(f"Node {rank}: Attempting PUT for key '{test_key_1}' with value '{test_value_1}'")
    put_success_1 = db.put(test_key_1, test_value_1)
    if put_success_1:
        db.logger.info(f"Node 0: PUT for '{test_key_1}' reported SUCCESS.")
    else:
        db.logger.error(f"Node 0: PUT for '{test_key_1}' reported FAILED.")

    db.logger.info(f"Node {rank}: Attempting PUT for key '{test_key_2}' with value '{test_value_2}'")
    put_success_2 = db.put(test_key_2, test_value_2)
    if put_success_2:
        db.logger.info(f"Node 0: PUT for '{test_key_2}' reported SUCCESS.")
    else:
        db.logger.error(f"Node 0: PUT for '{test_key_2}' reported FAILED.")
    
    # Ensure PUTs are attempted before GETs
    comm.Barrier()

    if rank == 0: print("Testing GETs.")

    # All nodes attempt to GET
    retrieved_values_1 = db.get(test_key_1)
    if put_success_1: # Only check if Node 0's put was supposed to succeed
        if all(retrieved_value_1 == test_value_1 for retrieved_value_1 in retrieved_values_1):
            db.logger.info(f"Node {rank}: GET for '{test_key_1}' PASS. Value: '{retrieved_values_1}'")
        else:
            db.logger.error(f"Node {rank}: GET for '{test_key_1}' FAIL. Expected '{test_value_1}', Got '{retrieved_values_1}'")
    elif retrieved_values_1 is not None: # If PUT failed, GET should ideally also fail or return None
         db.logger.warning(f"Node {rank}: GET for '{test_key_1}' returned '{retrieved_values_1}' even though PUT might have failed.")

    retrieved_values_2 = db.get(test_key_2)
    if put_success_2: # Only check if Node 0's put was supposed to succeed
        if all(retrieved_value_2 == test_value_2 for retrieved_value_2 in retrieved_values_2):
            db.logger.info(f"Node {rank}: GET for '{test_key_2}' PASS. Value: '{retrieved_values_2}'")
        else:
            db.logger.error(f"Node {rank}: GET for '{test_key_2}' FAIL. Expected '{test_value_2}', Got '{retrieved_values_2}'")
    elif retrieved_values_2 is not None: # If PUT failed, GET should ideally also fail or return None
         db.logger.warning(f"Node {rank}: GET for '{test_key_2}' returned '{retrieved_values_2}' even though PUT might have failed.")

    # Complete all GETs before exiting
    comm.Barrier()
    db.stop()
    comm.Barrier()
    db.logger.info(db.store.data)
    if rank == 0: print("Database operations test completed.")


def test_database_operations_with_fault():
    """Test basic PUT and GET operations with replication."""
    db = DynamoDB(LOGGING_DIR)
    rank = db.rank
    size = db.size
    comm = MPI.COMM_WORLD

    if rank == 0: print("Starting database operations test.")
    db.logger.info(f"Node {rank}: Starting database operations test (N={REPLICATION_FACTOR}, W={WRITE_QUORUM_W}, R={READ_QUORUM_R}).")

    # Wait for all nodes to initialize and heartbeats to stabilize
    time.sleep(3)
    comm.Barrier()

    if rank == 0: print("Testing PUTs.")

    test_key_1 = f"testKey{rank}_1"
    test_value_1 = f"valueForNode{rank}" # Each node tries to write, coordinator should win

    test_key_2 = f"testKey{rank}_2"
    test_value_2 = [rank, rank+1, rank+2] # Each node tries to write, coordinator should win

    db.logger.info(f"Node {rank}: Attempting PUT for key '{test_key_1}' with value '{test_value_1}'")
    put_success_1 = db.put(test_key_1, test_value_1)
    if put_success_1:
        db.logger.info(f"Node 0: PUT for '{test_key_1}' reported SUCCESS.")
    else:
        db.logger.error(f"Node 0: PUT for '{test_key_1}' reported FAILED.")

    db.logger.info(f"Node {rank}: Attempting PUT for key '{test_key_2}' with value '{test_value_2}'")
    put_success_2 = db.put(test_key_2, test_value_2)
    if put_success_2:
        db.logger.info(f"Node 0: PUT for '{test_key_2}' reported SUCCESS.")
    else:
        db.logger.error(f"Node 0: PUT for '{test_key_2}' reported FAILED.")
    
    # Ensure PUTs are attempted before GETs
    comm.Barrier()

    if rank == 0: print("Testing GETs.")

    # Simulate a node failure (only for testing, node 2 stops gossiping)
    if rank == 2 and size > 2:
        db.logger.info(f"Node {rank}: Simulating node failure (stopping heartbeat)")
        db.stop()
    else:
        # All nodes attempt to GET
        retrieved_values_1 = db.get(test_key_1)
        if retrieved_values_1 is not None and put_success_1: # Only check if Node 0's put was supposed to succeed
            if all(retrieved_value_1 == test_value_1 for retrieved_value_1 in retrieved_values_1):
                db.logger.info(f"Node {rank}: GET for '{test_key_1}' PASS. Value: '{retrieved_values_1}'")
            else:
                db.logger.error(f"Node {rank}: GET for '{test_key_1}' FAIL. Expected '{test_value_1}', Got '{retrieved_values_1}'")
        elif retrieved_values_1 is not None: # If PUT failed, GET should ideally also fail or return None
            db.logger.warning(f"Node {rank}: GET for '{test_key_1}' returned '{retrieved_values_1}' even though PUT might have failed.")
        else:
            db.logger.warning(f"Node {rank}: GET for '{test_key_2}' returned None")

        retrieved_values_2 = db.get(test_key_2)
        if retrieved_values_2 is not None and put_success_2: # Only check if Node 0's put was supposed to succeed
            if all(retrieved_value_2 == test_value_2 for retrieved_value_2 in retrieved_values_2):
                db.logger.info(f"Node {rank}: GET for '{test_key_2}' PASS. Value: '{retrieved_values_2}'")
            else:
                db.logger.error(f"Node {rank}: GET for '{test_key_2}' FAIL. Expected '{test_value_2}', Got '{retrieved_values_2}'")
        elif retrieved_values_2 is not None: # If PUT failed, GET should ideally also fail or return None
            db.logger.warning(f"Node {rank}: GET for '{test_key_2}' returned '{retrieved_values_2}' even though PUT might have failed.")
        else:
            db.logger.warning(f"Node {rank}: GET for '{test_key_2}' returned None")

    # Complete all GETs before exiting
    db.stop()
    comm.Barrier()
    if rank == 0: print("Database operations test completed.")


if __name__ == "__main__":
    comm_main = MPI.COMM_WORLD
    rank_main = comm_main.Get_rank()

    # Ensure LOGGING_DIR exists (again, in case it's used by __main__ directly)
    if rank_main == 0:
        os.makedirs(LOGGING_DIR, exist_ok=True)
    comm_main.Barrier()

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if rank_main == 0: print(f"Executing Test: {test_name}")
        comm_main.Barrier()

        if test_name == "heartbeat":
            test_heartbeat()
        elif test_name == "db_ops":
            test_database_operations()
        elif test_name == "db_ops_w_fail":
             test_database_operations_with_fault()
        else:
            if rank_main == 0: print(f"Unknown test: {test_name}")