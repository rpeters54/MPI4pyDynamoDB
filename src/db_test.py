from mpi4py import MPI
import time
import sys
import os

from db import DynamoDB, HeartbeatTable, HeartbeatEntry

LOGGING_DIR = '/Users/rileypeters/CSC569/dynamodb/logs/bingo'

def test_heartbeat():
    """Basic test to verify heartbeat functionality"""
    db = DynamoDB(LOGGING_DIR)
    rank = db.rank
    size = db.size
    
    comm = MPI.COMM_WORLD
    
    # Wait to give time for heartbeats to exchange
    time.sleep(5)

    # For server nodes: check the heartbeat table 
    live_nodes = db.table.get_live_nodes()
    db.logger.info(f"Node {rank}: Neighbors {db.neighbors}")
    db.logger.info(f"Node {rank}: Live nodes after 5s: {live_nodes}")
    
    # Print detailed heartbeat table info
    db.logger.info(f"Node {rank}: Heartbeat table details:")
    for idx, entry in enumerate(db.table.get_table_snapshot()):
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
        live_nodes = db.table.get_live_nodes()
        db.logger.info(f"Node {rank}: Live nodes after failure: {live_nodes}")
        
        # If node 2 was correctly marked as dead, it shouldn't be in the live nodes list
        if size > 2:
            if 2 in live_nodes:
                db.logger.info(f"Node {rank}: FAIL - Node 2 still considered alive when it should be dead")
            else:
                db.logger.info(f"Node {rank}: PASS - Node 2 correctly identified as dead")

    # Print detailed heartbeat table info
    db.logger.info(f"Node {rank}: Heartbeat table details:")
    for idx, entry in enumerate(db.table.get_table_snapshot()):
        if entry is not None:
            db.logger.info(f"  Node {idx}: Counter={entry.counter}, Alive={entry.alive}, Time={entry.ticks}")
    
    # Clean up
    if rank != 2:  # Only active servers need cleanup
        db.stop()
    
    # Final barrier to ensure all tests complete before exiting
    comm.Barrier()
    return



if __name__ == "__main__":
    # Check for test argument
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "heartbeat":
            test_heartbeat()
        else:
            print(f"Unknown test: {test_name}")
    else:
        # Run basic test by default
        test_heartbeat()
    
    # Wait a moment before exiting to allow for final cleanup
    time.sleep(1)