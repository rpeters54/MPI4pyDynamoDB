

from __future__ import annotations
from hashlib import md5
from mpi4py import MPI
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    NamedTuple,
    Union,
)

import time
import traceback
import random
import threading
import logging
import os

comm: MPI.Intracomm = MPI.COMM_WORLD
rank: int           = comm.Get_rank()
size: int           = comm.Get_size()

# Important Constants
FANOUT:                int   = 4
DEAD_TIMEOUT_S:        float = 1.0
CULL_TIMEOUT_S:        float = 2.0
HEARTBEAT_INTERVAL_S:  float = 0.25
THREAD_JOIN_TIMEOUT_S: float = 5.0



# MPI Message Tags
HEARTBEAT_TAG:         int   = 0
REQUEST_TAG:           int   = 1


# Data definition for database operation requests
class DBRequest(NamedTuple):
    source: int
    dest: int
    operation: str
    key: str
    value: T

class HeartbeatEntry:

    def __init__(self, counter: int=0, ticks: float=0, alive: bool=True):
        self.counter: int  = counter
        self.ticks: float  = time.time() if ticks == 0 else ticks
        self.alive: bool   = alive


    def beat(self):
        self.counter += 1
        self.ticks    = time.time()
        self.alive    = True


    def combine(self, other: HeartbeatEntry):
        if (other.counter > self.counter):
            self.counter = other.counter
            self.ticks   = other.ticks
            self.alive   = other.alive


class HeartbeatTable:

    def __init__(self, idx: int, num_nodes: int):
        self._table: List[Optional[HeartbeatEntry]] = num_nodes * [None]
        self._lock:  threading.Lock                 = threading.Lock()
        
        self._table[idx] = HeartbeatEntry()


    def update(self, other: List[HeartbeatEntry]):
        """ update the entries in a table with another table """
        with self._lock:
            for idx, entry in enumerate(other):
                if entry is None:
                    continue
                if self._table[idx] is None:
                    self._table[idx] = HeartbeatEntry()
                    self._table[idx].combine(entry)
                else:
                    self._table[idx].combine(entry)


    def entry_beat(self, idx: int):
        """ Have a single entry update its heartbeat """
        with self._lock:
            self._table[idx].beat()


    def entry_missing(self, idx: int) -> bool:
        with self._lock:
            return self._table[idx] is None


    def entry_alive(self, idx: int) -> bool:
        """ check if node is alive """
        with self._lock:
            return self._table[idx] is not None and self._table[idx].alive


    def mark_dead(self):
        """ read through list of entries, marking dead nodes and culling very dead nodes """
        current_time = time.time()

        with self._lock:    
            for idx, entry in enumerate(self._table):
                if entry is None:
                    continue
                time_since_update = current_time - entry.ticks
                if entry.alive and time_since_update > DEAD_TIMEOUT_S:
                    # Mark as dead if it hasn't been updated for mark_dead_threshold_ns
                    entry.alive = False
                elif not entry.alive and time_since_update > CULL_TIMEOUT_S:
                    self._table[idx] = None


    def get_live_nodes(self) -> List[int]:
        """Return a list of nodes that are currently alive"""
        with self._lock:
            return [idx for idx, entry in enumerate(self._table) if entry is not None and entry.alive]


    def get_table_snapshot(self) -> List[HeartbeatEntry]:
        """Get a thread-safe snapshot of the table for logging"""
        with self._lock:
            snapshot = []
            for entry in self._table:
                if entry is not None:
                    snapshot.append(HeartbeatEntry(entry.counter, entry.ticks, entry.alive))
                else:
                    snapshot.append(None)
            return snapshot


T = TypeVar('T')
class DynamoDB(Generic[T]):

    def __init__(self, log_dir: Optional[str] = None, seed: int = 42, ratio_vnode_to_node: int = 2):
        self.comm:              MPI.Intracomm              = comm
        self.rank:              int                        = rank
        self.size:              int                        = size
        self.data:              Dict[str, T]               = {}
        self.heartbeat_table:   HeartbeatTable             = HeartbeatTable(self.rank, self.size)
        self.neighbors:         List[int]                  = self._init_neighbors(FANOUT)
        self._stop_heartbeat:   bool                       = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._pending_sends:    List[MPI.Request]          = [] 
        self._mapping_table:    List[int]                  = self._create_vnode_table(size, seed, ratio_vnode_to_node)
        self.logger:            logging.Logger             = self._setup_logger(log_dir=log_dir)
        
        self._init_server()

    
    def _create_vnode_table(
        self,
        num_nodes: int,
        seed: int,
        ratio_vnode_to_node: int,
    ) -> List[int]:
        """Create a table that maps virtual node indices to real nodes"""
        # Set seed for consistent generation across all processes
        random.seed(seed)
        
        # Distribute virtual nodes as evenly as possible
        mapping_table = []
        for rank in range(num_nodes):
            mapping_table += [rank] * ratio_vnode_to_node
        
        # Shuffle to randomize the assignment while maintaining equal distribution
        random.shuffle(mapping_table)
        
        return mapping_table


    def _setup_logger(self, log_dir: Optional[str] = None) -> logging.Logger:
        """Setup a logger specific to this DynamoDB instance"""
        logger_name = f"DynamoDB_Node_{self.rank}"
        logger = logging.getLogger(logger_name)
        
        # Only add handler if it doesn't already exist (prevents duplicate logs)
        if not logger.handlers:
            if log_dir is not None:
                # Create directory if it doesn't exist
                os.makedirs(log_dir, exist_ok=True)
                
                # Create file handler for logging to file
                log_file = os.path.join(log_dir, f"node_{self.rank}.log")
                handler = logging.FileHandler(log_file, mode="w")
            else:
                # Fall back to console logging if no directory provided
                handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                f'%(asctime)s - Node {self.rank} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    

    def _init_neighbors(self, fanout: int) -> List[int]:
        """Determine what neighbors to gossip to"""
        # Create a list of all possible node IDs excluding 0 (client) and the current node
        valid_nodes = [i for i in range(0, self.size) if i != self.rank]
        
        # If requested fanout is larger than available nodes, cap it
        actual_fanout = min(fanout, len(valid_nodes))
        
        # Randomly select 'fanout' number of nodes from valid nodes
        return random.sample(valid_nodes, actual_fanout)


    def _init_server(self):
        """Initialize the server with a heartbeat thread"""
        
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self.logger.info("Server initialized with heartbeat thread")


    """
    Membership Functions
    """


    def _gossip(self):
        """Gossip heartbeat table to neighbors and update self"""

        # Clean up any pending sends
        self._cleanup_completed_sends()

        # Update heartbeat of node's entry
        self.heartbeat_table.entry_beat(self.rank)

        # Send table to all living neighbors
        snapshot = self.heartbeat_table.get_table_snapshot()
        for neighbor in self.neighbors:
            if self.heartbeat_table.entry_missing(neighbor) or self.heartbeat_table.entry_alive(neighbor):
                req = self.comm.isend(snapshot, dest=neighbor, tag=HEARTBEAT_TAG)
                self._pending_sends.append(req)


        status = MPI.Status()
        message_available = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=HEARTBEAT_TAG, status=status)
        while message_available:
            # recv next item
            source = status.Get_source()
            tag = status.Get_tag()
            other_table: List[HeartbeatEntry] = self.comm.recv(source=source, tag=HEARTBEAT_TAG)

            self.logger.debug(f"Got message from {source}")

            # update the table
            self.heartbeat_table.update(other_table)

            # probe the recv line for another message
            status = MPI.Status()
            message_available = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=HEARTBEAT_TAG, status=status)

        # mark nodes dead, and cull unavailable nodes
        self.heartbeat_table.mark_dead()


    def _heartbeat_loop(self):
        """Main heartbeat thread function that runs continuously"""
        self.logger.info("Heartbeat thread started")

        while not self._stop_heartbeat:
            try:
                self._gossip()
                # Sleep for the heartbeat interval
                time.sleep(HEARTBEAT_INTERVAL_S)
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                self.logger.debug(traceback.format_exc())
        self.logger.info("Heartbeat thread stopped")


    def stop(self):
        """Stop the heartbeat thread gracefully"""

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self.logger.info("Stopping heartbeat thread")
            self._stop_heartbeat = True
            self._heartbeat_thread.join(timeout=THREAD_JOIN_TIMEOUT_S)


    def _cleanup_completed_sends(self):
        """Clean up completed send requests"""
        completed = []
        for i, req in enumerate(self._pending_sends):
             # Do a non-blocking check to test for request completion
            if req.Test(): 
                completed.append(i)
        
        # Remove completed requests
        for i in reversed(completed):
            self._pending_sends.pop(i)

    """
    Client Interface Helper Functions
    """

    def _determine_coordinator_vnode(self, key: str) -> int:
        """Hash the key to find the coordinator it belongs to"""
        hash = md5(key.encode('UTF-8'))
        return hash % self.size
    

    def _get_live_node_by_vnode(
        self,
        vnode_idx: int,
    ) -> Optional[int]:
        """Try to get the node mapped by the vnode.
        if not alive, search sequentially for the next viable node"""
        
        original_vnode = vnode_idx

        alive = False
        while not alive:
            # lookup mapping
            node_idx = self._mapping_table[vnode_idx]

            # only check liveness if not self
            if node_idx != self.rank:
                alive = self.heartbeat_table.entry_alive(node_idx)
            
            # made it around the loop without success
            vnode_idx += 1
            if vnode_idx == original_vnode:
                return None

        return node_idx
        
    def _send_request_to_coordinator(self, request: DBRequest, timeout: float) -> bool:
        """send the request to the coordinator node"""
        try:
            # Use non-blocking send
            result = self.comm.isend(request, dest=request.dest, tag=REQUEST_TAG)
            
            start_time = time.time()
            while not result.test():
                if time.time() - start_time > timeout:
                    # Cancel the request if possible
                    try:
                        request.cancel()
                    except:
                        pass
                    self.logger.error(f"Coordinator Request timed out")   
                    return False
                
                # Small sleep to avoid busy waiting
                time.sleep(0.001)
            
            self.logger.info(f"Coordinator Request send successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending to node: {request.dest}")   
            return False


    """
    Client Interface Functions
    """


    def put(self, key: str, value: T) -> bool:
        self.logger.info(f"Preparing 'PUT' with key: {key}")
        # find the coordinator node to send to
        vnode_idx = self._determine_coordinator_vnode(key)
        node_idx = self._get_live_node_by_vnode(vnode_idx)
        if node_idx is None:
            self.logger.error(f"Failed to find valid coordinator for key: {key}")
            return False
        
        # format and send the message
        request = DBRequest(self.rank, node_idx, 'PUT', key, value)
        return self._send_request_to_coordinator(request, DEAD_TIMEOUT_S)
        
        
    def get(self, key: str) -> T:
        self.logger.info(f"Preparing 'GET' with key: {key}")
        # find the coordinator node to send to
        vnode_idx = self._determine_coordinator_vnode(key)
        node_idx = self._get_live_node_by_vnode(vnode_idx)
        if node_idx is None:
            self.logger.error(f"Failed to find valid coordinator for key: {key}")
            return False
        
        # format and send the message
        request = DBRequest(self.rank, node_idx, 'GET', key, None)
        return self._send_request_to_coordinator(request, DEAD_TIMEOUT_S)
