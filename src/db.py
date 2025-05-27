from __future__ import annotations
from hashlib import md5
from mpi4py import MPI
from typing import (
    Any,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    NamedTuple,
    Set,
)
from numpy.typing import NDArray
from collections import deque
from msg_helper import *

import numpy as np
import time
import traceback
import random
import threading
import logging
import os

comm: MPI.Intracomm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()

# Important Constants
FANOUT:                int   = 4
DEAD_TIMEOUT_S:        float = 1.0
CULL_TIMEOUT_S:        float = 2.0
HEARTBEAT_INTERVAL_S:  float = 0.25
THREAD_JOIN_TIMEOUT_S: float = 5.0
MSG_TIMEOUT_S:         float = 1.0

# Replication and Quorum Constants
REPLICATION_FACTOR: int = 3
REPLICATION_FACTOR = min(REPLICATION_FACTOR, size)
REPLICATION_FACTOR = max(1, REPLICATION_FACTOR)

WRITE_QUORUM_W: int = (REPLICATION_FACTOR // 2) + 1 if REPLICATION_FACTOR > 0 else 0
READ_QUORUM_R: int = (REPLICATION_FACTOR // 2) + 1 if REPLICATION_FACTOR > 0 else 0


# MPI Message Tags
HEARTBEAT_TAG:        int = 0
CLIENT_REQUEST_TAG:   int = 1
CLIENT_RESPONSE_TAG:  int = 2
REPLICA_REQUEST_TAG:  int = 3
REPLICA_RESPONSE_TAG: int = 4

def gen_tag(base_tag: int, rank: int) -> int:
    return base_tag * size + rank


# Generic TypeVar
T = TypeVar('T')

class DBOperation:
    CLIENT_PUT:           str = "CLIENT_PUT"
    CLIENT_GET:           str = "CLIENT_GET"
    REPLICA_CONFIRM_PUT:  str = "REPLICA_CONFIRM_PUT"
    REPLICA_COMMIT_PUT:   str = "REPLICA_COMMIT_PUT"
    REPLICA_GET:          str = "REPLICA_GET"

class DBOperationStatus:
    SUCCESS: str = "SUCCESS"
    FAILURE: str = "FAILURE"


# Data definition for database operation requests
class DBRequest(NamedTuple, Generic[T]):
    source: int
    dest: int
    operation: DBOperation
    key: str
    value: Optional[T]

# Data definition for database operation responses
class DBResponse(NamedTuple, Generic[T]):
    source_rank: int
    operation_status: DBOperationStatus 
    key: str
    value: List[T]          

class DBStore(NamedTuple, Generic[T]):
    data: Dict[str, T]   = {}
    lock: threading.Lock = threading.Lock()

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
        self._table: NDArray[np.object_] = np.full(num_nodes, None, dtype=object)
        self._lock:  threading.Lock      = threading.Lock()
        
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
        self.store:             DBStore                    = DBStore()
        self.heartbeat_table:   HeartbeatTable             = HeartbeatTable(self.rank, self.size)
        self.neighbors:         NDArray[np.int64]          = self._init_neighbors(FANOUT)
        self._stop_flag:        bool                       = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._client_thread:    Optional[threading.Thread] = None
        self._helper_thread:    Optional[threading.Thread] = None
        self._pending_sends:    deque[MPI.Request]         = deque()

        actual_ratio = max(1, ratio_vnode_to_node)
        if self.size > 0:
            self._mapping_table: NDArray[np.int64]         = self._create_vnode_table(self.size, seed, actual_ratio)
        else:
            self._mapping_table: NDArray[np.int64]         = np.array([], dtype=np.int64)
            global REPLICATION_FACTOR, WRITE_QUORUM_W, READ_QUORUM_R # Update globals if size is 0
            REPLICATION_FACTOR = 0
            WRITE_QUORUM_W = 0
            READ_QUORUM_R = 0
        self.logger:            logging.Logger             = self._setup_logger(log_dir=log_dir)
        
        self.logger.info(f"Node {self.rank} initialized. N={REPLICATION_FACTOR}, W={WRITE_QUORUM_W}, R={READ_QUORUM_R}")

        self._init_server()

    
    def _create_vnode_table(
        self,
        num_nodes: int,
        seed: int,
        ratio_vnode_to_node: int,
    ) -> NDArray[np.int64]:
        """Create a table that maps virtual node indices to real nodes"""
        # Set seed for consistent generation across all processes
        np.random.seed(seed)
        
        # Distribute virtual nodes as evenly as possible
        mapping_table = np.zeros(num_nodes * ratio_vnode_to_node, dtype=np.int64)
        for rank in range(num_nodes):
            base = rank * ratio_vnode_to_node
            for i in range(base, base + ratio_vnode_to_node):
                mapping_table[i] = rank
        
        # Shuffle to randomize the assignment while maintaining equal distribution
        np.random.shuffle(mapping_table)
        
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
    

    def _init_neighbors(self, fanout: int) -> NDArray[np.int64]:
        """Determine what neighbors to gossip to"""
        # Create a list of all possible node IDs excluding 0 (client) and the current node
        valid_nodes = [i for i in range(0, self.size) if i != self.rank]
        
        # If requested fanout is larger than available nodes, cap it
        actual_fanout = min(fanout, len(valid_nodes))
        
        # Randomly select 'fanout' number of nodes from valid nodes
        return np.array(random.sample(valid_nodes, actual_fanout))


    def _init_server(self):
        """Initialize the server with threads"""
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self.logger.info("Server initialized with heartbeat thread")

        self._client_thread = threading.Thread(target=self._client_loop, daemon=True)
        self._client_thread.start()
        self.logger.info("Server initialized with client thread")

        self._helper_thread = threading.Thread(target=self._helper_loop, daemon=True)
        self._helper_thread.start()
        self.logger.info("Server initialized with helper thread")


    def stop(self):
        """Stop the threads gracefully"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self.logger.info("Stopping heartbeat thread")
            self._stop_flag = True
            self._heartbeat_thread.join(timeout=THREAD_JOIN_TIMEOUT_S)

        if self._client_thread and self._client_thread.is_alive():
            self.logger.info("Stopping database thread")
            self._stop_flag = True
            self._client_thread.join(timeout=THREAD_JOIN_TIMEOUT_S)

        if self._helper_thread and self._helper_thread.is_alive():
            self.logger.info("Stopping database thread")
            self._stop_flag = True
            self._helper_thread.join(timeout=THREAD_JOIN_TIMEOUT_S)



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

        while not self._stop_flag:
            try:
                self._gossip()
                # Sleep for the heartbeat interval
                time.sleep(HEARTBEAT_INTERVAL_S)
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                self.logger.debug(traceback.format_exc())
        self.logger.info("Heartbeat thread stopped")


    def _cleanup_completed_sends(self):
        """Clean up completed send requests"""
        completed = []
        for i, req in enumerate(self._pending_sends):
             # Do a non-blocking check to test for request completion
            if req.Test(): 
                completed.append(i)
        
        # Remove completed requests
        for i in reversed(completed):
            self._pending_sends.pop()

    """
    Client Interface Helper Functions
    """

    def _determine_coordinator_vnode(self, key: str) -> int:
        hash_val = int(md5(key.encode('UTF-8')).hexdigest(), 16)
        num_vnodes = len(self._mapping_table)
        if num_vnodes == 0:
            self.logger.warning("Mapping table is empty! Cannot determine coordinator vnode.")
            return 0 # Fallback or raise error; assumes at least one node if called
        return hash_val % num_vnodes

    def _get_preference_list(self, key: str, num_target_replicas: int) -> List[int]:
        """Gets N distinct live physical nodes for a key's preference list."""
        preference_list: List[int] = []
        if num_target_replicas == 0 or len(self._mapping_table) == 0:
            return preference_list

        start_vnode_idx = self._determine_coordinator_vnode(key)
        num_vnodes = len(self._mapping_table)
        
        # Iterate through vnodes to find distinct, live physical nodes
        # might need to loop more than num_vnodes times if there are many dead nodes
        max_attempts = num_vnodes * 2 
        for i in range(max_attempts):
            current_vnode_idx = (start_vnode_idx + i) % num_vnodes
            physical_node_rank = self._mapping_table[current_vnode_idx]
            
            if self.heartbeat_table.entry_alive(physical_node_rank) and physical_node_rank not in preference_list:
                preference_list.append(physical_node_rank)
                if len(preference_list) == num_target_replicas:
                    break
        
        if len(preference_list) < num_target_replicas:
            self.logger.warning(f"Key '{key}': Found only {len(preference_list)}/{num_target_replicas} live distinct replicas.")
        
        return preference_list


    def _client_loop(self):
        while not self._stop_flag:
            request: Optional[DBRequest] = recv_with_timeout(comm, source=MPI.ANY_SOURCE, tag=gen_tag(CLIENT_REQUEST_TAG, self.rank), timeout=MSG_TIMEOUT_S)
            if self._stop_flag: break
            if request is not None:
                self.logger.debug(f"Node {self.rank} received Client Request: {request.operation} for key '{request.key}' from {request.source}")

            if request is None:
                self.logger.warning(f"Client thread timed out (may happen when db.Stop() is called)")
            elif request.operation == 'CLIENT_PUT':
                self._client_put(request)
            elif request.operation == 'CLIENT_GET':
                self._client_get(request)       
            else:
                self.logger.warning(f"Unknown operation: {request.operation}")
                if hasattr(request, 'original_client_rank'):
                        err_resp = DBResponse(self.rank, 'FAILURE', request.key, [])
                        self.comm.send(err_resp, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source))

        self.logger.info("Client thread stopped.")


    def _helper_loop(self):
        thread_pool: deque[threading.Thread] = deque()
      
        while not self._stop_flag:
            request: Optional[DBRequest] = recv_with_timeout(comm, source=MPI.ANY_SOURCE, tag=gen_tag(REPLICA_REQUEST_TAG, self.rank), timeout=MSG_TIMEOUT_S)
            if self._stop_flag: break
            if request is not None:
                self.logger.debug(f"Node {self.rank} received Replica Request: {request.operation} for key '{request.key}' from {request.source}")

            if request is None:
                self.logger.warning(f"Helper thread timed out (may happen when db.Stop() is called)")
            elif request.operation == 'REPLICA_CONFIRM_PUT':
                t = threading.Thread(target=self._replica_confirm_put, args=(request,))
                t.start()
                thread_pool.append(t)
            elif request.operation == 'REPLICA_COMMIT_PUT':
                t = threading.Thread(target=self._replica_commit_put, args=(request,))
                t.start()
                thread_pool.append(t)
            elif request.operation == 'REPLICA_GET':
                t = threading.Thread(target=self._replica_get, args=(request,))
                t.start()
                thread_pool.append(t)
            else:
                self.logger.warning(f"Unknown operation: {request.operation}")
                if hasattr(request, 'original_client_rank'):
                        err_resp = DBResponse(self.rank, 'FAILURE', request.key, [])
                        self.comm.send(err_resp, dest=request.source, tag=gen_tag(REPLICA_RESPONSE_TAG,request.source))
            
            # clean up completed threads
            temp_pool: deque[threading.Thread] = deque()
            for t in thread_pool:
                if t.is_alive():
                    temp_pool.append(t)
                else:
                    t.join()
            thread_pool = temp_pool

        self.logger.info("Helper thread stopped.")     


    def _client_put(self, request: DBRequest[T]):
        self.logger.info(f"Client {self.rank} initiating 'PUT' for key: {request.key}")
        
        # get list of replicas or return failure if impossible
        pref_list: List[int] = self._get_preference_list(request.key, REPLICATION_FACTOR)
        if (len(pref_list) < REPLICATION_FACTOR):
            err_resp = DBResponse(request.dest, 'FAILURE', request.key, [])
            send_with_timeout(comm, err_resp, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)
            return
        self.logger.info(f"Client {self.rank} preference list for 'PUT': {pref_list}")
        
        # ignore the first value since that is the current node
        pref_list = pref_list[1:]

        # send all the confirm requests
        confirm_packets: List[SendSpec] = []
        for node in pref_list:
            confirm_packets.append(SendSpec(
                data=DBRequest(self.rank, node, "REPLICA_CONFIRM_PUT", request.key, request.value), 
                dest=node, 
                tag=gen_tag(REPLICA_REQUEST_TAG, node)
            ))
        retvals = send_all_with_timeout(comm, confirm_packets, MSG_TIMEOUT_S)

        # cull failed sends
        remaining_nodes: List[int] = []
        for i, retval in enumerate(retvals):
            if retval:
                remaining_nodes.append(pref_list[i])

        # get the responses
        received_packets: List[RecvSpec] = []
        for node in remaining_nodes:
            received_packets.append(RecvSpec(
                source=node, 
                tag=gen_tag(REPLICA_RESPONSE_TAG, self.rank)
            ))
        confirm_receives: List[DBResponse] = recv_all_with_timeout(comm, received_packets, MSG_TIMEOUT_S)

        # tally acks
        count = 1
        for received in confirm_receives:
            if received is not None and (received.operation_status == "SUCCESS"):
                count += 1

        # if quorum not met, return failure
        if count < WRITE_QUORUM_W:
            err_resp = DBResponse(request.dest, 'FAILURE', request.key, [])
            send_with_timeout(comm, err_resp, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)
            return       

        # send all the commit requests
        commit_packets: List[SendSpec] = []
        for node in pref_list:
            commit_packets.append(SendSpec(
                data=DBRequest(self.rank, node, "REPLICA_COMMIT_PUT", request.key, request.value), 
                dest=node, 
                tag=gen_tag(REPLICA_REQUEST_TAG, node)
            ))
        retvals = send_all_with_timeout(comm, commit_packets, MSG_TIMEOUT_S)

        # cull failed sends
        remaining_nodes: List[int] = []
        for i, retval in enumerate(retvals):
            if retval:
                remaining_nodes.append(pref_list[i])  

        # get the responses
        received_packets: List[RecvSpec] = []
        for node in remaining_nodes:
            received_packets.append(RecvSpec(
                source=node, 
                tag=gen_tag(REPLICA_RESPONSE_TAG, self.rank)
            ))
        confirm_receives: List[DBResponse] = recv_all_with_timeout(comm, received_packets, MSG_TIMEOUT_S)

        # replicate value in own database
        self.store.lock.acquire()
        self.store.data[request.key] = request.value
        self.store.lock.release()
        response = DBResponse(request.dest, 'SUCCESS', request.key, [])
        send_with_timeout(comm, response, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)

              
    def _replica_confirm_put(self, request: DBRequest[T]):
        """Confirm to the requester that the node is alive and ready"""

        self.logger.info(f"Client {self.rank} received confirm put from {request.source}")

        # reply success on message received
        response = DBResponse(request.dest, 'SUCCESS', request.key, [None])
        send_with_timeout(comm, response, dest=request.source, tag=gen_tag(REPLICA_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)


    def _replica_commit_put(self, request: DBRequest[T]):
        """Commit the value and proceed"""

        self.logger.info(f"Helper {self.rank} received commit request from {request.source} to replicate {request.value} for key {request.key}")

        # commit value to db
        self.store.lock.acquire()
        self.store.data[request.key] = request.value
        self.logger.info(f"database state: {self.store.data}")
        self.store.lock.release()

        # respond successfully
        response = DBResponse(request.dest, 'SUCCESS', request.key, [None])
        send_with_timeout(comm, response, dest=request.source, tag=gen_tag(REPLICA_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)


    def _client_get(self, request: DBRequest[T]):
        self.logger.info(f"Client {self.rank} initiating 'GET' for key: {request.key}")
        
        # get list of replicas or return failure if impossible
        pref_list: List[int] = self._get_preference_list(request.key, REPLICATION_FACTOR)
        if (len(pref_list) < REPLICATION_FACTOR):
            err_resp = DBResponse(request.dest, 'FAILURE', request.key, [])
            send_with_timeout(comm, err_resp, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)
            return
        
        self.logger.info(f"Client {self.rank} preference list for 'GET': {pref_list}")
        
        # ignore the first value since that is the current node
        pref_list = pref_list[1:]

        # send all the confirm requests
        confirm_packets: List[SendSpec] = []
        for node in pref_list:
            confirm_packets.append(SendSpec(
                data=DBRequest(self.rank, node, "REPLICA_GET", request.key, None), 
                dest=node, 
                tag=gen_tag(REPLICA_REQUEST_TAG, node)
            ))
        retvals = send_all_with_timeout(comm, confirm_packets, MSG_TIMEOUT_S)

        # cull failed sends
        remaining_nodes: List[int] = []
        for i, retval in enumerate(retvals):
            if retval:
                remaining_nodes.append(pref_list[i])

        # get the responses
        received_packets: List[RecvSpec] = []
        for node in remaining_nodes:
            received_packets.append(RecvSpec(
                source=node, 
                tag=gen_tag(REPLICA_RESPONSE_TAG, self.rank)
            ))
        confirm_receives: List[DBResponse] = recv_all_with_timeout(comm, received_packets, MSG_TIMEOUT_S)

        # tally acks
        values: List[T] = []
        count:  int    = 0

        self.store.lock.acquire()
        local_value = self.store.data.get(request.key)
        self.logger.info(f"for client local lookup retrieved: {local_value}")
        self.store.lock.release()

        if local_value is not None:
            values.append(local_value)
            count += 1
        for received in confirm_receives:
            if received is not None and (received.operation_status == "SUCCESS"):
                values.append(received.value[0])
                count += 1 

        if (count < READ_QUORUM_R):
            err_resp = DBResponse(request.dest, 'FAILURE', request.key, [])
            send_with_timeout(comm, err_resp, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)
        else:
            response = DBResponse(request.dest, 'SUCCESS', request.key, values)
            send_with_timeout(comm, response, dest=request.source, tag=gen_tag(CLIENT_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)


    def _replica_get(self, request: DBRequest[T]):
        """Get value by key, or return error on failure"""

        self.store.lock.acquire()
        local_value = self.store.data.get(request.key)
        self.store.lock.release()

        self.logger.info(f"Helper {self.rank} received get request from {request.source} to return {local_value} for key {request.key}")

        if local_value is None:
            err_resp = DBResponse(request.dest, 'FAILURE', request.key, [])
            send_with_timeout(comm, err_resp, dest=request.source, tag=gen_tag(REPLICA_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)
        else:
            response = DBResponse(request.dest, 'SUCCESS', request.key, [local_value])
            send_with_timeout(comm, response, dest=request.source, tag=gen_tag(REPLICA_RESPONSE_TAG, request.source), timeout=MSG_TIMEOUT_S)


    # --- Client Interface Functions ---
    def put(self, key: str, value: T) -> bool:
        self.logger.info(f"Client {self.rank} initiating 'PUT' for key: {key}")
        
        temp_pref_list = self._get_preference_list(key, 1)
        if not temp_pref_list:
            self.logger.error(f"PUT key '{key}': No live node found to act as operation coordinator.")
            return False
        operation_coordinator_rank = temp_pref_list[0]


        client_put_req = DBRequest(
            source=self.rank, # This client node
            dest=operation_coordinator_rank,
            operation='CLIENT_PUT',
            key=key,
            value=value,
        )
        self.logger.debug(f"Client {self.rank} sending CLIENT_PUT for key '{key}' to coordinator {operation_coordinator_rank}")

        try:
            send_with_timeout(comm, client_put_req, dest=operation_coordinator_rank, tag=gen_tag(CLIENT_REQUEST_TAG, operation_coordinator_rank), timeout=MSG_TIMEOUT_S)
            final_response: DBResponse[T] = recv_with_timeout(comm, source=operation_coordinator_rank, tag=gen_tag(CLIENT_RESPONSE_TAG, self.rank), timeout=MSG_TIMEOUT_S)
            if final_response is not None:
                self.logger.info(f"Client {self.rank} received final response for PUT key '{key}': Status {final_response.operation_status}")
                return final_response.operation_status == 'SUCCESS'
            else: 
                return False

        except MPI.Exception as e:
            self.logger.error(f"Client {self.rank} MPI error during PUT for key '{key}': {e}")
            return False
        except Exception as e_gen:
            self.logger.error(f"Client {self.rank} general error during PUT for key '{key}': {e_gen} {traceback.format_exc()}")
            return False


    def get(self, key: str) -> List[T]:
        self.logger.info(f"Client {self.rank} initiating 'GET' for key: {key}")

        temp_pref_list = self._get_preference_list(key, 1) 
        if not temp_pref_list:
            self.logger.error(f"GET key '{key}': No live node found to act as operation coordinator.")
            return None
        operation_coordinator_rank = temp_pref_list[0]

        client_get_req = DBRequest(
            source=self.rank,
            dest=operation_coordinator_rank,
            operation="CLIENT_GET",
            key=key,
            value=None,
        )
        self.logger.debug(f"Client {self.rank} sending CLIENT_GET for key '{key}' to coordinator {operation_coordinator_rank}")

        try:
            send_with_timeout(comm, client_get_req, dest=operation_coordinator_rank, tag=gen_tag(CLIENT_REQUEST_TAG, operation_coordinator_rank), timeout=MSG_TIMEOUT_S)
            final_response: DBResponse[T] = recv_with_timeout(comm, source=operation_coordinator_rank, tag=gen_tag(CLIENT_RESPONSE_TAG, self.rank), timeout=MSG_TIMEOUT_S)
            if final_response is not None and final_response.operation_status == 'SUCCESS':
                self.logger.info(f"Client {self.rank} received final response for GET key '{key}': Status {final_response.operation_status}, Value: {final_response.value is not None}")
                return final_response.value
            else:
                return None

        except MPI.Exception as e:
            self.logger.error(f"Client {self.rank} MPI error during GET for key '{key}': {e}")
            return None
        except Exception as e_gen:
            self.logger.error(f"Client {self.rank} general error during GET for key '{key}': {e_gen} {traceback.format_exc()}")
            return None
        
