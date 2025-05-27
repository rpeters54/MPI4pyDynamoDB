import time
from mpi4py import MPI
from typing import Any, List, Tuple, Optional, Union, NamedTuple

# Helper type for send specifications
class SendSpec(NamedTuple):
    data: Any
    dest: int
    tag: int

# Helper type for receive specifications
class RecvSpec(NamedTuple):
    source: int
    tag: int


def send_with_timeout(
    comm: MPI.Comm, 
    data: Any, 
    dest: int, 
    tag: int, 
    timeout: float
) -> bool:
    """
    Sends a Python object to a destination rank with a timeout.

    The data is pickled by mpi4py's isend.

    Args:
        comm: The MPI communicator.
        data: The Python object to send.
        dest: The rank of the destination process.
        tag: The message tag.
        timeout: The maximum time (in seconds) to wait for the send to complete.
                 A value of 0 or less will effectively make it a non-blocking test.

    Returns:
        True if the send operation completed within the timeout, False otherwise.
    """
    if timeout < 0:
        timeout = 0
        
    req = comm.isend(data, dest=dest, tag=tag)
    start_time = time.monotonic()
    success = False

    while time.monotonic() - start_time < timeout:
        # For isend, test() returns a boolean indicating completion.
        if req.test():
            success = True
            break
        time.sleep(0.001)  # Small sleep to yield CPU and avoid busy-waiting

    if not success:
        # Timeout occurred. The request is still pending.
        # Attempt to cancel the request. Cancellation is a "best effort" in MPI.
        try:
            req.cancel()
        except MPI.Exception:
            # Ignore errors during cancellation (e.g., if it just completed, or MPI implementation details)
            pass
            
    return success

def recv_with_timeout(
    comm: MPI.Comm, 
    source: int, 
    tag: int, 
    timeout: float
) -> Optional[Any]:
    """
    Receives a Python object from a source rank with a timeout.

    The data is unpickled by mpi4py's irecv.

    Args:
        comm: The MPI communicator.
        source: The rank of the source process (e.g., MPI.ANY_SOURCE).
        tag: The message tag (e.g., MPI.ANY_TAG).
        timeout: The maximum time (in seconds) to wait for the receive to complete.
                 A value of 0 or less will effectively make it a non-blocking test.

    Returns:
        The received Python object if successful within the timeout, None otherwise.
    """
    if timeout < 0:
        timeout = 0

    req = comm.irecv(source=source, tag=tag)
    start_time = time.monotonic()
    received_data: Optional[Any] = None

    while time.monotonic() - start_time < timeout:
        # For irecv, test() returns a tuple: (bool_completed, data_if_completed_else_None)
        completed, data = req.test()
        if completed:
            received_data = data
            break
        time.sleep(0.001)  # Small sleep to yield CPU

    if received_data is None: # Timeout occurred or test() returned (False, None) initially
        # Attempt to cancel the pending receive request.
        try:
            req.cancel()
        except MPI.Exception:
            # Ignore errors during cancellation
            pass
            
    return received_data

def send_all_with_timeout(
    comm: MPI.Comm, 
    send_specs: List[SendSpec], 
    timeout: float
) -> List[bool]:
    """
    Sends multiple Python objects based on a list of specifications.
    Attempts to complete all send operations within a global timeout.

    Args:
        comm: The MPI communicator.
        send_specs: A list of tuples, where each tuple is (data, dest_rank, tag).
        timeout: The maximum total time (in seconds) to wait for all operations to be attempted.
                 If an operation is not completed by the time the loop finishes due to timeout,
                 it's marked as failed (False).

    Returns:
        A list of booleans, corresponding to the `send_specs`. Each boolean is
        True if the respective send operation completed, False otherwise (e.g., timed out).
    """
    if timeout < 0:
        timeout = 0
        
    num_messages = len(send_specs)
    if not num_messages:
        return []

    requests = [comm.isend(data, dest=dest, tag=tag) for data, dest, tag in send_specs]
    results = [False] * num_messages
    
    # Track indices of requests that are still pending
    pending_indices = list(range(num_messages))
    
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout and pending_indices:
        # Iterate over a copy of pending_indices because we might modify it during iteration
        indices_to_check_this_round = list(pending_indices) 
        
        for i in indices_to_check_this_round:
            if requests[i].test():  # test() for isend returns boolean
                results[i] = True
                pending_indices.remove(i) # Successfully sent, remove from pending
        
        if not pending_indices: # All messages have been sent
            break
        time.sleep(0.001) # Yield

    # For any requests that are still pending after the timeout or completion of others
    for i in pending_indices:
        results[i] = False # Ensure it's marked as False if not already True
        try:
            requests[i].cancel()
        except MPI.Exception:
            pass # Ignore cancellation errors

    return results

def recv_all_with_timeout(
    comm: MPI.Comm, 
    recv_specs: List[RecvSpec], 
    timeout: float
) -> List[Optional[Any]]:
    """
    Receives multiple Python objects based on a list of specifications.
    Attempts to complete all receive operations within a global timeout.

    Args:
        comm: The MPI communicator.
        recv_specs: A list of tuples, where each tuple is (source_rank, tag).
        timeout: The maximum total time (in seconds) to wait for all operations to be attempted.

    Returns:
        A list corresponding to `recv_specs`. Each element is the received data 
        for the respective operation, or None if that specific receive timed out or failed.
    """
    if timeout < 0:
        timeout = 0

    num_messages = len(recv_specs)
    if not num_messages:
        return []

    requests = [comm.irecv(source=source, tag=tag) for source, tag in recv_specs]
    # Initialize results with None; will be filled with data upon successful reception
    results: List[Optional[Any]] = [None] * num_messages
    
    pending_indices = list(range(num_messages))
    
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout and pending_indices:
        indices_to_check_this_round = list(pending_indices)

        for i in indices_to_check_this_round:
            # irecv().test() returns (bool_completed, data_if_completed_else_None)
            completed, data = requests[i].test()
            if completed:
                results[i] = data
                pending_indices.remove(i) # Successfully received, remove from pending
        
        if not pending_indices: # All messages have been received
            break
        time.sleep(0.001) # Yield

    # For any requests still pending, their entry in results is already None.
    # Attempt to cancel them.
    for i in pending_indices:
        # results[i] is already None
        try:
            requests[i].cancel()
        except MPI.Exception:
            pass # Ignore cancellation errors
            
    return results