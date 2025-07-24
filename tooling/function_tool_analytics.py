from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import uuid

class FunctionToolCall:
    """Represents a single function tool call with its details"""
    
    def __init__(self, function_name: str, inputs: Dict[str, Any] = None, 
                 call_id: str = None):
        self.call_id = call_id or str(uuid.uuid4())
        self.function_name = function_name
        self.inputs = inputs or {}
        self.output = None
        self.start_time = datetime.now()
        self.end_time = None
        self.duration_ms = None
        self.status = "started"  # started, completed, failed
        self.error_message = None
    
    def complete(self, output: Any):
        """Mark the function call as completed with output"""
        self.output = output
        self.end_time = datetime.now()
        self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        self.status = "completed"
    
    def fail(self, error_message: str):
        """Mark the function call as failed with error message"""
        self.error_message = error_message
        self.end_time = datetime.now()
        self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        self.status = "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "call_id": self.call_id,
            "function_name": self.function_name,
            "inputs": self.inputs,
            "output": self.output,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_message": self.error_message
        }
    
    def __repr__(self):
        return f"FunctionToolCall(name='{self.function_name}', status='{self.status}', duration={self.duration_ms}ms)"


class FunctionToolTracker:
    """Tracks all function tool calls made by an agent"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.function_calls: List[FunctionToolCall] = []
        self.active_calls: Dict[str, FunctionToolCall] = {}
        self.session_start_time = datetime.now()
    
    def start_function_call(self, function_name: str, inputs: Dict[str, Any] = None, 
                          call_id: str = None) -> str:
        """Start tracking a new function call"""
        call = FunctionToolCall(function_name, inputs, call_id)
        self.function_calls.append(call)
        self.active_calls[call.call_id] = call
        return call.call_id
    
    def complete_function_call(self, call_id: str, output: Any):
        """Complete a function call with its output"""
        if call_id in self.active_calls:
            self.active_calls[call_id].complete(output)
            del self.active_calls[call_id]
    
    def fail_function_call(self, call_id: str, error_message: str):
        """Mark a function call as failed"""
        if call_id in self.active_calls:
            self.active_calls[call_id].fail(error_message)
            del self.active_calls[call_id]
    
    def get_function_calls(self, function_name: str = None, 
                          status: str = None) -> List[FunctionToolCall]:
        """Get function calls filtered by name and/or status"""
        filtered_calls = self.function_calls
        
        if function_name:
            filtered_calls = [call for call in filtered_calls 
                            if call.function_name == function_name]
        
        if status:
            filtered_calls = [call for call in filtered_calls 
                            if call.status == status]
        
        return filtered_calls
    
    def get_call_by_id(self, call_id: str) -> Optional[FunctionToolCall]:
        """Get a specific function call by ID"""
        for call in self.function_calls:
            if call.call_id == call_id:
                return call
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about function calls"""
        total_calls = len(self.function_calls)
        completed_calls = len([c for c in self.function_calls if c.status == "completed"])
        failed_calls = len([c for c in self.function_calls if c.status == "failed"])
        active_calls = len(self.active_calls)
        
        # Function call counts by name
        function_counts = {}
        for call in self.function_calls:
            function_counts[call.function_name] = function_counts.get(call.function_name, 0) + 1
        
        # Average duration for completed calls
        completed_durations = [c.duration_ms for c in self.function_calls 
                             if c.status == "completed" and c.duration_ms is not None]
        avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        
        return {
            "session_id": self.session_id,
            "total_calls": total_calls,
            "completed_calls": completed_calls,
            "failed_calls": failed_calls,
            "active_calls": active_calls,
            "function_counts": function_counts,
            "average_duration_ms": round(avg_duration, 2),
            "session_duration_ms": int((datetime.now() - self.session_start_time).total_seconds() * 1000)
        }
    
    def export_to_json(self, filepath: str = None) -> str:
        """Export all function calls to JSON"""
        data = {
            "session_id": self.session_id,
            "session_start_time": self.session_start_time.isoformat(),
            "function_calls": [call.to_dict() for call in self.function_calls],
            "statistics": self.get_statistics()
        }
        
        json_str = json.dumps(data, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def clear_history(self):
        """Clear all function call history"""
        self.function_calls.clear()
        self.active_calls.clear()
    
    def __len__(self):
        return len(self.function_calls)
    
    def __repr__(self):
        return f"FunctionToolTracker(session_id='{self.session_id}', calls={len(self.function_calls)})"


# Decorator to automatically track function calls
def tracked_function_tool(tracker: FunctionToolTracker):
    """Decorator to automatically track function tool calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            call_id = tracker.start_function_call(
                function_name=func.__name__,
                inputs={"args": args, "kwargs": kwargs}
            )
            try:
                result = await func(*args, **kwargs)
                tracker.complete_function_call(call_id, result)
                return result
            except Exception as e:
                tracker.fail_function_call(call_id, str(e))
                raise
        
        def sync_wrapper(*args, **kwargs):
            call_id = tracker.start_function_call(
                function_name=func.__name__,
                inputs={"args": args, "kwargs": kwargs}
            )
            try:
                result = func(*args, **kwargs)
                tracker.complete_function_call(call_id, result)
                return result
            except Exception as e:
                tracker.fail_function_call(call_id, str(e))
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# # Example usage and integration with your existing code
# if __name__ == "__main__":
#     # Create tracker instance
#     tracker = FunctionToolTracker(session_id="user_session_123")
    
#     # Example of manual tracking (like in your existing code)
#     call_id = tracker.start_function_call(
#         "retrieve_previous_ticket",
#         inputs={"user_sys_id": "12345", "limit": 3}
#     )
    
#     # Simulate function execution
#     import time
#     time.sleep(0.1)  # Simulate processing time
    
#     # Complete with output (like your function's return)
#     output = {
#         "Previous Tickets": [
#             {
#                 "Incident Number": "INC0012345",
#                 "Short Description": "Email not working",
#                 "Status": "Open",
#                 "Created Date": "2024-01-15",
#                 "Link": "#"
#             }
#         ],
#         "Status": "Fetched tickets successfully"
#     }
#     tracker.complete_function_call(call_id, output)
    
#     # Print statistics
#     print("Tracker Statistics:")
#     print(json.dumps(tracker.get_statistics(), indent=2))
    
#     # Export to JSON
#     json_export = tracker.export_to_json()
#     print("\nJSON Export:")
#     print(json_export)