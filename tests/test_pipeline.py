import pytest
from memory.buffer_memory import BufferMemory
from memory.token_memory import TokenWindowMemory
from utils.metrics import MetricsRegistry

def test_buffer_memory():
    """Asserts that BufferMemory records messages without compression."""
    memory = BufferMemory()
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi there")
    
    history = memory.get_history()
    assert "User: Hello" in history
    assert "Assistant: Hi there" in history
    
    memory.clear()
    assert len(memory.get_messages()) == 0

def test_token_window_memory_pruning():
    """Asserts that TokenWindowMemory prunes history according to size bounds."""
    memory = TokenWindowMemory(max_tokens=30)
    
    # Each message is roughly 10 words (~13 tokens)
    memory.add_message("user", "This is a very long question that takes many tokens to verify.")
    memory.add_message("assistant", "This is another long answer that also consumes many tokens.")
    memory.add_message("user", "Third turn query to trigger pruning.")
    
    history = memory.get_history()
    
    # The oldest message should be pruned to keep estimated tokens under 30
    assert "takes many tokens" not in history
    assert "Third turn" in history

def test_metrics_registry():
    """Validates metrics calculations inside MetricsRegistry."""
    MetricsRegistry.clear()
    
    MetricsRegistry.record_latency("test_op", 150.0)
    MetricsRegistry.record_latency("test_op", 250.0)
    MetricsRegistry.record_tokens(100, 50)
    
    metrics = MetricsRegistry.get_metrics()
    assert metrics["avg_latency_ms"]["test_op"] == 200.0
    assert metrics["tokens"]["total_tokens"] == 150
