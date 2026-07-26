"""Quick test to verify Langfuse connection works."""
import os
import time
from dotenv import load_dotenv

load_dotenv(".env")

from langfuse import Langfuse

print("Connecting to Langfuse...")
client = Langfuse()

# Create a test trace (simulating a voice call)
trace = client.trace(
    name="demo-voice-call",
    user_id="demo-user",
    session_id="test-session-001",
    metadata={"direction": "inbound", "agent_id": "quickvoice-demo"},
    tags=["voice-agent", "demo"],
    input={"caller": "demo-user", "agent": "quickvoice-demo"},
    output={"status": "completed", "duration_seconds": 45},
)
trace_id = trace.id
print(f"Created trace: {trace_id}")

# Create a test generation (simulating an LLM turn)
generation = trace.generation(
    name="agent_llm_turn",
    model="gpt-4o",
    input=[
        {"role": "system", "content": "You are a helpful voice assistant for QuickVoice."},
        {"role": "user", "content": "Hi, I would like to book an appointment."},
    ],
    output="Sure! I'd be happy to help you book an appointment. What date and time works best for you?",
    metadata={"total_latency_seconds": 0.87},
    usage={"input": 35, "output": 22},
)
generation.end()
print("Created generation: agent_llm_turn")

# Create a test evaluation score
trace.score(
    name="call_success",
    value=1.0,
    comment="User goal was achieved",
)
print("Created score: call_success = 1.0")

# Flush and wait
client.flush()
time.sleep(3)
print("\nDone! Refresh your Langfuse dashboard now.")
print("URL: https://cloud.langfuse.com")
