#!/usr/bin/env python3
"""
Debugging Script for Stuttgart Building Regulations CrewAI System
This will help identify why agents aren't showing their work
"""

import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Enable maximum verbosity
os.environ['CREWAI_VERBOSE'] = 'True'
os.environ['CREWAI_LOG_LEVEL'] = 'DEBUG'
os.environ['OPENAI_LOG'] = 'debug'

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_crew.log', mode='w')
    ]
)

# Set all loggers to DEBUG
for logger_name in ['crewai', 'crewai.crew', 'crewai.agent', 'crewai.task', 
                    'optimized_crew_ai_system', 'optimized_vision_agent', '__main__']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

print("="*70)
print("CREWAI DEBUGGING DIAGNOSTIC TOOL")
print("="*70)

# ============================================================================
# STEP 1: Check Environment & Dependencies
# ============================================================================

print("\n[STEP 1] Checking Environment & Dependencies...")
print("-"*70)

# Check for .env file
env_files = ['.env', '.env1']
env_found = False
for env_file in env_files:
    if Path(env_file).exists():
        print(f"✅ Found environment file: {env_file}")
        from dotenv import load_dotenv
        load_dotenv(env_file)
        env_found = True
        break

if not env_found:
    print("❌ No .env file found")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ OPENAI_API_KEY is set: {api_key[:8]}...")
else:
    print("❌ OPENAI_API_KEY not found!")
    sys.exit(1)

# Check imports
try:
    import crewai
    print(f"✅ CrewAI version: {crewai.__version__ if hasattr(crewai, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"❌ CrewAI not installed: {e}")
    sys.exit(1)

try:
    from crewai import Agent, Task, Crew
    print("✅ CrewAI core components importable")
except ImportError as e:
    print(f"❌ CrewAI components error: {e}")

try:
    from optimized_crew_ai_system import StuttgartBuildingRegulationCrew, RegulationQuery
    print("✅ Custom crew system importable")
except ImportError as e:
    print(f"❌ Custom crew system error: {e}")
    print("   Trying alternative import...")
    try:
        from crew_ai_system import StuttgartBuildingRegulationCrew, RegulationQuery
        print("✅ Alternative crew system imported")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)

# ============================================================================
# STEP 2: Test CrewAI Verbose Settings
# ============================================================================

print("\n[STEP 2] Testing CrewAI Verbose Settings...")
print("-"*70)

# Check if CrewAI verbose is properly set
print(f"CREWAI_VERBOSE env var: {os.getenv('CREWAI_VERBOSE')}")
print(f"CREWAI_LOG_LEVEL env var: {os.getenv('CREWAI_LOG_LEVEL')}")

# Try to access CrewAI's internal verbose setting
try:
    from crewai import Crew
    test_crew = Crew(agents=[], tasks=[], verbose=True)
    print(f"✅ CrewAI Crew accepts verbose parameter")
    print(f"   Crew verbose setting: {test_crew.verbose if hasattr(test_crew, 'verbose') else 'Not accessible'}")
except TypeError as e:
    print(f"⚠️ CrewAI Crew doesn't accept verbose parameter: {e}")
except Exception as e:
    print(f"⚠️ Error creating test crew: {e}")

# ============================================================================
# STEP 3: Initialize Crew System with Debugging
# ============================================================================

print("\n[STEP 3] Initializing Crew System with Maximum Debugging...")
print("-"*70)

try:
    print("🔄 Creating StuttgartBuildingRegulationCrew instance...")
    
    # Try with different configurations
    crew = None
    configs_to_try = [
        {"openai_api_key": api_key, "use_gpt4": False, "enable_vision": False},
        {"openai_api_key": api_key, "use_gpt4": False},
        {"openai_api_key": api_key}
    ]
    
    for i, config in enumerate(configs_to_try, 1):
        try:
            print(f"   Attempt {i}: {config}")
            crew = StuttgartBuildingRegulationCrew(**config)
            print(f"✅ Crew initialized with config {i}")
            break
        except Exception as e:
            print(f"   ❌ Config {i} failed: {e}")
    
    if not crew:
        print("❌ Failed to initialize crew with any configuration")
        sys.exit(1)
        
    # Inspect crew structure
    print("\n📋 Crew Structure:")
    print(f"   Has 'crew' attribute: {hasattr(crew, 'crew')}")
    print(f"   Has 'agents' attribute: {hasattr(crew, 'agents')}")
    print(f"   Has 'execute_analysis' method: {hasattr(crew, 'execute_analysis')}")
    
    if hasattr(crew, 'agents'):
        print(f"   Number of agents: {len(crew.agents) if crew.agents else 0}")
        if crew.agents:
            for agent_name, agent in crew.agents.items():
                print(f"     - {agent_name}: {type(agent)}")
                if hasattr(agent, 'verbose'):
                    print(f"       Verbose: {agent.verbose}")
                if hasattr(agent, 'llm'):
                    print(f"       LLM: {type(agent.llm)}")
    
    if hasattr(crew, 'crew'):
        print(f"\n   Internal Crew object: {type(crew.crew)}")
        if hasattr(crew.crew, 'verbose'):
            print(f"   Crew verbose: {crew.crew.verbose}")
            # Try to set verbose
            crew.crew.verbose = True
            print(f"   Set crew.verbose = True")
        if hasattr(crew.crew, 'agents'):
            print(f"   Crew agents: {len(crew.crew.agents)}")
            for i, agent in enumerate(crew.crew.agents):
                print(f"     Agent {i}: {agent.role if hasattr(agent, 'role') else 'Unknown'}")
                if hasattr(agent, 'verbose'):
                    agent.verbose = True
                    print(f"       Set agent.verbose = True")
    
except Exception as e:
    print(f"❌ Error initializing crew: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Test Simple Query with Monitoring
# ============================================================================

print("\n[STEP 4] Testing Simple Query with Full Monitoring...")
print("-"*70)

# Create a simple test query
test_query = RegulationQuery(
    query="What is the maximum building height in Stuttgart?",
    project_type="Residential",
    location="Stuttgart",
    enable_vision=False  # Start without vision to simplify
)

print(f"📝 Test Query: {test_query.query}")
print(f"   Project Type: {test_query.project_type}")
print(f"   Vision Enabled: {test_query.enable_vision}")

# Hook into CrewAI's execution to monitor
print("\n🔄 Executing analysis...")
print("="*70)

# Try to capture all output
import io
import contextlib

# Create a string buffer to capture output
output_buffer = io.StringIO()

try:
    # Monitor the execution
    start_time = datetime.now()
    
    # Add custom logging handler to capture CrewAI internal logs
    class DebugHandler(logging.Handler):
        def emit(self, record):
            if 'crew' in record.name.lower() or 'agent' in record.name.lower():
                print(f"[{record.levelname}] {record.name}: {record.getMessage()}")
    
    debug_handler = DebugHandler()
    logging.getLogger().addHandler(debug_handler)
    
    print("\n🚀 Starting execute_analysis...")
    print("-"*70)
    
    # Execute with context capture
    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        result = crew.execute_analysis(test_query)
    
    # Get captured output
    captured_output = output_buffer.getvalue()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Execution completed in {duration:.2f} seconds")
    print("-"*70)
    
    # Analyze result
    print("\n📊 Result Analysis:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Has analysis: {'analysis' in result}")
    print(f"   Analysis length: {len(result.get('analysis', '')) if 'analysis' in result else 0} chars")
    print(f"   Vision used: {result.get('vision_used', False)}")
    print(f"   Processing time: {result.get('processing_time', 'N/A')}")
    
    if result.get('error'):
        print(f"   ❌ Error: {result['error']}")
    
    # Check captured output
    print("\n📜 Captured Output:")
    print("-"*70)
    if captured_output:
        print(captured_output[:2000])  # First 2000 chars
        if len(captured_output) > 2000:
            print(f"\n... (truncated, total {len(captured_output)} chars)")
    else:
        print("   No output captured from CrewAI execution")
    
except Exception as e:
    print(f"\n❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 5: Check Log Files
# ============================================================================

print("\n[STEP 5] Checking Log Files...")
print("-"*70)

log_files = ['debug_crew.log', 'app.log', 'crew.log']
for log_file in log_files:
    if Path(log_file).exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"\n📄 {log_file} ({len(lines)} lines):")
            # Show lines containing agent or crew
            relevant_lines = [l for l in lines if 'agent' in l.lower() or 'crew' in l.lower()]
            if relevant_lines:
                print("   Relevant lines (first 10):")
                for line in relevant_lines[:10]:
                    print(f"   {line.strip()}")
            else:
                print("   No agent/crew related lines found")

# ============================================================================
# STEP 6: Direct Agent Test
# ============================================================================

print("\n[STEP 6] Testing Direct Agent Execution...")
print("-"*70)

try:
    from crewai import Agent, Task
    from langchain_openai import ChatOpenAI
    
    print("Creating a simple test agent...")
    
    # Create a simple agent directly
    test_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=api_key
    )
    
    test_agent = Agent(
        role="Test Agent",
        goal="Test if agent output is visible",
        backstory="A test agent for debugging",
        llm=test_llm,
        verbose=True
    )
    
    print(f"✅ Test agent created: {test_agent.role}")
    print(f"   Verbose: {test_agent.verbose if hasattr(test_agent, 'verbose') else 'N/A'}")
    
    # Create a simple task
    test_task = Task(
        description="Say 'Hello, I am working!' to confirm you are active",
        expected_output="A greeting message",
        agent=test_agent
    )
    
    print("\n🔄 Executing test task directly...")
    
    # Try to execute the task
    from crewai import Crew
    mini_crew = Crew(
        agents=[test_agent],
        tasks=[test_task],
        verbose=True
    )
    
    print("Starting mini_crew.kickoff()...")
    result = mini_crew.kickoff()
    print(f"\n✅ Mini crew result: {result}")
    
except Exception as e:
    print(f"❌ Direct agent test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 7: Recommendations
# ============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("="*70)

print("\n🔍 Findings:")
print("-"*70)

# Check if we found the issue
issues_found = []

if not captured_output or len(captured_output) < 100:
    issues_found.append("CrewAI is not producing verbose output")

if not hasattr(crew, 'crew'):
    issues_found.append("Crew object doesn't have internal 'crew' attribute")

if hasattr(crew, 'crew') and not hasattr(crew.crew, 'verbose'):
    issues_found.append("Internal crew doesn't support verbose mode")

if len(issues_found) == 0:
    issues_found.append("No obvious issues found - may need deeper debugging")

for i, issue in enumerate(issues_found, 1):
    print(f"{i}. {issue}")

print("\n💡 Recommendations:")
print("-"*70)

recommendations = [
    "1. Try updating CrewAI: pip install --upgrade crewai",
    "2. Check if CrewAI version supports verbose output in your version",
    "3. Add manual logging in your execute_analysis method",
    "4. Use print statements in agent definitions to track execution",
    "5. Consider using CrewAI's built-in callbacks for monitoring"
]

for rec in recommendations:
    print(rec)

print("\n📁 Log files created:")
print("   - debug_crew.log (full debug output)")
print("   - Check these for detailed execution trace")

print("\n" + "="*70)
print("Debugging complete. Check the output above for issues.")
print("="*70)