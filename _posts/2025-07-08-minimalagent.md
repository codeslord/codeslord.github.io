---
layout: post
title: Minimal AI Agents - The Essential Code You Can Remember
date: 2025-07-08 15:00
summary: The most compact and practical Python code patterns to build AI agents with tool calling and MCP support, designed for easy recall without depending on ChatGPT or other external LLMs. It covers one-liner setups for OpenAI, MCP, and agentic SDKs, stripping complexity to expose only core logic for initialization, prompt handling, and minimal tool execution. The guide lets developers quickly build, adapt, and memorize agent workflows with absolute minimal code, supporting learning, prototyping, and production integration for modern AI platform managers and technical teams.
categories: General
---

<img src="https://i.ibb.co/ch83rmhK/blog-post.jpg" alt="memorizeagent" border="0">

Here's the most **bare minimum** Python code for AI Agents with tool calling and MCP support that you can remember without needing ChatGPT:

## **1. OpenAI Agent (Simplest)**

```python
import openai
import json

client = openai.OpenAI()

def weather_tool(city: str):
    return f"Weather in {city}: 22°C, sunny"

def run_agent(prompt):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
    
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
    
    if response.choices[^0].message.tool_calls:
        for tool_call in response.choices[^0].message.tool_calls:
            if tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                result = weather_tool(args["city"])
                print(f"Tool result: {result}")
    
    return response.choices[^0].message.content
```


## **2. MCP Server (3 Lines Core)**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

# Run: uv run mcp dev server.py
```


## **3. OpenAI Agents SDK (Newest - 4 Lines)**

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

agent = Agent(name="Assistant", instructions="You are helpful", tools=[get_weather])
result = Runner.run_sync(agent, "What's the weather in Tokyo?")
```


## **4. Ultra Minimal Loop**

```python
import openai

client = openai.OpenAI()

def agent_loop(prompt):
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        response = client.chat.completions.create(model="gpt-4", messages=messages)
        message = response.choices[^0].message
        messages.append({"role": "assistant", "content": message.content})
        
        if not message.tool_calls:
            return message.content
```


## **Key Installation Commands**

```bash
pip install openai           # OpenAI
pip install anthropic        # Anthropic  
pip install "mcp[cli]"       # MCP
pip install openai-agents    # OpenAI Agents SDK
```


## **Remember These Patterns**

1. **OpenAI**: `tools=[{"type": "function", "function": {...}}]`
2. **Anthropic**: `tools=[{"name": "...", "input_schema": {...}}]`
3. **MCP**: `@mcp.tool()` decorator
4. **Agent Loop**: LLM → Check tool_calls → Execute → Repeat

The **OpenAI Agents SDK**  is the newest and simplest approach, requiring just 4 lines for a working agent. For MCP servers, the FastMCP approach  gives you the most minimal implementation with just decorators


# **Memory Tips for Minimal AI Agent Code**

## **1. Mental Models \& Patterns**

### **The "3-Step Agent Dance"**

Remember every agent follows this pattern:

1. **Define Tools** (What can it do?)
2. **Send Message** (What does user want?)
3. **Handle Response** (Did it call tools?)
```python
# Pattern: DEFINE → SEND → HANDLE
tools = [...]           # DEFINE
response = client.create(...)  # SEND  
if tool_calls: ...      # HANDLE
```


### **The "JSON Sandwich"**

Tools are always wrapped in JSON schemas:

- **OpenAI**: `{"type": "function", "function": {...}}`
- **Anthropic**: `{"name": "...", "input_schema": {...}}`
- **MCP**: Just decorators `@mcp.tool()`


## **2. Mnemonics for API Differences**

### **"Open Functions, Anthropic Names"**

- **OpenAI**: Uses `"function"` everywhere → `tools["function"]["name"]`
- **Anthropic**: Direct `"name"` → `tools["name"]`


### **"OpenAI = Nested, Anthropic = Flat"**

- **OpenAI**: `response.choices.message.tool_calls`
- **Anthropic**: `response.content` (flatter structure)


## **3. Code Chunk Memorization**

### **Essential Imports Block**

```python
import openai      # Always openai
import json        # For parsing args
```


### **Tool Schema Template** (OpenAI)

```python
{
    "type": "function",
    "function": {
        "name": "TOOL_NAME",
        "description": "WHAT_IT_DOES", 
        "parameters": {
            "type": "object",
            "properties": {"PARAM": {"type": "TYPE"}},
            "required": ["PARAM"]
        }
    }
}
```

**Memory trick**: "Type-Function-Name-Desc-Params" (TFNDP)

### **Response Handling Pattern**

```python
if response.choices[^0].message.tool_calls:
    for tool_call in response.choices[^0].message.tool_calls:
        # Always: name → args → execute
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = my_tool(**args)
```


## **4. Visual Memory Aids**

### **Agent Flow Diagram** (Mental Picture)

```
USER → [AGENT] → LLM → Tool Call? 
                  ↓      ↓ Yes
               Response ← TOOL
```


### **MCP Memory**: "Decorators Do Everything"

- `@mcp.tool()` = Function tool
- `@mcp.resource()` = Data source
- `@mcp.prompt()` = Template


## **5. Minimal Templates to Memorize**

### **10-Second OpenAI Agent** (Core Template)

```python
import openai, json
client = openai.OpenAI()
tools = [{"type": "function", "function": {"name": "X", "parameters": {...}}}]
response = client.chat.completions.create(model="gpt-4", messages=[...], tools=tools)
```


### **5-Second MCP Server**

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Name")
@mcp.tool()
def func(): pass
```


## **6. Practice Techniques**

### **The "30-Second Challenge"**

Set a timer and write a basic agent from memory in 30 seconds. Focus on:

1. Import statement
2. Client initialization
3. Tool definition
4. API call

### **"Fill in the Blanks" Method**

Practice with templates like:

```python
import _____, _____
client = _____.OpenAI()
tools = [{"type": "_____", "_____": {...}}]
```


### **"API Diff Flashcards"**

Create mental flashcards:

- **Front**: "OpenAI tool calling structure"
- **Back**: `response.choices.message.tool_calls`


## **7. Common Gotchas to Remember**

### **JSON Loading**

Always `json.loads(tool_call.function.arguments)` - never forget!

### **Model Names**

- OpenAI: `"gpt-4"`
- Anthropic: `"claude-3-sonnet-20240229"`


### **Message Format**

Always: `[{"role": "user", "content": "..."}]`

## **8. Contextual Anchors**

### **Connect to Your Experience**

- **MCP**: "Model Context Protocol" → Think "My Custom Protocol"
- **Tool Calling**: Think of it as "Function Remote Control"
- **Agent Loop**: Like a conversation where AI asks for help


### **Real-World Analogies**

- **Tools**: Like having a Swiss Army knife
- **Agent**: Like a smart assistant who knows when to use each tool
- **MCP**: Like a universal translator between AI and tools


## **9. Quick Reference Card** (Print/Save)

```
OPENAI PATTERN:
├─ tools = [{"type": "function", "function": {...}}]
├─ response = client.chat.completions.create(...)
└─ if response.choices[^0].message.tool_calls:

MCP PATTERN:  
├─ from mcp.server.fastmcp import FastMCP
├─ mcp = FastMCP("Name")  
└─ @mcp.tool() def func():

ANTHROPIC PATTERN:
├─ tools = [{"name": "...", "input_schema": {...}}] 
├─ response = client.messages.create(...)
└─ if response.stop_reason == "tool_use":
```

The key is **repetition with understanding** - practice writing these patterns until they become muscle memory, but always understand what each piece does so you can adapt when needed.