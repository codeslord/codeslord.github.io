---
layout: post
title: Minimal AI Agents: The Essential Code You Can Remember
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