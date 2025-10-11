import ollama
import requests
import json
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="A client to interact with the Checkmk MCP server and an LLM.")
parser.add_argument("--mcp-url", type=str, default="http://localhost:8000", help="The URL of the Checkmk MCP server.")
parser.add_argument("--ollama-model", type=str, default="gemma:2b", help="The name of the Ollama model to use.")
args = parser.parse_args()

# --- Configuration ---
CHECKMK_MCP_URL = args.mcp_url
OLLAMA_MODEL = args.ollama_model

# --- MCP Client ---
def get_mcp_tools():
    try:
        response = requests.get(f"{CHECKMK_MCP_URL}/tools")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting tools from MCP server: {e}")
        return None

def call_mcp_tool(tool_name, **kwargs):
    try:
        response = requests.post(f"{CHECKMK_MCP_URL}/tools/{tool_name}", json=kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling tool '{tool_name}' on MCP server: {e}")
        return None

# --- Main Loop ---
def main():
    print("Getting tools from Checkmk MCP server...")
    tools = get_mcp_tools()
    if not tools:
        return

    print("Available tools:", [tool['name'] for tool in tools])

    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools
        )

        messages.append(response["message"])

        if response["message"].get("tool_calls"):
            for tool_call in response["message"]["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                print(f"LLM wants to call tool: {tool_name} with arguments: {tool_args}")
                tool_result = call_mcp_tool(tool_name, **tool_args)
                print(f"Tool result: {tool_result}")
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result)
                })
        else:
            print("LLM:", response["message"]["content"])

if __name__ == "__main__":
    main()