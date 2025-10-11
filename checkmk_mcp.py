from fastmcp import FastMCP, Context
import requests
import os

# Checkmk server configuration
CHECKMK_URL = 'http://localhost:8080'
CHECKMK_USERNAME = os.getenv('CHECKMK_USERNAME')
CHECKMK_PASSWORD = os.getenv('CHECKMK_PASSWORD')

if not CHECKMK_USERNAME or not CHECKMK_PASSWORD:
    raise ValueError("CHECKMK_USERNAME and CHECKMK_PASSWORD environment variables must be set")

mcp = FastMCP("Checkmk MCP Server")

@mcp.tool
def get_hosts(ctx: Context) -> str:
    """Get all hosts from Checkmk"""
    try:
        url = f'{CHECKMK_URL}/cmk/check_mk/api/1.0/domain-types/host/collections/all'
        headers = {
            'Authorization': f'Bearer {CHECKMK_USERNAME} {CHECKMK_PASSWORD}'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        ctx.error(f"Error connecting to Checkmk API: {e}")
        return f"Error: {e}"


@mcp.tool
def get_host_status(ctx: Context, host_name: str) -> str:
    """Get the status of a specific host."""
    try:
        url = f'{CHECKMK_URL}/cmk/check_mk/api/1.0/domain-types/service/collections/all'
        headers = {
            'Authorization': f'Bearer {CHECKMK_USERNAME} {CHECKMK_PASSWORD}',
            'Content-Type': 'application/json',
        }
        data = {
            "query": f'{{"op": "=", "left": "host_name", "right": "{host_name}"}}'
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        ctx.error(f"Error connecting to Checkmk API: {e}")
        return f"Error: {e}"


@mcp.tool
def get_service_status(ctx: Context, host_name: str, service_description: str) -> str:
    """Get the status of a specific service on a host."""
    try:
        url = f'{CHECKMK_URL}/cmk/check_mk/api/1.0/objects/host/{host_name}/actions/show_service/invoke'
        headers = {
            'Authorization': f'Bearer {CHECKMK_USERNAME} {CHECKMK_PASSWORD}'
        }
        params = {
            'service_description': service_description
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        ctx.error(f"Error connecting to Checkmk API: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()
