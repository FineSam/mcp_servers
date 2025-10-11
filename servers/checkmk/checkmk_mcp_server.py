from fastmcp import FastMCP, Context
import requests
import yaml
from cryptography.fernet import Fernet

# --- Settings and Encryption ---
def load_settings():
    with open("servers/checkmk/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def load_key():
    return open("servers/checkmk/key.key", "rb").read()

def decrypt_message(encrypted_message):
    key = load_key()
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    return decrypted_message.decode()

settings = load_settings()
CHECKMK_URL = settings["checkmk_url"]
CHECKMK_USERNAME = settings["checkmk_username"]
ENCRYPTED_TOKEN = settings["checkmk_token"].encode()

# Decrypt the token
try:
    CHECKMK_TOKEN = decrypt_message(ENCRYPTED_TOKEN)
except:
    print("Error decrypting token. Please ensure that the token is correctly encrypted and the key.key file is present.")
    exit()

mcp = FastMCP("Checkmk MCP Server")

@mcp.tool
def get_hosts(ctx: Context) -> str:
    """Get all hosts from Checkmk"""
    try:
        url = f'{CHECKMK_URL}/cmk/check_mk/api/1.0/domain-types/host/collections/all'
        headers = {
            'Authorization': f'Bearer {CHECKMK_USERNAME} {CHECKMK_TOKEN}'
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
            'Authorization': f'Bearer {CHECKMK_USERNAME} {CHECKMK_TOKEN}',
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
def get_service_.status(ctx: Context, host_name: str, service_description: str) -> str:
    """Get the status of a specific service on a host."""
    try:
        url = f'{CHECKMK_URL}/cmk/check_mk/api/1.0/objects/host/{host_name}/actions/show_service/invoke'
        headers = {
            'Authorization': f'Bearer {CHECKMK_USERNAME} {CHECKMK_TOKEN}'
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