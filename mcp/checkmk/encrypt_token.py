from cryptography.fernet import Fernet

# Generate a key and save it to a file
def generate_key():
    key = Fernet.generate_key()
    with open("key.key", "wb") as key_file:
        key_file.write(key)

# Load the key from the current directory
def load_key():
    return open("key.key", "rb").read()

# Encrypt a message
def encrypt_message(message):
    key = load_key()
    encoded_message = message.encode()
    f = Fernet(key)
    encrypted_message = f.encrypt(encoded_message)
    return encrypted_message

if __name__ == "__main__":
    generate_key()
    print("Encryption key generated and saved to key.key")
    token = input("Enter the token to encrypt: ")
    encrypted_token = encrypt_message(token)
    print("Encrypted token:", encrypted_token.decode())
