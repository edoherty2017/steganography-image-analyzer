# decoder.py
from PIL import Image

DELIMITER = "00000000" * 8

def decode(image_path):
    """Extracts a hidden message from an image."""
    try:
        image = Image.open(image_path, 'r').convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    binary_data = ""
    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            binary_data += str(r & 1)
            binary_data += str(g & 1)
            binary_data += str(b & 1)

    # Find the delimiter in the extracted binary data
    delimiter_index = binary_data.find(DELIMITER)
    
    if delimiter_index == -1:
        print("Error: Could not find message delimiter in the image.")
        return None
    
    # Get the binary data for the message (everything before the delimiter)
    message_binary = binary_data[:delimiter_index]
    
    message = ""
    # Process the binary string in 8-bit chunks (bytes)
    for i in range(0, len(message_binary), 8):
        byte = message_binary[i:i+8]
        # Ensure we have a full byte before converting
        if len(byte) == 8:
            # Convert the byte to an integer, then to a character
            message += chr(int(byte, 2))
    
    return message