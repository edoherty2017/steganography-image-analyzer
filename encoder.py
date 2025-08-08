# encoder.py
from PIL import Image

DELIMITER = "00000000" * 8

def message_to_binary(message):
    """Converts a string message to its binary representation with a delimiter."""
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += DELIMITER
    return binary_message

def encode(image_path, message, output_path):
    """Hides a message within an image using LSB steganography."""
    try:
        # Ensure image is in RGB format to handle 3 color channels
        image = Image.open(image_path, 'r').convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    binary_message = message_to_binary(message)
    width, height = image.size
    # Calculate the maximum number of bytes the image can hold
    max_bytes = (width * height * 3) // 8

    if len(binary_message) > max_bytes * 8:
        print("Error: Message is too large for the image's capacity.")
        return

    new_image = image.copy()
    pixels = new_image.load()
    data_index = 0

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            # Modify Red channel
            if data_index < len(binary_message):
                # Clear the LSB and then set it to the message bit
                r = (r & 0b11111110) | int(binary_message[data_index])
                data_index += 1
            
            # Modify Green channel
            if data_index < len(binary_message):
                g = (g & 0b11111110) | int(binary_message[data_index])
                data_index += 1

            # Modify Blue channel
            if data_index < len(binary_message):
                b = (b & 0b11111110) | int(binary_message[data_index])
                data_index += 1
            
            pixels[x, y] = (r, g, b)

            # Exit loops if all data has been embedded
            if data_index >= len(binary_message):
                break
        if data_index >= len(binary_message):
            break
    
    # The save functionality will be added in the next commit