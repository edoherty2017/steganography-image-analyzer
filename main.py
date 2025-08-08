# main.py
import argparse
from encoder import encode
from decoder import decode

def main():
    parser = argparse.ArgumentParser(description="A tool for LSB steganography.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser for the "encode" command
    encode_parser = subparsers.add_parser("encode", help="Encode a message into an image.")
    encode_parser.add_argument("-i", "--image", required=True, help="Path to the input image (PNG).")
    encode_parser.add_argument("-m", "--message", required=True, help="The message string to hide.")
    encode_parser.add_argument("-o", "--output", required=True, help="Path for the output encoded image.")

    # Parser for the "decode" command
    decode_parser = subparsers.add_parser("decode", help="Decode a message from an image.")
    decode_parser.add_argument("-i", "--image", required=True, help="Path to the encoded image.")

    args = parser.parse_args()

    if args.command == "encode":
        encode(image_path=args.image, message=args.message, output_path=args.output)
    elif args.command == "decode":
        # Call the decode function and print the result if successful
        message = decode(image_path=args.image)
        if message:
            print("Decoded Message Found:")
            print(message)

if __name__ == "__main__":
    main()