import os
import numpy as np
import imageio
import argparse

def convert_directory_npy_to_png(input_directory, output_directory):
    # Ensure the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            # Load the numpy array from the .npy file
            npy_path = os.path.join(input_directory, filename)
            image_array = np.load(npy_path)

            # Ensure the array is of type uint8 (values in the range [0, 255])
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)

            # Save the array as a PNG image
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_directory, png_filename)
            imageio.imwrite(png_path, image_array)

            print(f"Converted {filename} to {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all .npy files in a directory to .png format.")
    parser.add_argument("input_directory", help="Path to the input directory containing .npy files")
    parser.add_argument("output_directory", help="Path to the output directory for saving .png files")

    args = parser.parse_args()

    convert_directory_npy_to_png(args.input_directory, args.output_directory)
