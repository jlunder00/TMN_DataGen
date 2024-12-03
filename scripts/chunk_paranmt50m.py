import argparse
from pathlib import Path

def chunkify(input_path: Path, output_dir: Path, chunk_size: int) -> None:
    """Split file into chunks and save them in the specified directory"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r') as f:
        header = next(f)  # Read and save the header
        chunk = []
        chunk_index = 0

        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                # Write current chunk to a file
                output_file = output_dir / f"chunk_{chunk_index:04d}.txt"
                with open(output_file, 'w') as out:
                    out.write(header)
                    out.writelines(chunk)
                chunk_index += 1
                chunk = []

        if chunk:  # Final chunk
            output_file = output_dir / f"chunk_{chunk_index:04d}.txt"
            with open(output_file, 'w') as out:
                out.write(header)
                out.writelines(chunk)

def main():
    parser = argparse.ArgumentParser(description='Chunkify large input file')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_dir', type=str, help='Directory to store chunk files')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Number of lines per chunk (excluding header)')
    
    args = parser.parse_args()
    
    chunkify(Path(args.input_file), Path(args.output_dir), args.chunk_size)

if __name__ == '__main__':
    main()

