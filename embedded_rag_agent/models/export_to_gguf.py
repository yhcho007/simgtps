import argparse, subprocess, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()

    # llama.cpp conversion placeholder
    print(f"Convert {args.model_dir} to GGUF...")
    # subprocess.run([...])

if __name__ == '__main__':
    main()
