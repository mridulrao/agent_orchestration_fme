import os
import sys
import subprocess


def main():
    print("Setting up environment...")

    # Check if prisma is installed
    try:
        subprocess.run(["prisma", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Prisma CLI not found. Please install it first with:")
        print("npm install -g prisma")
        return 1

    # Specify the schema path
    schema_path = "db/schema.prisma"

    # Check if the schema file exists
    if not os.path.exists(schema_path):
        print(f"Schema file not found at {schema_path}")
        return 1

    # Run prisma generate with the specific schema path
    print("Generating Prisma client...")
    result = subprocess.run(
        ["prisma", "generate", "--schema", schema_path], check=False
    )

    if result.returncode != 0:
        print(
            f"Failed to generate Prisma client. Please check your schema file at {schema_path}"
        )
        return 1

    print("Prisma client generated successfully.")

    # print("Running embedding.py to download embedding model...")
    # result = subprocess.run([sys.executable, "./rag/embedding_model.py"])

    # if result.returncode != 0:
    #     print(f"Failed to run embedding.py and download the embedding model.")
    #     return 1

    # print("Embedding model downloaded successfully.")

    # run the trun model files download
    file_args = ["download-files"]

    # Add any additional arguments from the command line (excluding the script name itself)
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])

    # Run the main script with the arguments
    print(f"Running main application with arguments: {' '.join(file_args)}")
    result = subprocess.run([sys.executable, "main.py"] + file_args)

    # Get arguments to pass to main.py
    # The first argument is always "start" in this case
    args = ["dev"]

    if args == ["start"]:
        print(
            f"WARNING(added by mridul): Starting Livekit server in start mode, use 'dev' if running locally"
        )

    # Add any additional arguments from the command line (excluding the script name itself)
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])

    # Run the main script with the arguments
    print(f"Running main application with arguments: {' '.join(args)}")
    result = subprocess.run([sys.executable, "main.py"] + args)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
