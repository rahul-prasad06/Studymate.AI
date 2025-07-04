import os
import shutil

# Path to vectorstore folder
VECTORSTORE_DIR = "vectorstore/"

def clear_vectorstore():
    """Delete all contents of the vectorstore folder."""
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print(" Cleared vectorstore folder.")

    # Recreate empty vectorstore folder
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    print(" Vectorstore is now empty and ready for fresh testing.")

if __name__ == "__main__":
    clear_vectorstore()
