import os

def check_alignment(java_file, cs_file):
    if not os.path.exists(java_file):
        print(f"Error: Java file not found at {java_file}")
        return False
    if not os.path.exists(cs_file):
        print(f"Error: C# file not found at {cs_file}")
        return False

    with open(java_file, 'r') as f_java:
        java_lines = f_java.readlines()
    with open(cs_file, 'r') as f_cs:
        cs_lines = f_cs.readlines()

    if len(java_lines) != len(cs_lines):
        print(f"Mismatch in line counts for {java_file} ({len(java_lines)} lines) and {cs_file} ({len(cs_lines)} lines).")
        return False
    
    print(f"Files {java_file} and {cs_file} are aligned with {len(java_lines)} lines each.")
    return True

if __name__ == "__main__":
    data_dir = "."
    print("Checking alignment for training data...")
    check_alignment(f'{data_dir}/train.java-cs.txt.java', f'{data_dir}/train.java-cs.txt.cs')
    print("Checking alignment for validation data...")
    check_alignment(f'{data_dir}/valid.java-cs.txt.java', f'{data_dir}/valid.java-cs.txt.cs')
    print("Checking alignment for test data...")
    check_alignment(f'{data_dir}/test.java-cs.txt.java', f'{data_dir}/test.java-cs.txt.cs')
    print("Checking alignment for debug data...")
    check_alignment(f'{data_dir}/data/debug/train.java-cs.txt.java', f'{data_dir}/data/debug/train.java-cs.txt.cs')
