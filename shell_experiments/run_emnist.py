from concurrent.futures import ThreadPoolExecutor
import subprocess
import time

# Variables from the new shell script
max_concurrent_processes = 6
commands=()


# Function to execute a command
def execute_command(cmd):
    process = subprocess.Popen(cmd, shell=True)
    print(f"Process {process.pid} started")
    return process.pid

# Function for multi_run from the new shell script
def multi_run():
    current_processes = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_processes) as executor:
        futures = []
        for cmd in commands:
            while current_processes >= max_concurrent_processes:
                time.sleep(1)  # Waiting for a process to end
                current_processes -= 1  # Decrease count when a process ends
                print(f"current_processes = {current_processes}")

            # Submitting the command for execution
            future = executor.submit(execute_command, cmd)
            futures.append(future)
            current_processes += 1
            print(f"current_processes = {current_processes}")
            time.sleep(10)

        # Waiting for remaining processes to finish
        for future in futures:
            future.result()
            current_processes -= 1
            print(f"current_processes = {current_processes}")

    print("All programs have finished execution.")

# Example usage
# multi_run()
