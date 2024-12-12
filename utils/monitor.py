import subprocess 
import time
import sys
from datetime import datetime
import threading
from rich import print

def check_kernel_status(kernel_name):
    try:
        result = subprocess.run(['kaggle', 'kernels', 'status', kernel_name], 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

def check_stop(stop_event):
    while True:
        if input().lower() == 'stop':
            print("\nStopping monitor...")
            stop_event.set()
            sys.exit(0)

def monitor_kernel(kernel_name):
    print(f"Starting monitoring of kernel: {kernel_name}")
    print("Type 'stop' to end monitoring")
    
    stop_event = threading.Event()
    input_thread = threading.Thread(target=check_stop, args=(stop_event,), daemon=True)
    input_thread.start()
    
    try:
        while not stop_event.is_set():
            status = check_kernel_status(kernel_name)
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if status:
                print(f"[{current_time}] Status: {status}")
                
                if "complete" in status.lower():
                    print("\nKernel execution completed!")
                    time.sleep(20)
                    subprocess.run(['kaggle', 'kernels', 'pull', kernel_name])
                    break
                elif "error" in status.lower():
                    print("\nKernel execution failed!")
                    break
                    
            time.sleep(60)
    except SystemExit:
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py username/kernel-name")
        sys.exit(1)
        
    kernel_name = sys.argv[1]
    monitor_kernel(kernel_name)