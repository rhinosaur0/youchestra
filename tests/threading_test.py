import threading
import time
import queue

def continuous_output(output_queue):
    """Continuously processes and prints data from the output queue."""
    while True:
        if not output_queue.empty():
            message = output_queue.get()
            print(f"Output: {message}")
        time.sleep(0.1)  # Prevents busy-waiting

def continuous_input(input_queue):
    """Continuously reads input from the user and puts it into the input queue."""
    while True:
        user_input = input("Input something: ")
        input_queue.put(user_input)

def main():
    # Queues for communication between threads
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # Start continuous input thread
    input_thread = threading.Thread(target=continuous_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    # Start continuous output thread
    output_thread = threading.Thread(target=continuous_output, args=(output_queue,))
    output_thread.daemon = True
    output_thread.start()

    # Main loop for processing input and generating output
    print("App is running. Type something!")
    while True:
        if not input_queue.empty():
            user_input = input_queue.get()
            output_queue.put(f"You entered: {user_input}")
        time.sleep(0.1)  # Main loop delay to reduce CPU usage

if __name__ == "__main__":
    main()