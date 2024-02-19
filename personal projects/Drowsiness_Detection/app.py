from flask import Flask, render_template, request
import subprocess
from flask import send_from_directory
import os
import time

app = Flask(__name__)

# Use a global variable to store the subprocess object
script_process = None

@app.route('/latest_frame')
def latest_frame():
    # Get the list of frames in the output folder
    frames = os.listdir('output_frames')
    
    # Sort frames by modification time to get the latest one
    frames.sort(key=lambda x: os.path.getmtime(os.path.join('output_frames', x)), reverse=True)

    # Return the latest frame
    return send_from_directory('D:\\my_projects\\drowsiness\\output_frames', frames[0])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_script():
    global script_process

    if request.method == 'POST':
        try:
            if script_process is None or script_process.poll() is not None:
                # Run the new_main.py script if it is not already running
                script_process = subprocess.Popen(['python', 'D:\\my_projects\\drowsiness\\code\\new\\new_main.py'])
                return "Script started successfully"
            else:
                return "Script is already running"
        except Exception as e:
            return f"Error: {str(e)}"


@app.route('/halt', methods=['POST'])
def halt_script():
    global script_process

    if request.method == 'POST':
        try:
            if script_process is not None and script_process.poll() is None:
                # Terminate the script if it is running
                script_process.terminate()
                script_process.wait()  # Wait for the process to finish
                script_process = None  # Reset the subprocess object
                return "Script halted successfully"
            else:
                return "No script is currently running"
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
