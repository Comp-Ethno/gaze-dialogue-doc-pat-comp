import subprocess

# Extract the audio file from the original video file
command = "ffmpeg -i ./video/sample.mp4 -ab 160k -ac 2 -ar 8000 -vn ./audio/sample.wav"
subprocess.call(command, shell=True) 


# Extract the low resolution video file from the original video file
command = "ffmpeg -i ./video/sample.mp4 -vf scale=180:-2 -r 24 ./low-res-video/sample.mp4"
subprocess.call(command, shell=True)