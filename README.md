Environment setup:
-----------------
The code was tested on Ubuntu 18.04.5 LTS

- Python 3.6, Tensorflow 1.14.0 -> conda create --name env_name pip python=3.6 tensorflow=1.14.0
- opencv ->  conda install -c conda-forge opencv 
- tqdm ->  conda install -c conda-forge tqdm 
- sliding window -> pip install slidingwindow
- pycocotools ->  conda install -c conda-forge pycocotools 
- pandas -> conda install pandas
- webrtcvad -> pip install webrtcvad


If get-gaze.py does not work for you, make sure that the pose estimation is working or reinstall it by following the instructions in the original repository at https://github.com/ildoonet/tf-pose-estimation

Steps:
------
0- If you want to use the CMU trained model -> add the graph_opt.pb file to the models/graph/cmu folder:
http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb

1- run get-audio-video.py to extract the low resolution video and the audio from the original video. 
The audio will be saved in the "audio" directory.
The low res video will be saved in the "low-res-video" directory.

2- run get-speech.py to detect if there is speech or silence every 0.5 seconds.
The output is a csv file with two columns (time, speech) saved in the "classifications" directory. 
The time is in 0.5 seconds increments. 
The speech column shows 1 if speech is detected, and 0 otherwise.

3- run get-gaze.py to detect the number of face parts in each frame. 
The assumption is: you are facing the camera if both ears are showing and the nose is not too horizontally close to one of the eyes.
The output is a csv file with two columns (clock, gaze) saved in the "classifications" directory. 
The clock is in 0.5 seconds increments. 
The gaze column shows 1 if the person is looking at the screen, and 0 otherwise.

4- run get-interactions.py to get the classification results in a csv file saved in the "classifications" directory. You can also plot the results.