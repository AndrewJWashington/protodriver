# protodriver
Protodriver is an autonomous driver trained on Grid Autosport.  

## General information
* Does it work? Yes.  
* Will it work for other games? Yes, there's nothing specific to Grid Autosport in the code.  
* Will it work on your system? Maybe. Here's my setup for reference:
  * My system hardware is described on [PC Part Picker](https://pcpartpicker.com/list/bjXFyk).  
  * My system software is:
    * Python 3.8.3rc1
    * tensorflow 2.2
    * CUDA 10.1
    * Python packages described in requirements.txt

  
This is a weeekend project by Andrew Washington. It's far from scalable, but it's a working project. Feel free to clone/fork as you wish (in accordance with MIT license) and let me know if you have any questions. You can message me at AndrewJWashington on GitHub or just comment on the repo.  

## The Journey
Years ago, I watched Sentdex [create a self-driving GTA 5 bot](https://www.youtube.com/playlist?list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a) on YouTube and it was the coolest thing I could imagine. At the time, my python skills were not at the level to implement sucha project. However, recently I found the video again and thought "Hey, I can do that". After all, I now have a degree in Machine Learning and a couple years of experience as a Data Scientist working with python. Plus, the entire software stack I'm using has become much more user-friendly since I first watched those videos years ago.  

### Goals
* Get something running.
  * Deep RL with curiousity component? Yeah that'd be cool. Scalable and working on TPU's? Also cool. Simulate controller input to get smoother fidelity? Again, would be awesome. What do all of these have in common? They don't actually help get started. This is why I chose a basic CNN with one layer and WASD controls to get started. After that, we can play with different deep learning frameworks, image processing techniques, and fancy features.
* Still be able to play video games. 
  * I built this computer recently _to play video games_. Having an awesome deep learning machine is just a corollary. I don't want to deal with driver issues when I try to play Call of Duty: Warzone on Ultra quality.

### New things I learned along the way:
* Installing python on Windows. As basic as this sounds, all of my prior python development has been on Mac or Linux. Going to the Windows Store to install python was a pretty foriegn concept to me.
* GPU Support for tensorflow. 
  * Installation was a bit more involved than I had planned. There's several pieces of software to install and some steps even require manually moving C++ files from one directory to another.
  * Weird tensorflow errors and keeping an eye on GPU usage. 
    * I spent a few hours triaging this error combo: "could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED" and "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize". Almost all the Stack Overflow questions and GitHub Issues pointed to software version mismatches. It wasn't until I randomly checked Task Manager and saw a giant spike in GPU memory usage when starting the program, that I realized it was a GPU memory error. Turning down Grid Autosport's graphics settings settled the issue.
* Python Screen capture (PIL and Pillow)
* Python keyboard control (pyautogui and pydirectinput)
* Python/Windows user input (keyboard)

### Results
* The AI drives mostly just runs into walls. Everything up to the present has been focused on getting something running. Now that that's done, it's time to play with different deep learning and image processing techniques.

### Roadmap / potential improvements
* Deep learning framework
  * More layers
  * Careful tuning for # params vs training observations
  * Allow multiple outputs to be made at the same time
* Code clean up
* Pick an easier track
* More training data
* Move from supervised to RL framework

## Resources:
* [Sentdex's GTA 5 bot playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a)
* Create a virtual environment: https://docs.python.org/3/library/venv.html
* Pillow installation and documnentation: https://pillow.readthedocs.io/en/stable/installation.html
* OpenCV installation: https://pypi.org/project/opencv-python/
* OpenCV tutorials: https://docs.opencv.org/3.4/d7/da8/tutorial_table_of_content_imgproc.html
* PyDirectInput installation and documnentation: https://pypi.org/project/PyDirectInput/
* keyboard (python package) installation and documnentation: https://pypi.org/project/keyboard/
* Keras MNIST example: https://www.tensorflow.org/datasets/keras_example
  * It's usually easier to get started with a much simpler example and building out from there.
* https://www.tensorflow.org/install/gpu
  * ~~Who would have thought the actual documentation would be helpful?~~ Follow the steps _in order_ and make sure to read all the way through the bottom. It's easy to go to the Nvidia documentation and forget to come back to the TF documentation.
* Support matrix for Nvidia software: https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html
