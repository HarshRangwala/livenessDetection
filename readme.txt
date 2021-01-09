faceliveness.py 
This file is to map all the facial landmarks such as nose, mouth, eyebrows, eyes, jaw. The main aim was to
detect if mouth is open or not and it also detects when there is a head turn in particular direction*.

Future work
After some more experimentation, this work can be used to verify the user by detecting wheather the user is 
speaking or not. This file can also be further modified to detect number of blinks the user makes, 
which will help the system to detect a "live" face.

get_examples.py and liveNet.py
get_example file takes a video input of a fake face and a real face. This file extracts the face which is
our region of interest and creates a dataset of fake and real faces. This dataset is to be used by liveNet.py .
liveNet.py has convulated neural network that learns features from the fake and real images and perfrom 
classifications on it.

Future work
I have to create a program that takes real time input from
web camera and compare the results. 

=================================================================
main.py
Created: Sept 21
uploaded: October 13

This file is the main file of the whole facial recognition
system. This file currently detects person. Its a refined version 
of PersonDetection.py .

