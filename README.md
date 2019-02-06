# eyes_emergency
## Create emergency stop based on eye close event

### REQUIRES:
**head_detection** package in the same workspace

### BRINGUP:
```
roslaunch eyes_emergency eyes_emergency.launch
```

### OUTPUT TOPIC:
```
/eyes_emergency
```

### DETAILS:
The eyes_emergency node receives an image stream from a topic published by the camera node in the head_detection package      
The node then publishes the emergency signals over the topic *"/eyes_emergency"* of datatype boolean.    
Brings up a preview of the images received and the processing variables.        
If either or both of the eyes are closed, a *"True"* value is published and *"EMERGENCY!!!"* text is displayed on the screen, otherwise *"False"* value is published.      

### DISCLAIMER:
This package is based on the work of Adrian Rosebrock found at  https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/ and used under the GNU public license.