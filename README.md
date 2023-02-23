# FRONT

## How to run it
1. `install python 3.9+`
2. `pip install -r requirements.txt`
3. `python main.py`

## Result video
https://youtu.be/i-H_lZvdhPQ

## How it works:

Proposed solution tackles the problem with navigation based only on visual data.

### The main problem:

The main problem with visual navigation is to find current position. This solution finds current object position by “searching” image taken right below the object in the base map (for example satellite map provided). The search is conducted by simple statistical image comparison. 
Details:
The mechanism used for comparison is `matchTemplate()` function from OpenCV. This function uses 6 similarity calculation methods from which we use 3 to select single most matched piece of the image.
Methods used:

TM_SQDIFF = Template Matching Square Difference

![image](https://user-images.githubusercontent.com/51478114/220901926-ca3a3b1a-0715-402e-9f02-ff05b0647cf6.png)

TM_CCOEFF = Template Matching Correlation Coefficient

![image](https://user-images.githubusercontent.com/51478114/220901991-cd5af896-c0d5-4b79-a7f5-59ce17d2daf0.png)

TM_CCORR = Template Matching Cross Correlation

![image](https://user-images.githubusercontent.com/51478114/220902024-d8675a07-fa76-4c2f-a3ed-c941295333ea.png)

When current position is found its marked on the map. 

### Problem with object rotation.

Marking the object's position after rotation is possible by rotating the sector map by the appropriate angle and matching the drone's image.
Methods used:

![image](https://user-images.githubusercontent.com/51478114/220902091-dcb70b5e-e827-410d-9d00-a65041d81be8.png)

### Navigating to “failsafe” point:
When the current position is determined the navigated object rotates to certain angle and  goes forward.
The necessary angle of rotation of the drone is calculated based on the slope coefficients of the straights (the drone's heading straight and the straight to the target point).
Calculating the angle:

![image](https://user-images.githubusercontent.com/51478114/220902149-9666f502-e530-4746-a0be-f9a64b313495.png)

![image](https://user-images.githubusercontent.com/51478114/220902158-071a03b0-37cf-4aea-9d50-6a04bb2c1fd7.png)

### Assumptions:
- Need for a top down camera
- Current satellite image

### Advantages of the solution
- Lightweight - calculation position in real time
- Independence from sensors (vision only)

### Disadvantages of the solution:
- Not available for drones without a top down camera


