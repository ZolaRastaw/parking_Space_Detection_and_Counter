# parking_Space_Detection_and_Counter
parking Space Detection and Counter using the SIFT feature matching algorithm and Template matching.

Parking assistance with drones
There is a map of an empty parking lot and video footage from a drone flying over the area when most of the parking places are occupied.
we show the way from the entrance to the closest free place.
The project consists of some sub-tasks:
1. Creating a map on which squares indicate the parking places and a graph represents the routes
2. We pick one image from the footage and find common feature points on the image and the map. now using the SIFT feature matching algorithm we project or transform the reverse feature matrix on top of the image.
3. then we can check which squares on the map are occupied and which are free, by using traditional image processing or object detection in our case we use template matching from opencv.
4. then we'll find the free parking place with the shortest route to the entrance and display the route.
