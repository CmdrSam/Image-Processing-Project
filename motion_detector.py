import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE


class MotionDetector:
    LAPLACIAN = 2
    DETECT_DELAY = 1
    # Initializing object variables
    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []

    def detect_motion(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        # Printing the coordinates on video
        for p in coordinates_data:
            coordinates = self._coordinates(p)   # getting coordinates sent to the object
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)    
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]  # x coordinate(column) of all points(rows)
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]  # y coordinate(column) of all points(rows)
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)   # saving coordinates of rectangle around the car

            # Making Image for the zero matrix for selected car parking area
            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint16),  #
                [new_coordinates],
                contourIdx=-1,  # Drawing all contours
                color=255,      # Line styling below this
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)   # Initially setting status of all parking to empty
        times = [None] * len(coordinates_data)     # Array to store timeframes for each frame
        
        # Detecting the car parking
        while capture.isOpened():
            result, frame = capture.read()  
            if frame is None:     # Checking if the vidieo feed is still on
                break

            if not result:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))
            
            # Bluring the image and Greying
            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):   # loops (index, coordinate) on the coor. data
                status = self.__apply(grayed, index, c)    # Finding if car is in parking

                # If nothing is changed don't add timestamp
                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue
                
                # Check if status has changed after last change in status
                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue
                
                # Add time when the status of parking is changed
                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            # Coloring the contours
            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if statuses[index] else COLOR_BLUE
                draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)

            open_cv.imshow(str(self.video), new_frame)

            # Checking for user input to Exit the program
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()

    # Finds if the car has entered the zone
    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]  # Slicing img to get Region of Intrest
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)     # Applying laplacian to img
        logging.debug("laplacian: %s", laplacian)

        # Getting coordinates of the rectangle/parking area
        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        # Finding Average pixels in an parking lot
        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN  # Gives true/ false
        print(np.mean(np.abs(laplacian * self.mask[index])))
        logging.debug("status: %s", status)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])


    # Checking if the new status of parking is changed
    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]     

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass
