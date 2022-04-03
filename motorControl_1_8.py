import time
steeringAngle = 0
GPSFlag = 0
currThrottle = 0
clock = 0
universalSleepTime = .5
def throttleControl(R, O, U, S, D):
    #inputs: clock, R, O, U, S, D, inputThrottle
    #clock represents the 2 Hz, clock cycle that updates throttle control every half second
    #R: radar detection, radar detects and obstacle
    #O: image processing obstructed, camera and image classifier detects an obstacle
    #U: image processing unobstructed, camera and image classifier do not detect an obstacle
    #S: Stop sign approaching: GPS locational data indicates that golf cart is approaching a stop sign
    #D: Destination approaching: GPS locational data indicates that golf cart is approaching its end destination
    #inputThrottle: The current value of the throttle control, before changes are applied

    #outputs:
    #throttleVal: value of throttle control sent to microcontroller.
    #This value has a minimum value of -50 and a maximum value of 50.
    # -50 represents maximum braking power
    # 50 represents maximum throttle
    # 0 represents no brake and no throttle application
    # >0 means no brake applied
    # <0 means no throttle applied

    #if (clock == 1):
    global currThrottle
    throttleVal = currThrottle + ((-10 * R )+ (-3 * O) + (-2 * S) + (-2 * D) + (U))

    #else:

    #throttleVal = inputThrottle

    if throttleVal > 50:
        outputThrottle = 50
    elif throttleVal < -50:
        outputThrottle = -50
    else:
        outputThrottle = throttleVal

    currThrottle = outputThrottle
    #return outputThrottle
    print("Throttle Val: " + str(outputThrottle))
    #return outputThrottle


def steeringControl(pathArray):
    #inputs: clock, pathArray
    # clock represents the 2 Hz, clock cycle that updates throttle control every half second
    #pathArray: array of elements, from navigation system indicating what path is required to achieve destination
    #outputs: updates a global variable indicating how much the wheel should turn

    #path codes
    #0: end of path
    #1: straight line path
    #2: right turn path
    #3: left turn path

    throttleThreshold = 0
    degreeTurn = 90
    for pathElement in pathArray:
        if pathElement == 1:
            driveStraight()
        elif pathElement == 2:
            turnRight(throttleThreshold, degreeTurn)
        elif pathElement == 3:
            turnLeft(throttleThreshold, degreeTurn)
        else:
            print("Something went wrong")


def driveStraight():
    global GPSFlag
    turn(90)
    while GPSFlag != 0:
        steeringAngle = 0

def turnRight(numThrottleThreshold, degreeTurn):
    global universalSleepTime
    global clock
    global currThrottle
    throttleCount = 0
    turn(degreeTurn)
    curClock = clock
    while (numThrottleThreshold > throttleCount):
        time.sleep(universalSleepTime)
        if clock == 1:
            clock = 0
            throttleCount += currThrottle
        elif clock == 0:
            clock = 1

    turn(0)

def turnLeft(numThrottleThreshold, degreeTurn):
    global unviversalSleepTime
    global currThrottle
    global clock
    throttleCount = 0

    turn(-1 * degreeTurn)
    curClock = clock
    while (numThrottleThreshold > throttleCount):
        time.sleep(universalSleepTime)
        if clock == 1:
            clock = 0
            throttleCount += currThrottle
        elif clock == 0:
            clock = 1

    turn(0)

def turn(degrees = 0):
    #send to oscar that the cart should turn the wheel to input degrees
    print("send to oscar that the cart should turn the wheel to " + str(degrees) +" degrees")

def throttleControlTest():
    global clock
    clockCount = 0
    index        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
    Radar        = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    Obstructed   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    StopSign     = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
    Destination  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    Unobstructed = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    #inputThrottle = 0
    while(clockCount != 101):
        time.sleep(universalSleepTime)
        if clock == 0:
            clock = 1
            print("Clock count: " + str(clockCount))
            throttleControl(Radar[clockCount], Obstructed[clockCount], Unobstructed[clockCount], StopSign[clockCount], Destination[clockCount])
            clockCount += 1
        else:
            clock = 0


throttleControlTest()
