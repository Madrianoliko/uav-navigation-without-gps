import math


def calculateDegreeToDestination(currentX1, currentX2, currentX3, currentX4, destinationCenterH, destinationCenterW):

    # print(currentX1)
    # print(currentX2)
    # print(currentX3)
    # print(currentX4)

    centerFrontX = int((currentX1[0] + currentX2[0]) / 2)
    centerFrontY = int((currentX1[1] + currentX2[1]) / 2)
    # print("currentX1[0]")
    # print(currentX1[0])
    # print("currentX2[0]")
    # print(currentX2[0])
    # print("currentX1[1]")
    # print(currentX1[1])
    # print("currentX2[1]")
    # print(currentX2[1])
    centerFront = (centerFrontX, centerFrontY)
    # print(centerFront)

    centerBackX = int((currentX4[0] + currentX3[0]) / 2)
    centerBackY = int((currentX4[1] + currentX3[1]) / 2)
    centerBack = (centerBackX, centerBackY)
    # print(centerBack)

    currentCenterW = (centerFrontX + centerBackX) /2
    currentCenterH = (centerFrontY + centerBackY) /2

    # print(centerFront)
    # print(centerBack)
    if centerBack[0] - centerFront[0] == 0:
        a2 = 1
    else:
        a2 = (centerBack[1] - centerFront[1]) / (centerBack[0] - centerFront[0])
        # print("a2")
        # print(a2)
    # print("a2")
    # print(a2)

    # linia do celu
    a1 = (currentCenterH - destinationCenterH) / (currentCenterW - destinationCenterW)
    # print("a1")
    # print(a1)
    # a2 = 1
    radian = math.atan(abs((a2 - a1) / (1 + a1 * a2)))
    degree = radian * (180 / math.pi)
    return degree, centerFront, centerBack