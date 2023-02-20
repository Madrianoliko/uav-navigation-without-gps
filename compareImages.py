import cv2 as cv


def compareImagesMethod(largeMapImage, smallMapImage):
    # docelowy size obrazu wybrany na podstawie eksperymentu, w przyszłości zautomatyzować proces
    smallMapImage = cv.resize(smallMapImage, (143, 80))
    w, h = smallMapImage.shape[::-1]
    # 6 metod porównawczych, obecnie jest predykcja dla pierwszej bo return jest w pierwszym obrocie pętli
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    for meth in methods:
        largeMapImage = largeMapImage.copy()
        method = eval(meth)
        # Apply smallMapImage Matching
        res = cv.matchTemplate(largeMapImage, smallMapImage, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, bottom_right
