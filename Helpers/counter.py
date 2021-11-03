def grab_contours(cnts):
    """ TBD """
    
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def get_contour_bbox(mask):
    """Function to return the bounding box (tl, br) for a given mask"""
    # get contour -> there should be only one
    counts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = grab_contours(counts)

    if len(contour) == 0:return None
    else:
        contour = contour[0]

    # get extreme coordinates
    tl = (tuple(contour[contour[:,:,0].argmin()][0])[0], tuple(contour[contour[:,:,1].argmin()][0])[1])
    br = (tuple(contour[contour[:,:,0].argmax()][0])[0], tuple(contour[contour[:,:,1].argmax()][0])[1])

    return tl, br

