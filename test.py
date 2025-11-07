import cv2

def _annotate_images(exterior_img, wrist_img):
    exterior_annotated = exterior_img.copy()
    wrist_annotated = wrist_img.copy()

    h, w = exterior_annotated.shape[:2]

    # exterior annotations
    # can
    can_center = (int(9/16*w)+3, int(9/16*h)-5)
    cv2.circle(exterior_annotated, can_center, 7, (255, 0, 0), 2)
    # can_text = (int(8/16*w), int(10/16*h)+5)
    # cv2.putText(exterior_annotated, "can", can_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.putText(exterior_annotated, "1", can_center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    # mug
    mug_center = (int(10/16*w), int(8/16*h)+2)
    cv2.circle(exterior_annotated, mug_center, 7, (0, 0, 255), 2)
    # mug_text = (int(11/16*w)-6, int(8/16*h)+2)
    # cv2.putText(exterior_annotated, "mug", mug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(exterior_annotated, "2", mug_center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # wrist annotations
    # can
    can_center = (int(10/16*w), int(9/16*h)-6)
    cv2.circle(wrist_annotated, can_center, 16, (255, 0, 0), 2)
    # can_text = (int(10/16*w), int(7/16*h))
    # cv2.putText(wrist_annotated, "can", can_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.putText(wrist_annotated, "1", can_center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    # mug
    mug_center = (int(7/16*w)-2, int(9/16*h)-4)
    cv2.circle(wrist_annotated, mug_center, 20, (0, 0, 255), 2)
    # mug_text = (int(5/16*w), int(7/16*h)-5)
    # cv2.putText(wrist_annotated, "mug", mug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(wrist_annotated, "2", mug_center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return exterior_annotated, wrist_annotated

exterior_img = cv2.imread("exterior_img.png")
wrist_img = cv2.imread("wrist_img.png")

exterior_annotated, wrist_annotated = _annotate_images(exterior_img, wrist_img)

cv2.imwrite("exterior_annotated.png", exterior_annotated)
cv2.imwrite("wrist_annotated.png", wrist_annotated)
