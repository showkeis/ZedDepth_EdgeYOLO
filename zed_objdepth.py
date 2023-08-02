import sys
import numpy as np
import pyzed.sl as sl
import cv2
from loguru import logger
from association import *
import torch
import numpy as np
import random
from edgeyolo.detect import Detector
from PIL import ImageFont, ImageDraw, Image
import math 
import datetime
import time
help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

#======================== AI Setting ==============================
WEIGHTS = "./models/edgeyolo_tiny_lrelu_coco.pth"
CONF_THRES = 0.5 #0.25
NMS_THRES = 0.4
DET_THRES = 0.05
IOU_THRES = 0.05
MP = False
FP16 = True
NO_FUSE = False
INPUT_SIZE = [480,480] # h, w
TRT = False
LEGACY = False
USE_DECODER = False
BATCH = 1
NO_LABEL = False
SAVE_DIR = "./output/detect/imgs/"
FPS = 99999
IS_GPU = True
TRACKING = False
TRACKLINE = False
TARGET_BOX = True
TARGETNUM = [0] # 67 cellphone
NORMFLAME = 10
STATUS = "NONE"
TRIMCENTER = False
RESIZE = False
RESIZE_HW = [480,640]
DRAW_ADD_TEXT = True

DEPTH_METHOD = "nearest"  # "center" or "nearest"
DEPTH_MINIMUM_DISTANCE = 200
DEPTH_MAXIMUM_DISTANCE = 20000

def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)
    

def process_key_event(zed, key) :
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")

def get_color(id):
    # Set the seed for the random number generator
    random.seed(id)
    
    if id == 1:
        return (75, 0, 130)  # Indigo for ID 1
    else:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color for other IDs

def draw(imgs, results, class_names, point_cloud, depth_mat, line_thickness=2, draw_label=True, DEPTH_METHOD="center",fps_counter=None):
    corner_radius = 10
    transparency = 0.3
    single = False
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
        single = True
    out_imgs = []
    tf = max(line_thickness - 1, 1)
    for img, result in zip(imgs, results):
        if result is not None:
            for *xywh, obj, conf, cls in result:
                intbox = [int(i) for i in xywh]
                color = get_color(int(cls))

                # Calculate object's center
                center_x = int((intbox[0] + intbox[2]) / 2)
                center_y = int((intbox[1] + intbox[3]) / 2)

                # Define the corners of the rectangle
                xmin, ymin, xmax, ymax = intbox[0], intbox[1], intbox[2], intbox[3]

                # Get the depth map of the rectangle
                depth_map_rectangle = depth_mat[ymin:ymax, xmin:xmax] # assuming that the fourth channel represents depth

                # Compute the minimum depth in the rectangle from depth map
                if depth_map_rectangle.size > 0:  # make sure the array is not empty
                    min_depth = np.nanmin(depth_map_rectangle)  # depth_map_rectangle is now 2D, so no need for [..., 3]

                    # Get the indices of the minimum depth pixel
                    min_y, min_x = np.unravel_index(np.nanargmin(depth_map_rectangle), depth_map_rectangle.shape)  # now, this should return two values
                else:
                    print("Warning: depth_map_rectangle is empty.")
                    min_depth = np.nan
                    min_y, min_x = center_y, center_x

                # Compute the distance based on the method
                if DEPTH_METHOD == "center":
                    # Compute the distance of the center point
                    err, point_cloud_value = point_cloud.get_value(center_x, center_y)
                    if err == sl.ERROR_CODE.SUCCESS:
                        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                             point_cloud_value[1] * point_cloud_value[1] +
                                             point_cloud_value[2] * point_cloud_value[2])
                elif DEPTH_METHOD == "nearest":

                    # Compute the x and y positions of the minimum depth in the original image
                    min_x += xmin
                    min_y += ymin

                    # Set the distance to the minimum depth
                    distance = min_depth

                    # Draw a circle at the nearest point
                    cv2.circle(img, (min_x, min_y), corner_radius, color, thickness=line_thickness)

                # Draw the distance text
                if not np.isnan(distance):
                    cv2.putText(img, f"Distance: {int(distance)} mm", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                
                # Draw the corners
                cv2.ellipse(img, (intbox[0] + corner_radius, intbox[1] + corner_radius), (corner_radius, corner_radius),
                            180, 0, 90, color, thickness=line_thickness)
                cv2.ellipse(img, (intbox[2] - corner_radius, intbox[1] + corner_radius), (corner_radius, corner_radius),
                            270, 0, 90, color, thickness=line_thickness)
                cv2.ellipse(img, (intbox[0] + corner_radius, intbox[3] - corner_radius), (corner_radius, corner_radius),
                            90, 0, 90, color, thickness=line_thickness)
                cv2.ellipse(img, (intbox[2] - corner_radius, intbox[3] - corner_radius), (corner_radius, corner_radius),
                            0, 0, 90, color, thickness=line_thickness)

                # Draw diagonal lines
                cv2.line(img, (xmin, ymin), (xmax, ymax), (128, 128, 128), 1)
                cv2.line(img, (xmin, ymax), (xmax, ymin), (128, 128, 128), 1)

                # Draw the rounded rectangle
                intbox_poly = np.array([
                    [intbox[0] + corner_radius, intbox[1]],
                    [intbox[2] - corner_radius, intbox[1]],
                    [intbox[2], intbox[1] + corner_radius],
                    [intbox[2], intbox[3] - corner_radius],
                    [intbox[2] - corner_radius, intbox[3]],
                    [intbox[0] + corner_radius, intbox[3]],
                    [intbox[0], intbox[3] - corner_radius],
                    [intbox[0], intbox[1] + corner_radius],
                ])
                sub_img = img.copy()
                cv2.fillPoly(sub_img, [intbox_poly], color=color)
                img = cv2.addWeighted(sub_img, transparency, img, 1-transparency, 0)

                # Get the current date and time
                now = datetime.datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

                if draw_label:
                    label = f'{class_names[int(cls)]} {obj * conf:.2f}'
                    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                    c2 = intbox[0] + t_size[0], intbox[1] - t_size[1] - 3
                    cv2.putText(img, label, (intbox[0], intbox[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                # Draw the date, time and FPS
                if fps_counter:
                    cv2.putText(img, f"Date/Time: {date_time} | FPS: {fps_counter:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        out_imgs.append(img)
    return out_imgs[0] if single else out_imgs


def main() :

    # Initialize the FPS counter
    frame_counter = 0
    fps_counter = 0
    start_time = time.time()

    # detector setup
    detector = Detector
    detect = detector(
        weight_file=WEIGHTS,
        conf_thres=CONF_THRES,
        nms_thres=NMS_THRES,
        input_size=INPUT_SIZE,
        fuse=not NO_FUSE,
        fp16=FP16,
        use_decoder=USE_DECODER
    )

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_minimum_distance = DEPTH_MINIMUM_DISTANCE
    init.depth_maximum_distance = DEPTH_MAXIMUM_DISTANCE

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.enable_fill_mode = True

    # Prepare new image size to retrieve half-resolution images
    #image_size = zed.get_camera_information().camera_resolution
    camera_infos = zed.get_camera_information()
    image_size = camera_infos.camera_configuration.resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth = sl.Mat()
    point_cloud = sl.Mat()

    key = ' '
    while key != 113 :
        err = zed.grab(runtime)

        if err == sl.ERROR_CODE.SUCCESS :

            # Increment the frame counter
            frame_counter += 1

            # Update the FPS counter every second
            if time.time() - start_time >= 1:
                fps_counter = frame_counter / (time.time() - start_time)
                frame_counter = 0
                start_time = time.time()

            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Get the depth map

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            yoloframe = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
            #yoloframe = cv2.resize(yoloframe,tuple([INPUT_SIZE[1],INPUT_SIZE[0]]))
            results = detect([yoloframe], LEGACY)[0]

            depth_mat = depth.get_data() # (H,W,1channel Depth)

            if TARGET_BOX:
                if not results is None:
                    indices = [i for i, result in enumerate(results) if result[-1] in TARGETNUM]
                    results = results[indices]
                    class_names = [str(int(r[-1])) for r in results]  # クラス名を文字列化したリスト
                    image_ocv = draw([image_ocv], [results], class_names, point_cloud,depth_mat,DEPTH_METHOD=DEPTH_METHOD,fps_counter=fps_counter)[0]
                else:
                    continue

            depth_image_ocv = depth_image_zed.get_data()

            cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)

            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()