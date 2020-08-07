import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import math
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_helmet_wearing_(dataset_train_02)_graph-49407'
#MODEL_NAME = 'ssd_inception_v2_helmet_wearing_(dataset_train_05)_graph-20558'
PATH_TO_CKPT = 'D:/Electron/falcon_detecter/engine/ssd_mobilenet_v1_helmet_wearing_(dataset_train_02)_graph-49407/frozen_inference_graph.pb' #โมเดลที่ใช้

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'D:/Electron/gui-app/engine/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 3

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=6)#size line box
        #total_passed_vehicle = total_passed_vehicle + counter
    return image_np, boxes, scores, classes, num_detections


def worker(input_q, output_q,output_q2,output_q3,output_q4):
    # Load a (frozen) Tensorflow model into memory.
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np, boxes, scores, classes, num_detections = detect_objects(frame_rgb, sess, detection_graph)
        output_q.put(image_np)
        output_q2.put(boxes)
        output_q3.put(scores)
        output_q4.put(classes)

    fps.stop()
    sess.close()


if __name__ == '__main__':

 while(1):
    video_folder = 'D:/for_video/'
    video_file_name = 'test_helmet (29).avi'
    #video_file_name = '6.mp4'
    #cap = cv2.VideoCapture('D:')
    cap = cv2.VideoCapture('rtsp://admin:Hikvision@10.220.28.84/Streaming/channels/2/')
    
    if (cap.isOpened() == False): #checkFile
        print("Unable to read camera feed")
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print("frame_width", frame_width)
    print("frame_height", frame_height)
   
    #out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=500, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=500, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    output_q2 = Queue(maxsize=args.queue_size)
    output_q3 = Queue(maxsize=args.queue_size)
    output_q4 = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q, output_q2,output_q3,output_q4))
    ret, frame = cap.read()
    
    #Set Region of Interest (ROI)
    ROI_line_x = int(round(frame_width * 0.80, 0))
    ROI_line_y = int(round(frame_height * 0.45, 0))

    ROI_line_x2 = int(round(frame_width * 0.95, 0))
    ROI_line_y2 = int(round(frame_height * 0.8, 0))

    box1_x = int(round(frame_width * 0.6, 0))
    box1_y = int(round(frame_height * 0.1, 0))
    box2_x = int(round(frame_width * 0.6, 0))
    box2_y = int(round(frame_height * 0.15, 0))
    
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #video_capture = WebcamVideoStream('Original.mp4', width=args.width, height=args.height).start()
    fps = FPS().start()
    currentFrame = 0
    
    persondot = []
    licensedot = []
    threshold = 0.13
    last_frame_capture = 0
    last_frame_capture_license = 0
     
    people_notwearing  = 0
    people_wearinghelmet = 0
    
    while (1):

        ret,frame = cap.read()
        if ret == True:
            #frame=frame[1:720, 1100:1272]
            input_q.put(frame)

            t = time.time()
        
            image_np2 = output_q.get()
            outputVideo  = cv2.cvtColor(image_np2, cv2.COLOR_BGR2RGB)
            boxes2   = output_q2.get()
            scores2  = output_q3.get()
            classes2 = output_q4.get()

            height = len(image_np2)
            width = len(image_np2[0])
            #print("height = ", height)
            #print("width = ", width)
            #print(boxes2)
            #print(scores2)
            #print(classes2)
            count_scores = sum(1 for i in scores2[0] if i >= 0.5)
            #print(count_scores)
           
            License_plate = 0
            currentFrame+=1

            if (currentFrame - last_frame_capture > 9):
                persondot = []
            if (currentFrame - last_frame_capture_license > 9):
                licensedot = []
            
            for x in range(count_scores):
                if classes2[0][x] == 1:
                    if(currentFrame > 1):
                        
                        [y0, x0, y1, x1] = boxes2[0][x]
                        top_position = math.ceil(y0*height)
                        left = math.ceil(x0*width)
                        right = math.ceil(x1*width)
                        bottom = math.ceil(y1*height)
                        print("#######################")
                        print('frame:',currentFrame)
                        print('top_posiotion',top_position)
                        print('left',left)

                        if((top_position < ROI_line_y) and (left< ROI_line_x)):

                            match_dot = -1
                            match_box = []
                            print("toch line")
                                
                            last_frame_capture = currentFrame
                            print('persondot',persondot)
                                
                            for y in range(len(persondot)):
                                [y0_dot,x0_dot]=persondot[y]
                                distance = math.sqrt((y0_dot-y0)**2 + (x0_dot-x0)**2)
                                print('distance',distance)
                                #area = abs(top_position - bottom)*abs(left - right)
                                if((distance) < threshold):
                                    print("distance less more threshold")
                                    match_dot = y
                                    match_box = persondot[y]
                                    print('match box:',match_box)
                            persondot.append([y0, x0])

                            if(match_dot >= 0):
                                print('this is old motocycle')
                                print('remove match_box')
                                persondot.remove(match_box)
                            else:
                                print('Maybe new motocycle')
                                people_notwearing+=1
                                print ("position box1 y0 y1 x0 x1:",math.ceil(y0*height),math.ceil(y1*height),math.ceil(x0*height),math.ceil(x1*height))
                                print("people_notwearing")
                                name = './data/people_notwearing/' + video_file_name + '_frame' + str(currentFrame) + '_' + str(people_notwearing) + '.jpg'
                                print ('Creating...' + name)
                                cv2.imwrite(name, frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)])
                                image = cv2.imread(name,cv2.COLOR_BGR2RGB)

                                #Resize for Showing only
                                if (frame_height > 720):
                                    h,w,c = image.shape
                                    ratio_x = 1280/frame_width
                                    ratio_y = 720/frame_height
                                    image = cv2.resize(image, (int(round(w*ratio_x)),int(round(h*ratio_y))))
                                    
                                cv2.imshow('No Helmet',image)
                       
                elif classes2[0][x] == 2:

                    if(currentFrame > 1):
                        print("people_wearinghelmet")
                        
                        [y0, x0, y1, x1] = boxes2[0][x]
                        top_position = math.ceil(y0*height)
                        left = math.ceil(x0*width)
                        
                        if((top_position < ROI_line_y) and (left< ROI_line_x)):

                            match_dot = -1
                            match_box = []
                            print("toch line")

                            last_frame_capture = currentFrame
                            print('persondot',persondot)
                                    
                            for y in range(len(persondot)):
                                [y0_dot,x0_dot]=persondot[y]
                                distance = math.sqrt((y0_dot-y0)**2 + (x0_dot-x0)**2)
                                print('distance',distance)
                                #area = abs(top_position - bottom)*abs(left - right)
                                if((distance) < threshold):
                                    print("distance less more threshold")
                                    match_dot = y
                                    match_box = persondot[y]
                                    print('match box:',match_box)
                            persondot.append([y0, x0])

                            if(match_dot >= 0):
                                print('this is old motocycle')
                                print('remove match_box')
                                persondot.remove(match_box)
                            else:
                                print('Maybe new motocycle')
                                people_wearinghelmet+=1
                                print ("position box1 y0 y1 x0 x1:",math.ceil(y0*height),math.ceil(y1*height),math.ceil(x0*height),math.ceil(x1*height))
                                print("people_notwearing")
                                name = './data/people_wearinghelmet/' + video_file_name + '_frame' + str(currentFrame) + '_' + str(people_notwearing) + '.jpg'
                                print ('Creating...' + name)
                                cv2.imwrite(name, frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)])
                                image = cv2.imread(name,cv2.COLOR_BGR2RGB)

                                #Resize for Showing only
                                if (frame_height > 720):
                                    h,w,c = image.shape
                                    ratio_x = 1280/frame_width
                                    ratio_y = 720/frame_height
                                    image = cv2.resize(image, (int(round(w*ratio_x)),int(round(h*ratio_y))))
                                    
                                cv2.imshow('Wearing Helmet',image)
            
                elif classes2[0][x] == 3: #License Plate

                    if(currentFrame > 1):

                        print("License_plate")

                        [y0, x0, y1, x1] = boxes2[0][x]
                        top_position = math.ceil(y0*height)
                        
                        left = math.ceil(x0*width)                        

                        if ((top_position > ROI_line_y) and (top_position < ROI_line_y2) and (left< ROI_line_x2)):

                            match_dot = -1
                            match_box = []

                            last_frame_capture_license = currentFrame

                            for y in range(len(licensedot)):
                                [y0_dot,x0_dot]=licensedot[y]
                                distance = math.sqrt((y0_dot-y0)**2 + (x0_dot-x0)**2)
                                print('distance',distance)
                                #area = abs(top_position - bottom)*abs(left - right)
                                if((distance) < threshold):
                                    print("distance less more threshold")
                                    match_dot = y
                                    match_box = licensedot[y]
                                    print('match box:',match_box)
                            licensedot.append([y0, x0])

                            if(match_dot >= 0):
                                print('this is old licenseplate')
                                print('remove match_box')
                                licensedot.remove(match_box)
                            else:
                                print('New License Plate')
                                License_plate += 1

                                name = './data/License_plate/' + video_file_name + '_frame' + str(currentFrame) + '_' + str(License_plate) + '.jpg'
                                print ('Creating...' + name)

                                #Cut
                                newimage = frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)]
                                #Cut and Resize
                                #newimage = cv2.resize(frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)],(300,300))

                                cv2.imwrite(name, newimage)
                                image = cv2.imread(name,cv2.COLOR_BGR2RGB)
                                cv2.imshow('License_plate',image)

                
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            #cv2.imshow('object detection',cv2.resize(outputVideo,(800,600)))
            cv2.putText(outputVideo, "Person Wears No Helmet:"+str(people_notwearing), (box1_x, box1_y), cv2.FONT_HERSHEY_DUPLEX,0.6, (100,0,255), 2)
            cv2.putText(outputVideo, "Person Wearing Helmet:"+str(people_wearinghelmet), (box2_x, box2_y), cv2.FONT_HERSHEY_DUPLEX,0.6, (0,255,0), 2)
            cv2.line(outputVideo,(0,ROI_line_y),(ROI_line_x,ROI_line_y),(200,0,0),4) #ROI line horizontal (y)
            cv2.line(outputVideo,(ROI_line_x,0),(ROI_line_x,ROI_line_y),(200,0,0),4) #ROI line vertical (x)

            cv2.line(outputVideo,(0,ROI_line_y2),(ROI_line_x2,ROI_line_y2),(200,0,0),4) #ROI line horizontal (y)
            cv2.line(outputVideo,(ROI_line_x2,0),(ROI_line_x2,ROI_line_y2),(200,0,0),4) #ROI line vertical (x)

            #Resize for Showing only
            if (frame_height > 720):
                outputVideo = cv2.resize(outputVideo, (1280,720))
                
            cv2.imshow('object detection',outputVideo)
            #cv2.imshow('object detection',cv2.resize(outputVideo,(800,600)))
            #cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.stop()
            print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
            print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        else:
            break
    
    pool.terminate()
    cap.release()
    cv2.destroyAllWindows()

   
    
    

    
