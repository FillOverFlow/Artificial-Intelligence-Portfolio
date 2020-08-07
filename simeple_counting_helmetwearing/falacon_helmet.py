from tkinter  import *
from tkinter  import messagebox 
from tkinter  import filedialog 
from tkinter  import colorchooser
from couting_tracking_helmet import *
import os

def fopen(event):
    #file = filedialog.askopenfile() #return ที่อยู๋ของไฟล์ที่เราเลือก
    file  = filedialog.askdirectory()
    arr   = os.listdir(file)
    print(arr)
    i     = 0
    while i< len(arr):
        
        name = 'D:/Electron/falcon_detecter/engine/test_video/'+arr[i]
        print (name)
        i += 1
        counting_helmet(name)
        if i > len(arr):
            print('หมดวิดิโอที่ ให้นับแล้ว')
        

        #if file:
        #file_name = file.name
        #print(file.name)*/

def counting_helmet(filename):
    
    cap   = cv2.VideoCapture(filename)
    if (cap.isOpened() == False): #checkFile
        print("Unable to read camera feed")
    ROI_line = 160
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print("frame_width", frame_width)
    print("frame_height", frame_height)
    print("ROI_LINE",ROI_line)
    #out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
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
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #video_capture = WebcamVideoStream('Original.mp4',
    #                                   width=args.width,
    #                                    height=args.height).start()
    fps = FPS().start()
    currentFrame = 0
    
    persondot = []
    threshold = 0.15
     
    people_notwearing  = 0
    people_wearinghelmet = 0
    
   
    while (1):

        ret,frame = cap.read()
        if ret == True:
            #frame=frame[1:720, 1100:1272]
            input_q.put(frame)

            t = time.time()
        
            image_np2 = output_q.get()
            outputVedio  = cv2.cvtColor(image_np2, cv2.COLOR_BGR2RGB)
            boxes2   = output_q2.get()
            scores2  = output_q3.get()
            classes2 = output_q4.get()
           
            #write video
            
            
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
                        #เงื่อนไข ชนเส้นหรือหลังเส้น
                        if((top_position < ROI_line) and (left< 680)):
                                match_dot = -1
                                match_box = []
                                print("toch line")
                                print('persondot',persondot)
                                for y in range(len(persondot)):
                                    [y0_dot,x0_dot]=persondot[y]
                                    print('persondot', persondot)
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
                                    if(people_notwearing == 1):
                                        pass
                                    else:
                                        
                                        print ("position box1 y0 y1 x0 x1:",math.ceil(y0*height),math.ceil(y1*height),math.ceil(x0*height),math.ceil(x1*height))
                                        print("people_notwearing")
                                        name = './data/people_notwearing/frame' + str(currentFrame) + '_' + str(people_notwearing) + '.jpg'
                                        print ('Creating...' + name)
                                        cv2.imwrite(name, frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)])
                                        image = cv2.imread(name,cv2.COLOR_BGR2RGB)
                                        cv2.imshow('Not wearing ',image)
                           
                elif classes2[0][x] == 2:
                    
                    print("people_wearinghelmet")
                    
                    #if(currentFrame % 20 == 0):
                    name = './data/people_wearinghelmet/frame' + str(currentFrame) + '_' + str(people_wearinghelmet) + '.jpg'
                    print('Counting peoplewearing')
                    
                    print ('Creating...' + name)
                    [y0, x0, y1, x1] = boxes2[0][x]
                    top_position = math.ceil(y0*height)
                    left = math.ceil(x0*width)
                    print(left)
                    print (top_position)
                    if((top_position < ROI_line) and (left< 680)):
                            
                                
                                match_dot = -1
                                match_box = []
                                print("toch line")
                                
                                
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
                                    people_wearinghelmet+= 1
                                    print ("position box1 y0 y1 x0 x1:",math.ceil(y0*height),math.ceil(y1*height),math.ceil(x0*height),math.ceil(x1*height))
                                    print("people_wearinghelmet")
                                    name = './data/people_wearinghelmet/frame' + str(currentFrame) + '_' + str(people_notwearing) + '.jpg'
                                    print ('Creating...' + name)
                                    cv2.imwrite(name, frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)])
                                    image = cv2.imread(name,cv2.COLOR_BGR2RGB)
                                    cv2.imshow('wearing helmet',image)

              
                elif classes2[0][x] == 3:
                    print("License_plate")
                    License_plate += 1
                    if(currentFrame % 4 == 0):
                        name = './data/License_plate/frame' + str(currentFrame) + '_' + str(License_plate) + '.jpg'
                        print ('Creating...' + name)
                        [y0, x0, y1, x1] = boxes2[0][x]
                        cv2.imwrite(name, frame[math.ceil(y0*height):math.ceil(y1*height), math.ceil(x0*width):math.ceil(x1*width)])
                        image = cv2.imread(name,cv2.COLOR_BGR2RGB)
                        cv2.imshow('License_plate',image)
                 
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            #cv2.imshow('object detection',cv2.resize(outputVedio,(800,600)))
            
            cv2.putText(outputVedio, "Person Not Helmet:"+str(people_notwearing), (800, 55), cv2.FONT_HERSHEY_DUPLEX,0.75, (0,0,255), 1)
            cv2.putText(outputVedio, "Person Wearing Helmet:"+str(people_wearinghelmet), (800, 75), cv2.FONT_HERSHEY_DUPLEX,0.75, (255,0,0), 1)
            cv2.line(outputVedio,(30,ROI_line),(780,ROI_line),(0,255,0),5) #ROI line    point1 (x,y) point2(x,y) if make balance line y = 250
            #out.write(outputVedio)    
            
          
            cv2.imshow('object detection',outputVedio)
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
    #out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    gui = Tk()
    gui.title("Falcon Helmet Detection")
    gui.geometry("300x300")
    
    l1 = Label(text = "Welcome Counting Program" ,fg="red")
    l2 = Label(text = "Choose file video >> ")
    b1 = Button(text = "Choose")
    b1.bind('<Button-1>',fopen)
    b1.grid(row =1)
    
    l1.grid(row =0,column=0 )
    l2.grid(row =2 )
    e1 =Entry()
    #e1.grid(row =0,column =1)
    gui.mainloop()
    
        
