#* importing different libraries  
import csv
import copy
from gtts import gTTS
import os
import pyglet
from io import BytesIO
import threading
from playsound import playsound
import platform
import cv2 as cv
import mediapipe as mp
from customtkinter import *
from utils import CvFpsCalc
from model import KeyPointClassifier
from collection import get_args,calc_bounding_rect,calc_landmark_list,pre_process_landmark,draw_landmarks,draw_info
from colors import *
from imagepath import *
from runprogram import open_main_ui
from PIL import Image,ImageTk
is_playing =True

#*****************************Exits the current Window*****************# 
def exit ():
    root.destroy()
    open_main_ui()

#*************************Closes the Camera*****************# 
def on_closing ():
    global is_playing
    is_playing = False
    canvas.image = None
    canvas.update()

#*****************change between Fullscreen and window **********#
def toggle_fullscreen(event=None):
    state = not root.attributes('-fullscreen')
    root.attributes('-fullscreen', state)


def main():
    #******************* Argument parsing ****************************#
    global is_playing
    is_playing = True
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    #************** Camera preparation***************************#
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #************* Model load **********************************#
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mode = 0
    if is_playing:
        close_btn.place(relx = 0.96, rely=0.065,anchor = "center")
    while True:
        
        fps = cvFpsCalc.get()

        #******  Process Key (ESC: end) ********************#
        if not is_playing: 
            close_btn.place(relx = 50, rely=50,anchor = "center") # ESC
            break

        #******************* Camera capture ****************#
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  #* Mirror display
        debug_image = copy.deepcopy(image)

        #************* Detection implementation ************#
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #************ processes hand tracking data **************************#
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # *Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # *Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                #* Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                #* Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                #* Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )
        number=0
        debug_image = draw_info(debug_image, fps, mode,number)
        debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        img = Image.fromarray(debug_image)
        imgtk = ImageTk.PhotoImage(image=img)
        image.flags.writeable = False
    
        canvas.create_image(0,0,anchor=NW,image=imgtk)
        canvas.image = imgtk
        canvas.update()
    cap.release()
    

#* draws a rectangle
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        #* Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

 #* Global variable sign is used to compare with hand_sign_text for the first time
sign = " "
lock = threading.Lock()
def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
   
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text

        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        
        threading.Thread(target=play_audio, args=(hand_sign_text,)).start()
    
    return image


#***************Plays audio of sign language *************************#    
def play_audio(text):
    global sign
    
    if sign != text:
        
        sign = text
        lock.acquire()
        speech = gTTS(text=text, lang='en', tld='us')
        speech_bytes = BytesIO()
        speech.write_to_fp(speech_bytes)
        speech_bytes.seek(0)
        temp_file = "temp.mp3"
        with open(temp_file, 'wb') as f:
            f.write(speech_bytes.read())
        if platform.system()=="Darwin":
            playsound(temp_file)
        else:
            music = pyglet.media.load(temp_file, streaming=False)
            music.play()
        
        os.remove(temp_file)
        lock.release()



#************** ui design starts *********************#

#********** main window configuration ************#
root = CTk()
root.title("Speech to Sign")
root.configure(fg_color = main_bg_color)
root.geometry("1920x1080")
root.after(0,root.wm_state,"zoomed")
root.attributes('-fullscreen', True)
root.bind('<Escape>', toggle_fullscreen)

#*****************************Back To Home button *******************************************#
exit_image = CTkImage(light_image=Image.open(exit_path),
                                  dark_image=Image.open(exit_path),
                                  size=(30, 30))
exit_btn = CTkButton(master=root,hover_text_color=button_bg_color,text_color=button_bg_color,
                     width=40,height=40,text="",fg_color="transparent",
                     image=exit_image,hover=False,command=exit)
exit_btn.place(relx=0.075,rely=0.05,anchor="ne")
label5 = CTkLabel(root,text="Back To Home Page",text_color=button_bg_color)
label5.place(relx=0.095,rely=0.09,anchor="ne")

#***************frame Starts**********************************************#
frame = CTkFrame(master=root,width=980,height=960,fg_color = main_bg_color)
frame.place(relx = 0.5,rely = 0.5,anchor="center")

canvas = CTkCanvas(frame,width=960,height=540,bg=frame_bg_color,borderwidth=0,highlightthickness=0)
canvas.place(relx=0.1,rely=0.3)
logo_image = CTkImage(light_image=Image.open(logo_path),
                                  dark_image=Image.open(logo_path),
                                  size=(200, 75))
image_label = CTkLabel(frame,image=logo_image,text="")
image_label.place(relx = 0.5,rely = 0.15,anchor = "center")
label3 =CTkLabel(master= frame,text="Sign To Speech TRANSLATION",text_color=brown_color,font=("Arial",-24))
label3.place(relx = 0.5 , rely=0.2,anchor = "center")
btn1 =CTkButton(master=frame,text="Open Camera",hover_text_color=white_color,corner_radius=20,
                fg_color=button_bg_color,border_color=button_bg_color,border_width=2,
                hover_color=button_bg_color,text_color=white_color,width=150,height=40,
                font=("Arial",-14),command=main)
btn1.place(relx = 0.5 , rely=0.25,anchor = "center")
close_image = CTkImage(light_image=Image.open(close_path),
                                  dark_image=Image.open(close_path),
                                  size=(30, 30))

close_btn =CTkButton(master=canvas,hover_text_color=button_bg_color,width=40,height=40,text="",
                     fg_color="transparent",image=close_image,hover=False,command=on_closing)

#***************** Frame Ends *************************************************************#
root.mainloop()
