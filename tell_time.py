# ** tell_time.py
from datetime import datetime, timedelta
from gtts import gTTS
import io
from io import BytesIO
import time
import schedule
import playsound
from tempfile import TemporaryFile
from IPython.display import Audio

# TELL TIME PROGRAM
# To play audio text-to-speech during execution
# version 0.1
#



condition = 60

print("= * tell_time.py is running * =") 
print("= * condition => ",condition)
print("= * first start at => ",datetime.now().strftime("%H:%M"))

def call_sound():
    msg = make_msg()
    tts = gTTS(text=msg, lang='th')
    tts.save('test.mp3')
    playsound.playsound('test.mp3', True)


def make_msg():
    time = datetime.now().strftime("%H:%M")
    msg = "ตอนนี้เวลา "+time+" แล้วค่ะ"
    print("= * log =>",time)
    return msg


schedule.every(condition).minutes.do(call_sound)

while(1):
    schedule.run_pending()
    time.sleep(1)
