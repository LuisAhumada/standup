from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import pandas as pd
import os

#----------------------------------------------------------------------------------------------------
# 1. Downloading Transcripts from YouTube Links
#----------------------------------------------------------------------------------------------------

# same urls as you had... we can make a text file maybe that has the urls
links = ["https://www.youtube.com/watch?v=KijAPJXjg8c", "https://www.youtube.com/watch?v=Fv3fkcCrn6k", "https://youtu.be/jQHN1ipLPdY"]

transcripts = []

for i in links:
    video_id = i.split("v=")
    if len(video_id) == 1:
        video_id = i.split("e/")

    # get transcripts
    transcripts.append(YouTubeTranscriptApi.get_transcript(video_id[1]))

#----------------------------------------------------------------------------------------------------
# 2. Read in Laugh Data
#----------------------------------------------------------------------------------------------------

file_dict = dict()

# you might need to change this path and get rid of "standup"
# I had one extra folder layer on my comp
for subdir, dirs, files in os.walk(os.getcwd() + "/standup/Results/"):
    for file in files:
        file_name = file.split(".")[0].lower()
        csv = pd.read_csv(subdir + file)
        file_dict[file_name] = csv

#----------------------------------------------------------------------------------------------------
# 3. Match Up Laugh Data to Text Data
#----------------------------------------------------------------------------------------------------

i = 0
for show in file_dict:
    previous_laugh_time = 0
    text_before_laugh = []
    for laugh_time in file_dict[show]['end']:
            text = ""
            for entry in transcripts[i]:
                time = entry['start'] + (entry['duration']/6)
                if time < laugh_time:
                   if time > previous_laugh_time:
                        text += " " + entry['text']
                else:
                    break
            text_before_laugh.append(text)
            previous_laugh_time = laugh_time
            print(text)
    file_dict[show]['joke'] = text_before_laugh
    i = i + 1
print(file_dict)

#----------------------------------------------------------------------------------------------------
# 4. Perform NLP Analysis on the Text
#----------------------------------------------------------------------------------------------------

# need a way to figure out subject
# maybe do POS tagging and keep the nouns or something