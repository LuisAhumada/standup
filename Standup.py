import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------------------------------
# 1. Downloading YouTube files to .wav
#----------------------------------------------------------------------------------------------------

links = ["https://www.youtube.com/watch?v=KijAPJXjg8c", "https://www.youtube.com/watch?v=Fv3fkcCrn6k", "https://youtu.be/jQHN1ipLPdY"]
for i in links:
    os.system("python YouTube_to_WAV.py {}".format(i))


#----------------------------------------------------------------------------------------------------
# 2. Renaming .wav files
#----------------------------------------------------------------------------------------------------

directory = ("AudioFiles")
for file in os.listdir(directory):
    file_n = str(file)
    name_file = file_n.split(" ")[0]
    os.rename(os.path.join(directory, file), os.path.join(directory, name_file + '.wav'))


#----------------------------------------------------------------------------------------------------
# 3. Running Laughter detection algorithm on each .wav file
#----------------------------------------------------------------------------------------------------

for filename in os.listdir(directory):
    os.system("python segment_laughter.py AudioFiles/{} models/model.h5 my_folder 0.8 1".format(filename))


#----------------------------------------------------------------------------------------------------
# 4. Convert results to pandas Dataframe
#----------------------------------------------------------------------------------------------------

results = ("Results")
df_list = []
for file in os.listdir(results):
    csv = pd.read_csv("Results/{}".format(file))
    df_list.append(csv)

for df in df_list:
    df.drop('Unnamed: 0', axis= 1, inplace=True)
    df["Duration"] = df["end"] - df["start"]
    print(df)