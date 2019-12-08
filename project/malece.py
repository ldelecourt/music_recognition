"""
Created on Sat Nov 30 19:10:21 2019

@author: MAckenson Greffin, LEo Delecourt, CEdric Cally--caballero
"""

import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
from datetime import datetime
from mutagen.easyid3 import EasyID3
from os import listdir
from os.path import isfile, join
from pathlib import Path

# Usage : la base sera charger dans df. Faire:
# start(df,seconds)

# CONSTANTES

COLUMNS = ['sound', 'peaks']


def init_db(input_folder=".", output_file="malece.pikl"):
    # If the output file already exist, we concatenate the data.
    # df_old is an empty dataframe if it is not existing or not in good format
    df_old = load_db(output_file)
    
    folder = Path(input_folder)
    #os.chdir(path)     
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    
    spectro_list = []
    
    for f in files :
        # Check if the file is a mp3
        _, ext = os.path.splitext(f)
        # Then extract metadata and calculate vectors
        if ext == ".mp3":
            file = folder / f
            metadata = extract_meta(file)
            # Check if it is not already présent in the database
            if all(df_old["sound"]!=metadata):
                peaks = create_spec(file)
                spectro_list.append([metadata,peaks])
                print("File "+f+" added")
            else:
                print("File "+f+" already present in database ("+metadata[0]+")")
        else:
            print("File "+f+" skipped : it is not an MP3 file")
    
    df = pd.DataFrame(spectro_list, columns=COLUMNS)

    df = pd.concat([df_old, df])
    df = df.reset_index()
    df = df.drop("index",1)
    df.to_pickle(output_file)
    print("File "+output_file+" saved in "+ os.getcwd())
    return df

def load_db(db_file="malece.pikl"):
    try:
        df = pd.read_pickle(db_file)
        print("Database loaded")
        if not all(df.columns == COLUMNS):
            print("This file does not match with columns "+COLUMNS)
            df = pd.DataFrame(columns=COLUMNS)   
    except:
        df = pd.DataFrame(columns=COLUMNS)
        print("Error during loading database.")
    return df        

def create_spec(f):
    y, sr = librosa.load(f)

    S = np.abs(librosa.cqt(y=y, sr=sr))
    
    # Position of peaks :
    Slog = librosa.amplitude_to_db(S, ref=np.max)
    onset_peaks = librosa.onset.onset_strength(y, S=Slog)
    return onset_peaks

# slide_padding is an hyperparameter. It represents the size of the translation between two tests.
# If it is 1 (minimum), the prediction will be the best possible but the delay to test all the database is the worst
# For slide_padding, the time execution is 2 times faster, but the correlation score il not be perfect ...
def correl(v_test, v_music):
    n=len(v_music)
    m=len(v_test)
    i=0
    slide_padding = 4
    correlation=[]
    while (i+m)<=n:
        v_music_extract=v_music[i:(i+m)]
        correlation.append(np.corrcoef(v_test,v_music_extract)[0,1])
        i=i+slide_padding
    return max(correlation)


def extract_meta(file):
    audio = EasyID3(file)
    try:
        date = audio['date'][0]
        # Cut the date to keep only the information of Year
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').year
    except:
        date = "NA"
    
    try:
        title = audio["title"][0]
    except KeyError:
        title = "NA"
        
    try:
        artist = audio['artist'][0]
    except KeyError:
        artist = "NA"
     
    try:
        album = audio['album'][0]
    except KeyError:
        album = "NA"

    return title, artist, album, date
    
# db (Pandas dataframe) is the database of musics
def predict(vector, db):
    # Test with every musics in the database
    best = 0
    sound = "" 
    for index, row in db.iterrows():
        score = correl(vector, row['peaks'])
        if score > best:
            best = score
            sound = row['sound']
    return sound


def record_from_mic(output="tmp.wav", seconds_of_record=10, number_of_channels=1):
    import wave, pyaudio
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = number_of_channels
    fs = 44100  # Record at 44100 samples per second
    seconds = seconds_of_record

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(output, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return output

def start(df, seconds=5):
    rec = record_from_mic(seconds_of_record=seconds)
    v=create_spec(rec)
    pred = predict(v, df)
    print(pred) 
    return(pred)      
 



#df = load_db("test_small.pikl") 


##########
# tests
##########

# db.malece sera la base complete
# TODO : décommenter et lancer le code suivant pur générer la base :

        # creating the full database  
        #path = r"C:\Users\Souffle-Sucre\Documents\Musics"
        #for path, subdirs, files in os.walk(path):
        #    for rep in subdirs:
        #        df = init_db(path+"\\"+rep)
        
# Autres bases pour les tests :
        
# test_small.pikl include directories 010 050
#init_db(r"C:\Users\Souffle-Sucre\Documents\Musics\000", "malece.pikl")     
    
        
#p = r"C:\Users\Souffle-Sucre\Dropbox\SISE\Python\Projet_shazam"
#os.chdir(p)
#df = load_db("test_small.pikl")

# relancer cette commande quand on change la fonction de spectrogrammage. Supprimer le fichier "test_small.pkl" avant
#df = init_db(r"C:\Users\Souffle-Sucre\Documents\Musics\050", "test_small.pikl")
#df = init_db(r"C:\Users\Souffle-Sucre\Documents\Musics\010", "test_small.pikl")
 

########
# Tests
########

# =============================================================================
# 
# # test avec un morceau de musique sur l'ordi
# v1 =create_spec(r"050264.mp3")
# pred = predict(v1[10:20],df)
# print(pred) 
#     # OK
#     
# v2 =create_spec(r"010673.mp3")
# pred = predict(v2[150:600], df)
# print(pred)
#     # OK
# 
# # test avec un record.
# vr =create_spec(r"rec1.m4a")
# pred = predict(vr, df)
# print(pred)    
#     # OK !
#     
# # test avec un record en live
# rec = record_from_mic()
# vr2=create_spec(rec)
# pred = predict(vr2, df)
# print(pred)    
# =============================================================================


#######
# Parallele predict
#######

# =============================================================================
# import multiprocessing as mp
# 
# def test(vector,db,index):
#     print(index)
# 
# def predict_1(vector,db,index_music):
#     return correl(vector,db['peaks'][index_music])
# 
# def predict_2(vector,db):
#     
#     pool=mp.Pool(mp.cpu_count())
#     
#     nrow=db.shape[0]
#     results=[pool.apply(test,args=(vector,db,index_music)) for index_music in range(0,nrow)]
#     print("aaaaaaaaa")
#     results=np.array(results)
#     
#     pool.close()
#     
#     index=np.argmax(results)
#     sound=db['sound'][index]
#     
#     return(sound)
#     
# pred = predict_2(v1, df)
# print(pred)
# =============================================================================


#######
# Ajout clustering
######

# =============================================================================
# def fit_clusters(df):
#     musics=[]
#     for index, row in df.iterrows():
#         spectro = row['peaks']
#         musics.append(spectro)
#         
#     data = pd.DataFrame(musics)
#     data = data.iloc[:,10:410]
#     data = data.fillna(0)
#     print(data)
#     #k-means sur les données centrées et réduites
#     from sklearn import cluster
#     kmeans = cluster.KMeans(n_clusters=4)
#     kmeans.fit(data)
#     #index triés des groupes
#     idk = np.argsort(kmeans.labels_)
#     clusters = pd.DataFrame(df.index[idk],kmeans.labels_[idk])
#    
#     return clusters, kmeans
# 
# 
# def pred_cluster(peaks, kmeans):
#     tmp = pd.DataFrame(peaks)
#     tmp = tmp.transpose()
# #    tmp = tmp.iloc[:,410:810]
#     cluster = kmeans.predict(tmp)
#     return cluster
# 
# def wich_cluster(v_music, kmeans):
#     n=len(v_music)
#     i=0
#     clusters=[]
#     while (i+400)<=n:
#         v_music_extract=v_music[i:(i+400)]
#         clusters.append(pred_cluster(v_music_extract, kmeans))
#         i=i+1
#     return clusters
# 
#     
# clusters, k = fit_clusters(df)
# cluster = pred_cluster(v1, k)
# print(cluster)
# 
# c = wich_cluster( df["peaks"][1], k)
# print(c)
#     
# for index, row in df.iterrows():
#     vv1 = np.array(df["peaks"][0])
#     c = wich_cluster( df["peaks"][1], k)
#     print(c)
# =============================================================================
