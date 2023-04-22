import os
import music21 as m21
import pathlib
import json
import keras 
import numpy as np

MAPPING_PATH="mapping.json"
SEQUENCE_LENGTH=64
#SINGLE_FILE_DATASET="final_dataset1.txt"
SINGLE_FILE_DATASET="final_dataset.txt"
#SAVE_DIR="data/dataset1/"
SAVE_DIR="dataset/"
KERN_DATASET_PATH = 'LSTM/data/deutschl/essen/europa/deutschl/erk/'
#KERN_DATASET_PATH = 'LSTM/data/deutschl/essen/europa/deutschl/test/'
ACCEPTABLE_DURATIONS=[
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

def load_songs_in_kern(dataset_path):
    #go through all the files in dataset and load them with music21

    #os.walk walks through the entire folder structure
    for path,subdir,files in os.walk(dataset_path):
        songs=[]
        for file in files:
            if(file[-3:]=="krn"):
                #music21 score
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
        return songs

def has_acceptable_durations(song,acceptable_durations):
    #flattens the song structure to a list
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

#Transpose to Amin/CMaj
#DOUBT=> Cant understand music theory
def transpose(song):
    #get key from song
    parts=song.getElementsByClass(m21.stream.Part)
    measures_part0=parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    #estimate key using music21 if no key
    if not isinstance(key,m21.key.Key):
        #music21 analyses the song and generates key
        key=song.analyze("key")
    #get interval for transposition
    if key.mode=="major":
        interval=m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode=="minor":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))
    #transpose song by calculated interval
    transposed_song=song.transpose(interval)
    return transposed_song

def encode_song(song,time_step=0.25):
    #p=60,d-1.0->[60,"_","_","_"]
    encoded_song=[]
    for event in song.flat.notesAndRests:
        #handle notes
        if isinstance(event,m21.note.Note):
            symbol=event.pitch.midi
        #handle rests
        elif isinstance(event,m21.note.Rest):
            symbol="r"
        #DOUBT=>What does this line do?
        steps=int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    encoded_song=" ".join(map(str,encoded_song))
    return encoded_song

def load(file_path):
    with open(file_path,"r") as fp:
        song = fp.read()
    return song

#why delimiter?: You have to pass data to the LSTM in terms of a fixed sequence. 
#Thats why the delimiter should be the same as the number of items you want to consider in a 
# training sequrnce
def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    #with open(file_dataset_path, "w") as fp:
     #   fp.write(songs)

    return songs

def preprocess(dataset_path):
    print("Loading Songs...")
    songs=load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i,song in enumerate(songs):
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue
        song=transpose(song)
        encoded_song=encode_song(song)
        save_path=os.path.join(SAVE_DIR,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)

def create_mapping(songs,mapping_path):
    mappings={}
    #identify the vocabulary
    songs=songs.split()
    vocabulary=list(set(songs))
    for i,symbol in enumerate(vocabulary):
        mappings[symbol]=i
    #save vocabulary to a json file
    with open(mapping_path,"w") as fp:
        json.dump(mappings,fp,indent=4)

def convert_songs_to_int(songs):
    #load mappings
    int_songs=[]
    with open(MAPPING_PATH,"r") as fp:
        mappings=json.load(fp)
    #cast songs string to a list
    songs=songs.split()
    #map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs

def generate_training_sequences(sequence_length):
    #input is fixed length time series data. output is the next value
    #load songs and map to int
    songs=load(SINGLE_FILE_DATASET)
    int_songs=convert_songs_to_int(songs)
    inputs=[]
    targets=[]
    #generate training sequences
    num_sequences=len(int_songs)-sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    #one hot encode sequences dimension=>(# of sequences, sequence length,vocabulary size)
    vocabulary_size=len(set(int_songs))
    inputs=keras.utils.to_categorical(inputs,num_classes=vocabulary_size)
    targets=np.array(targets)
    return inputs,targets


def main():
    #preprocess(KERN_DATASET_PATH)
    songs=create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)
    #inputs,targets=generate_training_sequences(SEQUENCE_LENGTH)
if __name__=="__main__":
    main()