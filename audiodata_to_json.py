import os
import librosa
import time
import json

dataset_path = "/home/akansha/RESEARCH/Datasets/ESC-50-master/audio"
json_path1 = "/home/akansha/Desktop/john-wright/A/JsonFiles/fold1.json"
json_path2 = "/home/akansha/Desktop/john-wright/A/JsonFiles/fold2.json"
json_path3 = "/home/akansha/Desktop/john-wright/A/JsonFiles/fold3.json"
json_path4 = "/home/akansha/Desktop/john-wright/A/JsonFiles/fold4.json"
json_path5 = "/home/akansha/Desktop/john-wright/A/JsonFiles/fold5.json"

SAMPLE_RATE = 22050

start = time.time()

mfcc_dict1 = {
        "label": [],
        "mfcc": []
    }

mfcc_dict2 = {
        "label": [],
        "mfcc": []
    }

mfcc_dict3 = {
        "label": [],
        "mfcc": []
    }

mfcc_dict4 = {
        "label": [],
        "mfcc": []
    }

mfcc_dict5 = {
        "label": [],
        "mfcc": []
    }



for root, dirs, files in os.walk(dataset_path):
    print ("files",len(files))
    for f in files:
        file_path = os.path.join(root, f)
        #print("file_path",file_path)
        f_split=f.split("-") #splitting the filename
        f_fold=f_split[0]    #fold number of filename
        #print("type of f_fold",type(f_fold))
        #print("f_fold",f_fold)
        t1=f_split[3].split(".")
        f_label =t1[0]       #label of filename
        #print("f_label",f_label)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc=librosa.feature.mfcc(signal,sr=sample_rate,n_mfcc=20)
        #print("mfcc",mfcc)
        
        if f_fold=='1':
            #print("yes")
            #mfcc_dict1["fold"].append(f_fold)
            mfcc_dict1["label"].append(f_label)
            mfcc_dict1["mfcc"].append(mfcc.tolist())
            
        if f_fold=='2':
            #mfcc_dict2["fold"].append(f_fold)
            mfcc_dict2["label"].append(f_label)
            mfcc_dict2["mfcc"].append(mfcc.tolist())
            
        if f_fold=='3':
            #mfcc_dict3["fold"].append(f_fold)
            mfcc_dict3["label"].append(f_label)
            mfcc_dict3["mfcc"].append(mfcc.tolist())
            
        if f_fold=='4':
            #mfcc_dict4["fold"].append(f_fold)
            mfcc_dict4["label"].append(f_label)
            mfcc_dict4["mfcc"].append(mfcc.tolist())
            
        if f_fold=='5':
            #mfcc_dict5["fold"].append(f_fold)
            mfcc_dict5["label"].append(f_label)
            mfcc_dict5["mfcc"].append(mfcc.tolist())
            
        
end = time.time()
print("time taken", end-start)        


with open(json_path1, "w") as fp:
    json.dump(mfcc_dict1, fp, indent=3)
    
with open(json_path2, "w") as fp:
    json.dump(mfcc_dict2, fp, indent=3)
    
with open(json_path3, "w") as fp:
    json.dump(mfcc_dict3, fp, indent=3)
    
with open(json_path4, "w") as fp:
    json.dump(mfcc_dict4, fp, indent=3)
    
with open(json_path5, "w") as fp:
    json.dump(mfcc_dict5, fp, indent=3)

  
                
        
    
