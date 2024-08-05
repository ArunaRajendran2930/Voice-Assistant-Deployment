# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:50:04 2020

@author: pramod.singh
"""

from flask import Flask, request, render_template, send_file
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
import matplotlib.pyplot as plt
import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from IPython.display import Audio
from pathlib import Path

app=Flask(__name__)
Swagger(app)

# Editing the descriptions: Convert to lower case and add beginning and ending
def edit_description(mapping):
    for key, desc in mapping.items():
        for i in range(len(desc)):
            x = desc[i]
            x = x.lower()
            x = x.replace('[^A-Za-z]', '')
            x = x.replace('\s+', ' ')
            x = 'beginning ' + " ".join([word for word in x.split() if len(word)>1]) + ' ending'
            desc[i] = x
    
def mapping_toword(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
    
def readImageDesc():    
    # Reading the descriptions.txt file
    with open(os.path.join('/usr/ML/app', 'captions.txt'), 'r') as f:
        next(f)
        desc_doc = f.read()

    #Mapping the descriptions to the images 
    mapping = {}
    for each_desc in tqdm(desc_doc.split('\n')):
        tokens = each_desc.split(',')
        if len(each_desc) < 2:
            continue
        image_id, desc_of = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        desc_of = " ".join(desc_of)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(desc_of)
    
    # Calling the preprocessing text function
    edit_description(mapping)

    # Appending all descriptions into a list: Each image with 5 descriptions
    img_desc = []
    for key in mapping:
        for caption in mapping[key]:
            img_desc.append(caption)
    
    return img_desc

def predict_description(model, image, tokenizer, max_length):
    in_text = 'beginning'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        desc_predict = model.predict([image, sequence], verbose=0)

        desc_predict = np.argmax(desc_predict)
        word = mapping_toword(desc_predict, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'ending':
            break

    return in_text 


#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/return-files/')
def return_files_tut():
	try:
		return send_file('/usr/ML/app/info.wav')    
        #send_file('/usr/ML/app/info.wav', mimetype="audio/wav", as_attachment=True, attachment_filename='info.wav')
	except Exception as e:
		return str(e)
        
@app.route('/predict_file',methods=["POST"])
def prediction_test_file():
    """Prediction on input test file .
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        500:
            description: Test file Prediction
        
    """    
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join('/usr/images', filename)
    file.save(file_path)    
    
    ##################
    # Pre-Processing #
    ##################        
    # Loading the VGG16
    model = VGG16()    
    
    #Changing the model: Removing the predicted values from the existing VGG16 model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)       
    features = {}
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = Path(file_path).stem
    features[image_id] = feature
    
    # Read Image Desc from the dataset
    img_desc=readImageDesc();
    
    # Tokenizing the text: finding the unique words from all the captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(img_desc)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Get the maximum description length for the padding required
    max_length = max(len(text.split()) for text in img_desc)
    
    print("Unique words in the captions are: " + str(vocab_size)+ "max_length:" + str(max_length))
    
    
    #############
    # DNN Model #
    #############    
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    output = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.load_weights('/usr/ML/app/best_model.h5')    
    
    ##############
    # Prediction #
    ############## 
    y_pred = predict_description(model, features[image_id], tokenizer, max_length)
    text = str(y_pred)
    
    print(text)

    res = text.split(' ', 1)[1]
    text = res.rsplit(' ', 1)[0]

    tts = gTTS(text)
    with open('/usr/ML/app/info.wav', 'wb+') as fp:
        pass
    fp.close()
    
    tts.save('/usr/ML/app/info.wav')
    #sound_file = 'info.wav'
    #Audio(sound_file, autoplay=True)        
    
    return text
        
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
    
    
