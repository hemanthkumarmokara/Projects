from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)


####################
from PIL import Image
import numpy as np
import pickle


# def preprocess_image(img):
#     """Resizes and preprocesses the image for the model."""
#     target_size = (224, 224)  # Adjust as per your pre-trained model

#     # Option 1: Convert image to a NumPy array and copy data
#     # img_array = np.array(img)  # Create a view
#     # img_array = img_array.copy()  # Copy data for resizing

#     # Option 2: Use Pillow's resize method (recommended)
#     img = img.resize(target_size)  # Resize using Pillow

#     # Convert to NumPy array
#     img_array = np.array(img)

#     # Reshape to separate channels (if your model expects RGB)
#     if img_array.ndim == 2:  # Grayscale image
#         img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
#     elif img_array.ndim == 3 and img_array.shape[-1] == 4:  # RGBA image
#         img_array = img_array[:, :, :3]  # Remove alpha channel (if not needed)

#     return img_array

# load features from pickle
with open("D:\\data analytics\\REAL-TIME PROJECTS\\image_captioning\\features.pkl", 'rb') as f:
    features = pickle.load(f)

#------
def load_vgg16():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model
#------
with open( 'D:\\data analytics\\REAL-TIME PROJECTS\\image_captioning\\captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()
# create mapping of image to captions
mapping = {}
# process lines
for line in captions_doc.split('\n'):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)
    
# def clean(mapping):
for key, captions in mapping.items():
    for i in range(len(captions)):
        # take one caption at a time
        caption = captions[i]
        # preprocessing steps
        # convert to lowercase
        caption = caption.lower()
        # delete digits, special chars, etc., 
        caption = caption.replace('[^A-Za-z]', '')
        # delete additional spaces
        caption = caption.replace('\s+', ' ')
        # add start and end tags to the caption
        caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
        captions[i] = caption
        
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
        
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq '
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == ' endseq':
            break
      
    return in_text

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    print("image_name:  ",image_name)
    image_id = image_name.split(".")[0]
    print("image_id",image_id)
    # image_id = os.path.splitext(os.path.basename(image_name))[0]

    # BASE_DIR = os.path.abspath(".")
    
    # img_path = os.path.join(BASE_DIR,"Images", image_name)
    # image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    # plt.imshow(image)
    return y_pred
#-----------            



# Define the model architecture
def build_model(max_length, vocab_size):
    # Define input layers
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    # Merge layers
    decoder1 = tf.keras.layers.add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Create the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# Load weights from the saved model
def load_weights(model, weights_path):
    model.load_weights(weights_path)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Read the image file
        
        babu = "example.jpg"
        upload_fold = "D:\\data analytics\\REAL-TIME PROJECTS\\image_captioning\\static"
        saved_path = os.path.join(upload_fold,babu)
        file.save(saved_path)
        
        kol = file
        img = Image.open(file)
        file = file.filename

        # Convert the image to a format compatible with the model (e.g., resize, preprocess)
        # Then, predict the caption
        # caption = predict_caption(img)
        # For now, let's assume predict_caption returns a dummy caption
        # caption = "A beautiful scene"
        caption = generate_caption(file)
        # Pass the image and predicted caption to the HTML template
        img_data = base64.b64encode(kol.read()).decode('utf-8')
        return render_template('home.html', img_data=img_data, caption=caption)

if __name__ == '__main__':
    # Load model weights
    max_length = 35  # Update with the actual max_length used in training
    vocab_size = 8485  # Update with the actual vocab_size used in training
    model = build_model(max_length, vocab_size)
    weights_path = 'D:\\data analytics\\REAL-TIME PROJECTS\\image_captioning\\best_model.h5'  # Update with the path to the saved weights file
    load_weights(model, weights_path)
    
    # Run Flask app
    app.run(debug=True)