# app.py

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained model
model = load_model('D:\\placements\\interships\\smartbridge data science\\project\\main\\UI\\sports_classification.h5')

# Function to preprocess the image before making predictions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the pixel values to [0, 1]
    return img_array

# Function to make predictions
def predict_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']
        
        # Save the file to a temporary location
        file_path = 'D:\\placements\\interships\\smartbridge data science\\project\\main\\UI\\static\\hoh.jpg'
        file.save(file_path)

        # Make a prediction on the uploaded image
        predicted_class = predict_class(file_path)

        # Redirect to the result page with the predicted class and image path
        return redirect(url_for('result', predicted_class=predicted_class, image_path=file_path))
    
    # If the request method is GET, render the main.html page
    return render_template('main.html')
# Define the list of class names
class_names = ['Air Hockey', 'Amputee Football', 'Archery', 'Arm Wrestling', 'Axe Throwing', 'Balance Beam', 'Barrel Racing', 'Baseball',
               'Basketball', 'Baton Twirling', 'Bike Polo', 'Billiards', 'Bmx', 'Bobsled', 'Bowling', 'Boxing', 'Bull Riding', 
               'Bungee Jumping', 'Canoe Slalom', 'Cheerleading', 'Chuckwagon Racing', 'Cricket', 'Croquet', 'Curling', 'Disc Golf', 
               'Fencing', 'Field Hockey', 'Figure Skating Men', 'Figure Skating Pairs', 'Figure Skating Women', 'Fly Fishing', 'Football',
               'Formula 1 Racing', 'Frisbee', 'Gaga', 'Giant Slalom', 'Golf', 'Hammer Throw', 'Hang Gliding', 'Harness Racing', 'High Jump',
               'Hockey', 'Horse Jumping', 'Horse Racing', 'Horseshoe Pitching', 'Hurdles', 'Hydroplane Racing', 'Ice Climbing', 'Ice Yachting',
               'Jai Alai', 'Javelin', 'Jousting', 'Judo', 'Lacrosse', 'Log Rolling', 'Luge', 'Motorcycle Racing', 'Mushing', 'Nascar Racing',
               'Olympic Wrestling', 'Parallel Bar', 'Pole Climbing', 'Pole Dancing', 'Pole Vault', 'Polo', 'Pommel Horse', 'Rings', 
               'Rock Climbing', 'Roller Derby', 'Rollerblade Racing', 'Rowing', 'Rugby', 'Sailboat Racing', 'Shot Put', 'Shuffleboard', 
               'Sidecar Racing', 'Ski Jumping', 'Sky Surfing', 'Skydiving', 'Snowboarding', 'Snowmobile Racing', 'Speed Skating',
               'Steer Wrestling', 'Sumo Wrestling', 'Surfing', 'Swimming', 'Table Tennis', 'Tennis', 'Track Bicycle', 'Trapeze', 
               'Tug Of War', 'Ultimate', 'Uneven Bars', 'Volleyball', 'Water Cycling', 'Water Polo', 'Weightlifting', 'Wheelchair Basketball',
               'Wheelchair Racing', 'Wingsuit Flying']
# Create a dictionary mapping class indices to class names
class_index_to_name = {i: name for i, name in enumerate(class_names)}

# ...

@app.route('/result/<int:predicted_class>/<path:image_path>')
def result(predicted_class, image_path):
    # Get the class name corresponding to the predicted class
    predicted_class_name = class_index_to_name.get(predicted_class, 'Unknown')

    # Render the result.html page with the predicted class and image path
    return render_template('result.html', predicted_class=predicted_class, predicted_class_name=predicted_class_name, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
