from flask import Flask, render_template, Response
from Camera import Handgesture, EmotionDetection

app = Flask(__name__)

# Render the index page
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

# Function to generate frames for the Hand Gesture video feed
def generate_hand(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for the Hand Gesture video feed
@app.route("/video_Hand")
def video_Hand():
    return Response(generate_hand(Handgesture()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Render the Hand Gesture video page
@app.route("/handgesture")
def handgesture():
    return render_template("video.html")

# Function to generate frames for the Emotion Detection video feed
def generate_emotion(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for the Emotion Detection video feed
@app.route("/video_emotion")
def video_emotion():
    return Response(generate_emotion(EmotionDetection()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Render the Emotion Detection video page
@app.route("/EmotionDetection")
def emotion_detection():
    return render_template("video.html", data="emotion")

# Main method to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
 