from flask import Flask,render_template,request,send_from_directory

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import decode_predictions

app = Flask(__name__)
model =VGG19()
@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods =['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/'+ imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Predict the class of the image
    predictions = model.predict(image)
    label = decode_predictions(predictions)
    label = label[0][0]  # Get the top prediction

    classification = '%s (%.2f%%)' % (label[1],label[2]*100)

    return render_template('index.html', prediction = classification,image_path=image_path)


@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory('./images', filename)
    

if __name__ =='__main__':
    app.run(debug=True)