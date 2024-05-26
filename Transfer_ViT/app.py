import logging
import os
import sys
import time

logging.basicConfig(level=logging.DEBUG)

try:
    import tensorflow as tf
    import numpy as np
    from flask import Flask, render_template, request, jsonify
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.applications import ResNet50, VGG16
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
    from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
    from sklearn.metrics import f1_score
    import random
    import base64
    from PIL import Image
    from io import BytesIO
    import requests
    from transformers import CLIPProcessor, CLIPModel

    logging.debug("All imports were successful")

except ImportError as e:
    logging.error("ImportError: %s", e)
    sys.exit(1)

app = Flask(__name__)


# LOADING DEFAULT DATASETS

# URL to fetch the ImageNet class index to label mapping
url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
response = requests.get(url)
class_idx = response.json()
# Extract labels from the class index dictionary
imagenet_labels = [class_idx[str(i)][1] for i in range(len(class_idx))]
# Save the labels to imagenet_labels.txt
with open('imagenet_labels.txt', 'w') as f:
    for label in imagenet_labels:
        f.write(f"{label}\n")
logging.info("ImageNet labels have been saved to imagenet_labels.txt")
# Load CIFAR-10 dataset
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Load MNIST dataset
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_mnist = np.expand_dims(x_train_mnist, axis=-1)
x_test_mnist = np.expand_dims(x_test_mnist, axis=-1)
x_train_mnist = np.repeat(x_train_mnist, 3, axis=-1)
x_test_mnist = np.repeat(x_test_mnist, 3, axis=-1)
mnist_labels = [str(i) for i in range(10)]
# Load Fashion MNIST dataset
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
x_train_fashion = np.expand_dims(x_train_fashion, axis=-1)
x_test_fashion = np.expand_dims(x_test_fashion, axis=-1)
x_train_fashion = np.repeat(x_train_fashion, 3, axis=-1)
x_test_fashion = np.repeat(x_test_fashion, 3, axis=-1)
fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Global variable to control the classification process
classification_active = False

# LOADING DEFAULT MODELS

def load_model(model_cls, preprocess_input):
    base_model = model_cls(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess_input

# Load pretrained models once at the start
resnet_model, preprocess_input_resnet = load_model(ResNet50, preprocess_input_resnet)
vgg_model, preprocess_input_vgg = load_model(VGG16, preprocess_input_vgg)

def resize_image(image, target_size=(32, 32)):
    return tf.image.resize(image, target_size).numpy()

def classify_image(image_array, model, preprocess_input):
    image_resized = resize_image(image_array)
    processed_img = preprocess_input(np.expand_dims(image_resized, axis=0))
    start_time = time.time()
    predictions = model.predict(processed_img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return predictions, elapsed_time

def get_top_predictions(predictions, top_k=3):
    class_indices = np.argsort(predictions[0])[::-1][:top_k]
    return [(i, predictions[0][i]) for i in class_indices]

# Load CLIP model once at the start
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def clip_classify(image_array, text_labels):
    image = Image.fromarray(image_array.astype('uint8'))
    inputs = clip_processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    start_time = time.time()
    outputs = clip_model(**inputs)
    end_time = time.time()
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    elapsed_time = end_time - start_time
    return probs, elapsed_time


# FLASK WEB INTERFACE

@app.route('/')
def index():
    return render_template('select_dataset.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    global classification_active
    if request.method == 'POST':
        logging.debug("Received a POST request for classification")
        file = request.files.get('file')
        
        # Select the labels to use:
        label_set = request.form.get('label_set')
        if label_set == 'cifar10':
            labels = cifar10_labels
        elif label_set == 'imagenet':
            logging.debug("Selected label set: ImageNet")
            try:
                labels = list(open('imagenet_labels.txt').read().splitlines())  # Ensure you have a file with ImageNet labels
            except Exception as e:
                logging.error("Error loading ImageNet labels: %s", e)
                return render_template('select_dataset.html', error="Error loading ImageNet labels.")
        elif label_set == 'mnist':
            labels = mnist_labels
        else:
            labels = cifar10_labels  # Default to cifar10 if not specified

        # Classification with custom image:
        if file:
            logging.debug("Processing uploaded image")
            # Handle uploaded image
            try:
                image = Image.open(file.stream)
                image = image.convert('RGB')  # Ensure the image is in RGB format
                image_array = np.array(image)

                # Process and classify the image using each model
                resnet_predictions, resnet_time = classify_image(image_array, resnet_model, preprocess_input_resnet)
                vgg_predictions, vgg_time = classify_image(image_array, vgg_model, preprocess_input_vgg)
                clip_probs, clip_time = clip_classify(image_array, labels)

                resnet_top_preds = get_top_predictions(resnet_predictions)
                vgg_top_preds = get_top_predictions(vgg_predictions)
                clip_top_preds = get_top_predictions(np.array([clip_probs[0]]))

                # Map predictions to labels
                resnet_top_preds = [(labels[idx], prob) for idx, prob in resnet_top_preds]
                vgg_top_preds = [(labels[idx], prob) for idx, prob in vgg_top_preds]
                clip_top_preds = [(labels[idx], prob) for idx, prob in clip_top_preds]

                # Convert image to base64 for displaying in HTML
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                result = {
                    'image_data': img_str,
                    'resnet_predictions': resnet_top_preds,
                    'vgg_predictions': vgg_top_preds,
                    'clip_predictions': clip_top_preds,
                    'resnet_time': resnet_time,
                    'vgg_time': vgg_time,
                    'clip_time': clip_time,
                }

                return render_template('upload_result.html', result=result)
            except Exception as e:
                logging.error("Error processing uploaded image: %s", e)
                return render_template('select_dataset.html', error="Error processing uploaded image.")

        # Classification with default datasets:
        else:
            logging.debug("Processing dataset classification")
            # Handle dataset classification
            try:
                # Select dataset and labels
                dataset = request.form['dataset']
                num_images = int(request.form['number'])
                if dataset == 'cifar10':
                    x_test = x_test_cifar10
                    y_test = y_test_cifar10
                    dataset_labels = cifar10_labels
                elif dataset == 'mnist':
                    x_test = x_test_mnist
                    y_test = y_test_mnist
                    dataset_labels = mnist_labels
                elif dataset == 'fashion_mnist':
                    x_test = x_test_fashion
                    y_test = y_test_fashion
                    dataset_labels = fashion_labels
                else:
                    return render_template('select_dataset.html')

                # Draw random images from dataset:
                indices = np.random.choice(len(x_test), num_images, replace=False)
                results = []
                resnet_correct_predictions = 0
                vgg_correct_predictions = 0
                clip_correct_predictions = 0

                resnet_top3_correct_predictions = 0
                vgg_top3_correct_predictions = 0
                clip_top3_correct_predictions = 0

                y_true = []
                resnet_y_pred = []
                vgg_y_pred = []
                clip_y_pred = []

                classification_active = True

                for idx in indices:
                    if not classification_active:
                        break
                    test_image = x_test[idx]
                    actual_label_index = y_test[idx][0] if dataset == 'cifar10' else y_test[idx] 
                    actual_label = dataset_labels[actual_label_index]

                    y_true.append(actual_label_index)

                    resnet_predictions, resnet_time = classify_image(test_image, resnet_model, preprocess_input_resnet)
                    vgg_predictions, vgg_time = classify_image(test_image, vgg_model, preprocess_input_vgg)
                    clip_probs, clip_time = clip_classify(test_image, labels)

                    resnet_top_preds = get_top_predictions(resnet_predictions)
                    vgg_top_preds = get_top_predictions(vgg_predictions)
                    clip_top_preds = get_top_predictions(np.array([clip_probs[0]]))
                    
                    resnet_is_correct = resnet_top_preds[0][0] == actual_label_index
                    vgg_is_correct = vgg_top_preds[0][0] == actual_label_index
                    clip_is_correct = clip_top_preds[0][0] == actual_label_index

                    resnet_top3_correct = actual_label_index in [pred[0] for pred in resnet_top_preds]
                    vgg_top3_correct = actual_label_index in [pred[0] for pred in vgg_top_preds]
                    clip_top3_correct = actual_label_index in [pred[0] for pred in clip_top_preds]

                    resnet_y_pred.append(resnet_top_preds[0][0])
                    vgg_y_pred.append(vgg_top_preds[0][0])
                    clip_y_pred.append(clip_top_preds[0][0])

                    if resnet_is_correct:
                        resnet_correct_predictions += 1
                    if vgg_is_correct:
                        vgg_correct_predictions += 1
                    if clip_is_correct:
                        clip_correct_predictions += 1

                    if resnet_top3_correct:
                        resnet_top3_correct_predictions += 1
                    if vgg_top3_correct:
                        vgg_top3_correct_predictions += 1
                    if clip_top3_correct:
                        clip_top3_correct_predictions += 1

                    img_pil = Image.fromarray(test_image.astype('uint8'))
                    buffered = BytesIO()
                    img_pil.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    results.append({
                        'image_data': img_str,
                        'actual_label': actual_label,
                        'resnet_predictions': resnet_top_preds,
                        'vgg_predictions': vgg_top_preds,
                        'clip_predictions': clip_top_preds,
                        'resnet_correct': resnet_is_correct,
                        'vgg_correct': vgg_is_correct,
                        'clip_correct': clip_is_correct,
                        'resnet_time': resnet_time,
                        'vgg_time': vgg_time,
                        'clip_time': clip_time,
                    })

                resnet_accuracy = (resnet_correct_predictions / num_images) * 100
                vgg_accuracy = (vgg_correct_predictions / num_images) * 100
                clip_accuracy = (clip_correct_predictions / num_images) * 100

                resnet_top3_accuracy = (resnet_top3_correct_predictions / num_images) * 100
                vgg_top3_accuracy = (vgg_top3_correct_predictions / num_images) * 100
                clip_top3_accuracy = (clip_top3_correct_predictions / num_images) * 100

                resnet_f1 = f1_score(y_true, resnet_y_pred, average='macro')
                vgg_f1 = f1_score(y_true, vgg_y_pred, average='macro')
                clip_f1 = f1_score(y_true, clip_y_pred, average='macro')

                return render_template('classification_results.html',
                        labels=dataset_labels,
                        results=results,
                        resnet_accuracy=resnet_accuracy,
                        vgg_accuracy=vgg_accuracy,
                        clip_accuracy=clip_accuracy,
                        resnet_top3_accuracy=resnet_top3_accuracy,
                        vgg_top3_accuracy=vgg_top3_accuracy,
                        clip_top3_accuracy=clip_top3_accuracy,
                        resnet_f1=resnet_f1,
                        vgg_f1=vgg_f1,
                        clip_f1=clip_f1)
            
            except Exception as e:
                logging.error("Error during dataset classification: %s", e)
                return render_template('select_dataset.html', error="Error during dataset classification.")
    return render_template('select_dataset.html')

# Classification with custom labels. 
@app.route('/classify_custom', methods=['GET', 'POST'])
def classify_custom():
    if request.method == 'POST':
        logging.debug("Received a POST request for custom label classification")
        file = request.files.get('file_custom')
        custom_labels_only = request.form.get('custom_labels_only')

        if file and custom_labels_only:
            labels = [label.strip() for label in custom_labels_only.split(',')]
            logging.debug("Using custom labels for classification: %s", labels)

            try:
                image = Image.open(file.stream)
                image = image.convert('RGB')
                image_array = np.array(image)

                clip_probs, clip_time = clip_classify(image_array, labels)
                clip_top_preds = get_top_predictions(np.array([clip_probs[0]]))
                clip_top_preds = [(labels[idx], prob) for idx, prob in clip_top_preds]

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                result = {
                    'image_data': img_str,
                    'clip_predictions': clip_top_preds,
                    'clip_time': clip_time,
                }

                return render_template('classify_custom.html', result=result)
            except Exception as e:
                logging.error("Error processing uploaded image with custom labels: %s", e)
                return render_template('select_dataset.html', error="Error processing uploaded image with custom labels.")

    return render_template('select_dataset.html')

@app.route('/stop_classification', methods=['POST'])
def stop_classification():
    global classification_active
    classification_active = False
    return jsonify({'status': 'Classification stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
