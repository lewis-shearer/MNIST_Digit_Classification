# Handwritten digit classifier

A convolutional neural network that classifies handwritten digits from the MNIST dataset, built with TensorFlow and Keras in a Jupyter notebook. The trained model is saved as `mnist.h5` and used by two small OpenCV scripts: one where you draw a digit with the mouse, and one that classifies a frame from a webcam.

## How it works

1. The notebook loads MNIST with `tf.keras.datasets.mnist.load_data()`, pads each 28 by 28 image with zeros to 32 by 32, and scales the pixel values to the range 0 to 1.
2. The model follows the LeNet-5 layout: a 5 by 5 convolution with 6 filters, 2 by 2 max pooling, a 5 by 5 convolution with 16 filters, 2 by 2 max pooling, then dense layers of 120 and 84 units and a 10-way softmax output. All hidden layers use ReLU. It has 61,706 trainable parameters.
3. It is compiled with the Adam optimiser and sparse categorical cross-entropy loss.
4. Training runs 300 steps of `train_on_batch`, each on a random batch of 100 training images. The variable is called `epochs`, but each step is one mini-batch, so 300 steps cover 30,000 images. Every 5 steps the model is evaluated on a random sample of 500 images from the test set.
5. The notebook then plots training loss and accuracy, shows predictions for six test images, plots a confusion matrix over the full 10,000-image test set, and saves the model to `mnist.h5`.

`app.py` opens a 280 by 280 black canvas. When you release the mouse button, it converts the drawing to greyscale, resizes it to 32 by 32, scales it to 0 to 1, and writes the predicted digit in the top-left corner. `webcam.py` does the same preprocessing on the whole webcam frame when you press `c`.

## Results

The saved model `mnist.h5` scores 97.86% accuracy on the full 10,000-image MNIST test set.

During training the notebook checks accuracy on random 500-image samples of the test set. Over the 60 checks in the saved output these ranged from 96% to 99.6%, and the last one was 98.2%. The execution counts show the training cell was run more than once on the same model, so the saved output only covers the last run. The progress line prints "Epoch # 2" every time because it uses `(1+1)` where it should use `(i+1)`.

## Repository layout

| Path | What it is |
|---|---|
| `mnist.ipynb` | Data loading, model definition, training, plots and confusion matrix |
| `mnist.h5` | Trained model saved by the notebook, loaded by both scripts |
| `app.py` | Draw a digit with the mouse and see the prediction |
| `webcam.py` | Capture a webcam frame and predict a digit |

## Running it

There is no requirements file. The scripts need TensorFlow, OpenCV and NumPy:

```
pip install tensorflow opencv-python numpy
python app.py
```

In `app.py`, draw with the left mouse button and press `q` to quit. In `webcam.py`, press `c` to capture and classify a frame, any key to carry on, and `q` to quit.

The notebook also imports matplotlib, pandas, scikit-learn, `tensorflow_datasets` and `kagglehub`. Only matplotlib is used, but the others need to be installed for the first cell to run. MNIST is downloaded automatically by Keras.

## Limitations

- `webcam.py` does not work well. It feeds the whole frame to the model without cropping to the digit or inverting it, while MNIST digits are white on a black background and digits on paper are usually dark on light.
- `app.py` predicts each time the mouse button is released, so digits drawn in more than one stroke are classified after the first stroke. It also draws the predicted digit onto the canvas, which then becomes part of the next input.
- Both scripts only handle a single digit.
