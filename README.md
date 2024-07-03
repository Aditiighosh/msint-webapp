# Handwritten Digit Recognizer

This web application predicts the number that you have drawn on the canvas from 0-9 using a trained neural network model. It leverages the MNIST dataset and provides a simple interface for users to draw digits and see predictions.

## Features

- Draw digits on a canvas and get real-time predictions.
- Train the model directly from the web interface if the model does not exist.
- Visualize the input image and prediction results.

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- OpenCV
- NumPy

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Aditiighosh/handwritten-digit-system.git
cd handwritten-digit-system
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`.

3. If the model is not already trained, click the "Train Data" button to train the model.

4. Use the canvas to draw a digit and click "Predict" to see the result.

## Deployed Application

You can also test the application online at the following URL: [Handwritten Digit Recognizer](https://msint-draw.streamlit.app)

## File Structure

- `app.py`: The main Streamlit application file.
- `train.py`: Contains the function to train the model using the MNIST dataset.
- `model.keras`: The trained model file (generated after training).

## Possible Errors in Output

- **Incorrect Predictions**: The model might predict an incorrect digit due to unclear or ambiguous drawings.
- **Low Confidence**: For some drawings, the model's confidence in its prediction may be low, leading to less reliable results.
- **Canvas Issues**: The drawing on the canvas might not be correctly captured or processed, leading to errors in prediction.

## Future Plans

- **Model Improvement**: Train the model on a larger and more diverse dataset to improve accuracy.
- **Data Augmentation**: Implement data augmentation techniques to make the model more robust to variations in handwriting.
- **UI Enhancements**: Improve the user interface to provide better feedback and usability.
- **Real-time Feedback**: Provide real-time feedback and suggestions to users to help them draw clearer digits.
- **Deployment**: Deploy the application on a more robust platform to handle higher traffic and ensure better performance.

## Credits

- [Streamlit](https://www.streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

Additionally, here is the `requirements.txt` file content for reference:

```
matplotlib
tensorflow
streamlit
opencv-python
numpy
streamlit-drawable-canvas
```
