# senti-app
# Sentiment Analysis Web Application

## Overview

Welcome to the Sentiment Analysis Web Application! This project utilizes a pre-trained LSTM model to analyze the sentiment of user-provided text. The application is built using Flask for the backend and Keras/TensorFlow for the machine learning model. The frontend is designed with Bootstrap for a clean and responsive user interface.

## Features

- **Sentiment Analysis:** Classifies input text as Positive or Negative using a pre-trained LSTM model.
- **Emoji Feedback:** Displays an emoji based on the sentiment of the text.
- **Responsive Design:** Built with Bootstrap for a modern and responsive user experience.
- **Interactive UI:** Includes animated icons and a visually appealing layout.

## Getting Started

To get started with the Sentiment Analysis Web Application, follow these steps:

### Prerequisites

Ensure you have Python 3.10 and the required Python packages installed. You can install the dependencies using the provided `requirements.txt` file.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/nitesh-bhadoriya/senti-app.git
    cd senti-app
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download or ensure the pre-trained model file `optimized_lstm_sentiment_model.h5` is in the project root directory.

### Running the Application

1. Start the Flask development server:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000` to access the application.

### Usage

- Enter the text you want to analyze in the input field and click "Submit".
- The application will display the sentiment of the text along with the probability and an emoji corresponding to the sentiment.

## Project Structure

- `app.py`: Main Flask application file.
- `static/`: Directory for static files (CSS, images).
- `templates/`: Directory for HTML templates.
- `requirements.txt`: List of required Python packages.
- `optimized_lstm_sentiment_model.h5`: Pre-trained LSTM model file.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests. For any issues or feature requests, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, you can reach out to me:

- [GitHub Profile](https://github.com/nitesh-bhadoriya/senti-app/tree/main)

---

Thank you for checking out my project!
