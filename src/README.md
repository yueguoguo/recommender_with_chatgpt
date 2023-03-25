# System Requirements

To run this project, you need:

* Python 3.x installed on your system
* Conda package manager installed on your system

# Installation

* Clone the project repository to your local machine:

    ```sh
    git clone https://github.com/yueguoguo/recommender_with_chatgpt.git
    ```

* Navigate to the project directory:

    ```sh
    cd recommender_with_chatgpt
    ```

* Create a new conda virtual environment:

    ```sh
    conda create --name movie-recommender
    ```

* Activate the newly created conda virtual environment:

    ```sh
    conda activate movie-recommender
    ```

* Install the required libraries by running the following command in your terminal:

    ```sh
    pip install -r requirements.txt
    ```

# Running the Recommender System

    python recommender.py

If the precision of the model is above 0.7, the model object will be returned.
Otherwise, an exception will be raised.

# Updating the System

If you need to update the Python libraries in the project, update the
requirements.txt file with the latest versions of the libraries you need.

* Activate the conda virtual environment:

    ```sh
    conda activate movie-recommender
    ```

* Update the installed libraries by running the following command:

    ```sh
    pip install -r requirements.txt --upgrade
    ```

## Uninstalling the System

* Deactivate the conda virtual environment:

    ```sh
    conda deactivate
    ```

* Delete the conda virtual environment:

    ```sh
    conda env remove --name movie-recommender
    ```

* Delete the project directory from your system:

    ```sh
    rm -rf recommender_with_chatgpt
    ```
