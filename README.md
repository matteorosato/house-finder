Property finder
==============================

This project is dedicated to retrieving and analyzing property advertisements from real estate websites. It provides a 
convenient way to gather information for analysis, research, or any other purposes related to the real estate domain.

The project is developed entirely using Python and follows object-oriented programming (OOP) practices. The initial template is provided by [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

## Who is this for?
This tool is intended for:
- Developers interested in real estate data extraction and analysis.
- Real estate agents/companies looking to integrate listing data into their systems.
- Anyone curious about exploring the world of real estate through data.

## Fair Use Disclaimer
Note that this code is provided free of charge as is. For any bugs, see the issue tracker.

## Setup and Use
To use the tool, follow these steps:

1. Ensure you have Python 3.10 and pip installed on your system.
2. Clone the repository to your local machine:
   ```shell
   git clone https://github.com/matteorosato/property-finder.git
   ```

3. Navigate to the project directory:
   ```
   cd property-finder
   ```

4. Create a virtual environment for the project:
   ```
   python -m venv venv
   ```

5. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

6. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```
7. Fill the `.env` file with the required environment variables. Use the `.env.example` file as reference.

8. Fill the `config.toml` file according to your preferences and needs.

9. Run the tool:
   ```
   TODO
   ```

## Supported websites
Currently, the following websites are supported:
 - [Idealista](https://www.idealista.com/)

### Idealista
This tool utilizes the APIs provided by Idealista to extract real estate listing data. To execute the tool, you need to obtain an API_KEY and SECRET by requesting access through the following link: [Idealista API Access Request](https://developers.idealista.com/access-request).

Please note that the free access is limited to 100 requests per month and 1 request per second. Therefore, it's important to configure the filtering parameters carefully to avoid an unnecessary number of requests.

For further information, refer to the documents located in the _references_ folder.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
