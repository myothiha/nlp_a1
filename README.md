# NLP Assignment 1 (AIT - DSAI)

- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Evaluation](#evaluation)

## Student Information
 - Name: Myo Thiha
 - ID: st123783

## Files Structure
 - In the code folder, The Jupytor notebook files (training) can be located.
 - The 'app' folder include 
 -- `app.py` file for the web application
 -- Dockerfile and docker-compose.yaml for containerization of the application.
 -- `template` folder to hold the HTML pages.
 -- `models` folder which contains four model exports and their metadata files.

## How to run
 - Run the `docker compose up` in the app folder.
 - Then, the application can be accessed on http://localhost:8000
 - You will directly land on the "Search" page.

## Dataset
- I used `brown` dataset (category 'News') from `nltk`.

 ## Evaluation

| Model             | Window Size | Training Loss | Training Time | Semantic Accuracy | Syntactic Accuracy | Similarity (Correlation Score) |
|-------------------|-------------|---------------|---------------|--------------------|-------------------|-------------------|
| Skipgram          | 2     | 10.16       | 0 min 03 sec       | 0.00%            | 0.00%           | 0.08   |
| Skipgram (NEG)    | 2     | 2.61       | 0 min 04 sec       | 0.00%            | 0.00%           | 0.22   |
| Glove             | 2     | 44.37       | 0 min 42 sec       | 0.00%            | 0.00%           | -0.02   |
| Glove (Gensim)    | - | -       | -       | 45.89%            | 50.61%           | 0.54   |