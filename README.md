# Billboard100

An End to End Machine Learning Project to Predict whether songs would land on the Billboard Top 100 Charts using song data from Spotify and Billboard.

### By: Chinmay Gharpure, Siddharth Kumaran, Arham Baid

## Motivation 

Often times, music producers invest in songs that don’t click with the audience and lead to major losses. These investment decisions are highly dependent on producers’ subjective opinions. The main goal of this project is to judge the quality of songs using quantitative metrics and help produce music in accordance with the audience’s tastes and preferences. The Billboard End of Year Hot 100 chart is one of the most notable representations of a song’s success. Using the Billboard charts as a metric of success, we want to help artists determine what technical music features of a song can they work on in order to produce a highly probable Billboard hit. We even broaden our scope of technical features to include lyrics which can help artists tweak their songs a little bit to better click with the audience. The applications of the project range from artists better understanding their audience to music producers generating higher return on investment for the music they produce.

## Data Collection and EDA 
We initiated the data collection process by scraping the Wikipedia page for the Top 100 charts from 1990 to 2018. Thereafter, we used a dataset of ~19000 Spotify songs from Kaggle to get a large sample of artists and song titles. This dataset was appended to the Billboard songs that we scraped from Wikipedia (removing all duplicates). In order to classify the hits and non-hits distinctly we classified a Billboard hit as 1 and a non-hit as 0. We then used Spotify’s API to get a multitude of a song’s technical audio features given the artist and song name. We looped through our dataset and added the features for songs that showed a match on both the artist name and song title and dropped the rest of the songs. Apart from that, we also collected the lyrics of every song using Musixmatch’s API. This was done to prepare the data for NLP analysis, which could potentially add more predictive power to our model. Multiple visualisations of our model like the one below in Fig 2a showed that the value of the features that classifies a song as a hit changes over time. We thereby segregated our dataset by decades.

## Model Results Summary

| Model         | Accuracy | TPR | FPR | AUC (if applicable)|
| ------------- | ------------- | --- | --- | -------------------|
| Logistic Regression | 0.8075  | 0.4976 | 0.0814 | 0.854 |
| Ridge Regression | 0.7988  | 0.4 | 0.0582 | NA |
| Lasso Regression | 0.7950  | 0.4126 | 0.0678 | NA |




## Project Structure 
The folder contains three subfolders, which are labeled “data”, “feature engineering”, and “machine learning”. The significance of the files in each of the folders is listed below. 

### data:

“billboard_top100.csv” - These are the Top 100 Billboard songs from 1990 to 2019 that we scraped from the Billboard website. 

“spotify_songs.csv” - This is the set of ~19,000 Spotify songs that we downloaded from a Kaggle competition 

“data_cleaning.ipynb” - This is a Jupyter notebook that contains code that merges the data from “billboard_top100.csv” and “spotify_songs.csv” into “merged_data.csv”

“merged_data.csv” - CSV formed as a result of running data_cleaning.ipynb. 


### feature engineering:

“spotify_api_2.ipynb” - This is a Jupyter notebook that contains the code which retrieves Spotify song features and lyrics using the Spotify and MusixMatch API’s. 

“FINAL_DATA_2.csv” - This is a .csv file with all of the Spotify features for all of the songs we used for our models. 

“Filtering Pop and Decade Datasets - IEOR142Project.ipynb” - This is a Jupyter notebook that filters FINAL_DATA_2.csv in order to get three separate datasets. These datasets are music in the 2010’s decade, pop music, and pop music from the 2010’s decade. These decades are used for the models in the “machine learning” section. 

“PopVsClassicalAverages.ipynb” - This is a Jupyter notebook that calculates the averages of a variety of Spotify features for pop versus classical music. This wasn’t used in any of the models, but was referred to in the project report. 

“all_data_with_features_lyrics.csv “ - This is a .csv file with all of the Spotify data plus the lyrics of the songs. This is used later for NLP analysis in the machine learning section. 


### machine learning: 

“Datasets for Genre/Decade Submodels” - This contains all of the relevant datasets necessary for any of the machine learning models in the machine_learning folder.

“FINAL_DATA_2.csv” - This is a .csv file with all of the Spotify features for all of the songs we used for our models. 

“IEOR 142 - Initial Models.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all of the songs. 

“IEOR 142 - Pop Music.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all pop songs. 

“IEOR 142 - Music by Decade.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all songs from the 2010’s decade. 

“IEOR 142 - Pop Music in the 2010’s Decade.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all pop songs from the 2010’s decade.
