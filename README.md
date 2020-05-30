# Billboard100
An End to End Machine Learning Project to Predict the Billboard 100 Status of Songs
### By: Chinmay Gharpure, Siddharth Kumaran, Arham Baid

The folder contains three subfolders, which are labeled “data”, “feature engineering”, and “machine learning”. The significance of the files in each of the folders is listed below. 

“data”:

“billboard_top100.csv” - These are the Top 100 Billboard songs from 1990 to 2019 that we scraped from the Billboard website. 

“spotify_songs.csv” - This is the set of ~19,000 Spotify songs that we downloaded from a Kaggle competition 

“data_cleaning.ipynb” - This is a Jupyter notebook that contains code that merges the data from “billboard_top100.csv” and “spotify_songs.csv” into “merged_data.csv”

“merged_data.csv” - CSV formed as a result of running data_cleaning.ipynb. 


“feature engineering” :

“spotify_api_2.ipynb” - This is a Jupyter notebook that contains the code which retrieves Spotify song features and lyrics using the Spotify and MusixMatch API’s. 

“FINAL_DATA_2.csv” - This is a .csv file with all of the Spotify features for all of the songs we used for our models. 

“Filtering Pop and Decade Datasets - IEOR142Project.ipynb” - This is a Jupyter notebook that filters FINAL_DATA_2.csv in order to get three separate datasets. These datasets are music in the 2010’s decade, pop music, and pop music from the 2010’s decade. These decades are used for the models in the “machine learning” section. 

“PopVsClassicalAverages.ipynb” - This is a Jupyter notebook that calculates the averages of a variety of Spotify features for pop versus classical music. This wasn’t used in any of the models, but was referred to in the project report. 

“all_data_with_features_lyrics.csv “ - This is a .csv file with all of the Spotify data plus the lyrics of the songs. This is used later for NLP analysis in the machine learning section. 


“machine learning” : 

“Datasets for Genre/Decade Submodels” - This contains all of the relevant datasets necessary for any of the machine learning models in the machine_learning folder.

“FINAL_DATA_2.csv” - This is a .csv file with all of the Spotify features for all of the songs we used for our models. 

“IEOR 142 - Initial Models.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all of the songs. 

“IEOR 142 - Pop Music.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all pop songs. 

“IEOR 142 - Music by Decade.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all songs from the 2010’s decade. 

“IEOR 142 - Pop Music in the 2010’s Decade.R” - This model is a .R file that contains the code for the 10 machine learning models(and other graphics) we applied on all pop songs from the 2010’s decade.
