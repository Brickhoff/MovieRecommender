# MovieRecommender
This is a movie recommender using python and tensorflow. The GUI is built by Tkinter.

## Install Dependencies
-	Python: 3.6
-	Tkinter
-	Tensorflow: 1.13.1
-	scikit-learn
-	Pandas
-	Numpy

## Dataset
The dataset used in this project is the MovieLens 1M dataset ml-1m.zip, which contains 6040 users with 1,000,209 reviews (ratings) on 3952 movies. This dataset has three files: users.dat, movies.dat and ratings.dat.

## How to Run
The processed data `preprocess.p` file is uploaded in the Github repository, so there is no need to run the `data_download.py` file and `data_processing.py` file to download data and process it. 
Since Github doesn’t allow file has size more than 100MB, I need to compress the preprocess.p file to `prepocess.zip` file which only is 11.7MB. Therefore, after you download or cloned this repository, you need to unzip `prepocess.zip` file.
However, if you want to test it by yourself, you can run those files. In `data_dowload.py`, You need to change`data_dir = './'` to the directory you want to store the data. In `data_processing.py`, you need to change the directory to load the corresponding data and dump the processed data into pickle file in the directory you want. Be careful, there are 7 lines in `data_processing.py` you need to modify. Therefore, the easiest way is to use the already processed data file.
After you get the processed data, you can run the following steps.

1)	In `model_traning.py` file, change `save_dir = 'F:/movieRecommender/save'` to your own directory to store the trained model and checkpoint.
2)	Run `python model_traning.py` to train the model.
3)	In `movie_recommender.py` file, change `save_dir = 'F:/movieRecommender/save'` to the same directory you used in step 1.
4)	The calculated `movie_matrics.py` file is uploaded in the Github. However, if you want to calculate the movie_matrics by yourself, you need to firstly change `save_dir = 'F:/movieRecommender/save'` to your own directory and then run the `matrix_generation.py` file. In `movie_recommender.py` file, you need to change `movie_matrics = pickle.load(open('F:/movieRecommender/movie_matrics.p', mode='rb'))` to the correct directory where you stored the `movie_matrics.p` file. You also need to change directory for `user_matrics` in `movie_recommender.py` file.
5)	Run `python movie_recommender.py` to open the GUI.
6)	Type an integer number in range 1 to 3952 in the required filed, click **“Generate Recommendations From Chosen Movie”** button to generate recommendations.
7)	Click **“Reset”** button to clear the current result.
8)	Type an integer number I range 0 to 6040 in the required filed, click **“Generate Recommendations For Chosen User”**  button to generate recommendations.
9)	Click **“Reset”** button to clear the current result.
10)	 You can repeat step 6 to step 9 to find out more recommendations.



