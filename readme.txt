Within the submitted code, the website our model is hosted on can be run through 
using a python script call.  One way to do this is through the terminal.  Before 
opening the file, python must be installed, as well as the following libraries: 
librosa, matplotlib, numpy, seaborn, tensorflow, keras, flask, flask_cors, and 
json.  

Now open the ‘MusicWebsite’ file, and then run ‘py server.py’. This will then 
initiate the server locally and print out a http address to copy into a web 
browser.  The website can then be traversed normally.  For the statistics tab, 
it only demonstrates the test Naïve Bayes model, which is not testable for users. 
The CNN tab does provide usability for the user.  It can be tested by uploading 
any size wav file. Clicking upload and then submit will then return a predicted 
genre. 

Our dataset is also too large to contain it on one page.  The csv file is very 
large, as well as the number of images. The ‘music.csv’ file will be included in 
the ‘MusicWebsite’ file.
