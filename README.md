# Road noise prediction

A project that shows how road noise could be predicted using different kinds of supervised learning models. LightGBM, XGBoost, BiLSTM, LSTM, BiGRU, GRU and Linear Regression are used.
The data is the IDMT Traffic dataset. It contains 9,362 2-second long stereo sound recordings of passing vehicles of different types at different locations, as well as 8,144 background noise recordings.
The .wav data is converted to a decibel value and is used as the target for the aforementioned regression models. The goal is to predict the dB level from 8 input variables. The results show that the data or approach is not good enough for this kind of prediction.

IDMT Traffic data: https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html <br>


## The Data
Recordings of passing vehicles were collected from the roadside.
<p align="center">
  <img src="https://github.com/user-attachments/assets/15229f28-ba14-4735-af51-9cebfb33d751" alt="Alt Text" width="400" height="300"><br>
  <em>Figure 1: This picture from the original paper gives a hint of the setup: Source DOI: 10.23919/EUSIPCO54536.2021.9616080</em>
</p>


The sound clips are 2 seconds long and were recorded by two stereo microphones. Here an overview of the information on every soundclip: <br>
- 4 location <br>
- 4 diffeerent speed zones <br>
- daytime: morning and afternoon <br>
- weather: dry and wet <br>
- vehicle: bus, car, motorcycle, truck <br>
- direction: coming from left or right <br>
- microphonetype: ME or SE microphone <br>
- is background: is there a passing vehicle or not <br>

The goal is to predict the dB level from these 8 input variables.

## Data cleaning
Some recordings are not usable, because of unexpected background noises. Many recordings also show a sudden spike in the amplitude due to a microphone fault. These entries have to be removed. <br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/65657881-7116-4745-b484-df14e4a04c20" alt="Alt Text" width=1000" height="300"><br>
  <em>Figure 1: Collection of examples of unusable recordings.</em>
</p>


Sound files are deleted if the amplitude reaches an appointed threshold. The thresholds are determined by vehicle type and microphone type.
<p align="center">
  <img src="https://github.com/user-attachments/assets/06053498-99fd-41b6-855d-a513643cb4be" alt="Alt Text" width="600" height="100"><br>
  <em>Figure 1: Threshold table. Values indicate the maximum allowed amplitude.</em>
</p>
These thresholds were decided by the largest outlier when looking at boxplots of the amplitudes<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/7aea884b-edb5-4d40-965e-81210b5bbac5" alt="Alt Text" width="200" height="200"><br>
  <em>Figure 1: Shows the upper whisker value in a boxplot of the amplitudes for Buses.</em>
</p>


After cleaning 472 files were deleted, leaving 17,034 files.
<p align="center">
  <img src="https://github.com/user-attachments/assets/6acd1773-1861-46ec-bfa4-c59bdedf2464" alt="Alt Text" width="300" height="500"><br>
  <em>Figure 1: This boxplot compares the dB values of entries before cleaning and after cleaning. It shows that outliers have been removed, especially at the background recordings.</em>
</p>


## Data processing
The 2 second stereo wav data is averaged into mono and an A-weighting filter is applied. 
The recording is then converted into a decibel value using the reference value 2e-5.
<p align="center">
  <img src="https://github.com/user-attachments/assets/c38bb421-7f3c-490f-8673-505ef55ad952" alt="Alt Text" width="300" height="200"><br>
  <em>Figure 1: How wav data is converted to dB</em>
</p>

To get other dB values as potential targets, octave analysis was performed. A frequency filter is applied to the wav data for each octave band. The conversion to dB afterwards is the same as before.
<p align="center">
  <img src="https://github.com/user-attachments/assets/9de61454-d2bd-465d-8cc1-43a3a9099d1d" alt="Alt Text" width="300" height="200"><br>
  <em>Figure 1: How wav data is filtered by octaves and converted to dB</em>
</p>

## The different models
LightGBM, XGBoost, BiLSTM, LSTM, BiGRU, GRU and Linear Regression are used. 

### Parameters
All the deep learning approaches BiLSTM, LSTM, BiGRU and GRU have been trained using two layers in the following structure:
- Layer 1: 128 units
- LeakyReLU with alpha=0.1
- Layer 2: 64 units
- LeakyReLU with alpha=0.2
- Dropout 0.3
- Fully connected layer: 1 unit

### Tuning
Grid search and random search for hyperparameters was used for LightGBM and XGBoost. The hyperparameter tuning showed only minimal improvements.
<p align="center">
  <img src="https://github.com/user-attachments/assets/0efbabe2-9b75-4896-8976-75707855975c" alt="Alt Text" width="500" height="300"><br>
  <em>Figure 1: Comparison of R2 scores before and after hyperparameter tuning</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b730da39-f704-4d54-af73-a7ef06e2bc07" alt="Alt Text" width="500" height="200"><br>
  <em>Figure 1: Grid search performance plot for the regularization parameters lambda_l1 and lambda_l2</em>
</p>

## Results
<p align="center">
  <img src="https://github.com/user-attachments/assets/f4bc3226-3b2d-449d-9745-6d6bc34cf2ad" alt="Alt Text" width="500" height="300"><br>
  <em>Figure 1: Comparison of R2 scores of each model.</em>
</p>
The diagram above shows the average R2 scores of each algorithm after 5-fold cross validation. Most of them perform the same. A better score is not achievable due to suboptimal data. <br>


<p align="center">
  <img src="https://github.com/user-attachments/assets/7b2bf966-3651-4588-8f2a-1c9f3ca252cc" alt="Alt Text" width="600" height="300"><br>
  <em>Figure 1: Example prediction graph with LightGBM</em>
</p>

##Contributors

Dr.Mustafa Demetgül – Academic Supervisor
 marc-heinzelmann- Student



