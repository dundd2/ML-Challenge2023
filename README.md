# ML-Challenge 2023-2024
##  Most creative solution prize Submission Brief Write-up
I am truly honoured to have received the **Most Creative Solution** prize for my submission. This recognition validates my approach to incorporating AI into my work, which I didn't initially expect to be accepted so positively.
![Most Creative Solution Prize Certificate](https://github.com/user-attachments/assets/374948b5-7e4b-4e24-a2ac-9dd0094fe1c9)

# Task
Your task is to classify sensor data to determine whether the movement is part of walking or driving. The data set has been prepared with columns representing time step, action (1 for walking and 4 for driving), and then the acceleration in the x, y and z. This data is a subset of the data from <a href="https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/#files">this dataset</a>. 

There is an award for best accuracy and another for the most creative solution. 

## Downloading the data
You can simply download the csv file or clone this repository. Once you have the csv file local you can open and view it with pandas:
```
import pandas as pd

df=pd.read_csv("path/to/file/movementSensorData.csv")
print(df)
```
You should get something like this:
```
       activity  time_s   lw_x   lw_y   lw_z
66077         1  660.78  0.066 -1.270 -0.020
66078         1  660.79  0.082 -1.281 -0.063
...         ...     ...    ...    ...    ...
```

To visualise this data take a look at the visualiser code provided. 

## Rules
Anyone is welcome to give it a go - but to win prizes you must be a Sussex undergraduate or master's student. 

By the end we would like you to submit your model (the file and code to load the parameters into a functioning model). For example, if you made a pytorch neural network, we would want the model.pth and then a .py or .ipynb file to open and load your network parameters. Any data preprocessing should be clear on this page as a function that takes in the parameter of the dataframe $df$ mentioned above, and outputs in the format ready for your model. SO if the data needs to be in a certain format, for example it concatenates values based on time windows, then we will need a function from you that takes in the pandas frame we give you and returns runable data. Otherwise there is no way we can test our data on your model. This is so we can see how your model performs on unseen data. This file should be less than 10MB. 

Your solution does not have to be a typical ML approach, any computer science approach is valid. If you do not want to use Python we are open to ideas, please get in contact so we can work out a way how we can test your model. If you do submit python please provide it as a .ipynb notebook. 

The main thing is to have fun with it, this is not coursework and we encourage you to explore and learn new ideas while undertaking the challenge. 

## Submission
Deadline: May 29th 2024
Submit using this link: (https://docs.google.com/forms/d/e/1FAIpQLSe1fRH39M6WCmksYrPqIi_xRKz_hlALWSCIrbduFhzYBrDnsg/viewform?usp=sf_link)

## Prizes
We have two prizes, one for best accuracy and one for most creative solution. Both these prizes are FitBit watches! 

# Bibliography
Karas, M., Urbanek, J., Crainiceanu, C., Harezlak, J., & Fadel, W. (2021). Labelled raw accelerometry data captured during walking, stair climbing and driving (version 1.0.0). PhysioNet. https://doi.org/10.13026/51h0-a262.
