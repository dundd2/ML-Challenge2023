# ML-Challenge 2023-2024
##  Most creative solution prize Submission Brief Write-up
I am truly honoured to have received the **Most Creative Solution** prize for my submission. This recognition validates my approach to incorporating AI into my work, which I didn't initially expect to be accepted so positively.
![Most Creative Solution Prize Certificate](https://github.com/user-attachments/assets/374948b5-7e4b-4e24-a2ac-9dd0094fe1c9)

## Acknowledgments
This project leverages a Large Language Model (LLM) known as **"im-a-good-gpt2-chatbot."** While I did not write any of the code myself, I was responsible for training the model and converting the Python code to Jupyter Notebook format. It has been an exciting journey to witness the advancements in LLMs as of May 2024.

## Motivation

As a first-year university student with no prior experience in machine learning, I decided to participate in this challenge to explore the field further. I had not selected machine learning as part of my coursework, making this an invaluable opportunity to see if it aligns with my interests.

I joined the challenge after receiving an email from **"Informatics Peer Assisted Learning,"** which mentioned the chance to earn a **Limited Edition Signed Ian Mackie Certificate.** The thought of obtaining such a unique recognition was incredibly motivating!

## Discovering "im-a-good-gpt2-chatbot"

In May 2024, I learned about **"im-a-good-gpt2-chatbot,"** a powerful yet little-known LLM that is available for free at **[chat.lmsys.org](https://chat.lmsys.org/)**. Speculation suggests it may be linked to OpenAI, but no confirmation exists as of now (09/05/2024).

Developers have indicated that this LLM surpasses the capabilities of OpenAI's GPT-4-Turbo-0409 and other predecessors. Given the opportunity to experiment with such a tool, I decided to explore its potential. My analysis of AI's impact on machine learning jobs suggested that it may play a significant role in the future.

## Project Summary

I dedicated approximately 5 hours to this project, and I am curious if my solution achieved one of the highest accuracy rates. Although I occasionally feel like I might be "cheating" by using an LLM, there were no restrictions on the GitHub page regarding its use. Since this project is not part of my coursework, I feel confident moving forward.

This experience has been both enjoyable and educational, enhancing my understanding of machine learning despite my lack of formal study in the area. Ultimately, I am thrilled about the opportunity to earn the **Limited Edition Signed Ian Mackie Certificate!**

## Results

I regret to mention that I utilized an LLM (which some may consider "cheating") to attain a **99.89% accuracy rate.** I plan to upload a screenshot demonstrating how I prompted the LLM to achieve this outcome.

## Reflection

According to the guidelines on the GitHub README page, I believe **"Creative solutions"** could include the use of LLMs, although I'm not entirely certain. Nevertheless, it was an enjoyable experience, especially since **"any computer science approach is valid."** Perhaps I can even say I used machine learning to facilitate my machine learning efforts! 

I find it amusing and exciting that, despite having never formally studied machine learning, I was able to achieve a high accuracy rate in this challenge. It remains a significant hurdle for many machine learning students, yet I managed to succeed (albeit with some assistance).

---
## Task
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
Deadline: April 29th 2024
Submit using this link: (https://docs.google.com/forms/d/e/1FAIpQLSe1fRH39M6WCmksYrPqIi_xRKz_hlALWSCIrbduFhzYBrDnsg/viewform?usp=sf_link)

## Prizes
We have two prizes, one for best accuracy and one for most creative solution. Both these prizes are FitBit watches! 

# Bibliography
Karas, M., Urbanek, J., Crainiceanu, C., Harezlak, J., & Fadel, W. (2021). Labelled raw accelerometry data captured during walking, stair climbing and driving (version 1.0.0). PhysioNet. https://doi.org/10.13026/51h0-a262.
