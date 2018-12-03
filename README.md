# Data Programming 1st Coursework
In this project, I upload relevant packages in Python, clean a dataset, make a brief description of the variables it has, and  normalize data to do data minning in another coursework.

## Dataset

The dataset used is a sample a fraction of the FIFA18 complete dataset. It contains basic information about the best 100 footballers acording to the FIFA18 videogame.  

## Variables

Name, Age, Nationality *, Overall *, Potential *, Club, Value, Wage

Index: Ranking

* FIFA18 evaluated these categories

## Exploring and Tidyng Data

In this section, the main objective is to prepare the data to do analysis in the following sections. Before anything, I displayed the complete dataset to detect any clear anomalies using pandas. with the following command (df = pd.read_excel("FIFAFORDP.xlsx", index_col = 0) and running "df" in the next command line.

#### Replacing zeros with NaNs

To begin with the exloratory analysis, data visualization is a powerful tool to see if there are anamalies in the data. The dataset showed that it contained many zeros that presumably werent reliable, so I decided to replace them with NaNs. This way, this would make a more trustworthy description of the variables. 

#### Answering which columns have complete data

Before doing any type of analysis, checking which categories have more data is a prudent thing to do. (I used df.isnull().sum())

## Exploratory Analysis

I ran different commands such as df.describe() in order summarize the main characteristics of the sample I would be working. There were no surprising facts about the sample, I attribute this because my sample is on the best 100 footballers, the difference in the quality of each of them is not much and they have to be in a very reduced age interval in order to be as fit as possible.

#### Special Questions

I wanted to expand my basic but meaningful knowledge on the dataset with simple commands. I chose two questions: (1) Which are the 10 countries with most top 100 players?, (2) Which are the most repeated ages?

With this two questions I confirmed that Spain now has the most talented footballers, because of different cultural, and other types of reasons. The answers for which ages are most repeated is very straight forward: (ages between 25-30) which is a common interval in elite sports.

## Normalizing Data

Before normalizing the data I tryed to look for graphical evidence of relationship between different variables. Since the data was not standarized, I could not see any clear trend. The main purpose of this is to see for patterns in the next coursework (where I will do proper data minning).

I also plotted some relations to confirm common sense relations among variables. First table shows "Value vs Wage", the second on "Overall vs Wage", and the last one, "Potential vs Value". The three graphs show a clear direct relation. 


# Conclusions

This is just a starting point for Coursework 2. This project contains a part of the exploratory analysis, data cleaning and normalization of the FIFA18 footballers database. 

# Further work

The complete FIFA18 dataset is inmense (17k observations and 78 columns). I have been working with that dataset, I still think I should use a reduced version of the table (with about 2k observations). There are many questions I am interested in answering but overall I would like to know which players would be the most advisable to hire with a certain budget (I was thinking of clustering analysis) . Also I would like to know profiles of countries and interesting comparissons, such as which countries have better younger and older talent, or which positions fit best specific footballer traits such as speed in order to get have some knowledge about this topic.





