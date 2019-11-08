
##            Imanol Belausteguigoitia Ibarrola                    
![Screen Shot 2019-11-05 at 15 38 01](https://user-images.githubusercontent.com/44293686/68244446-d7386580-ffe2-11e9-8ad5-46c0f3d5cbd4.png)
###  FIFA19: Exploratory Data Analysis and Predictive Modelling

## Introduction

This document analyses the **complete FIFA 19 video game dataset**, which contains attributes for every player registered in the latest edition of FIFA 19 database. The code was runned in R Studio, and it contains exploratory analysis, data mining and modelling to predict the market value of professional footballers contained in the FIFA 19 videogame developed by EA Games.

### Objective

Make a model to predict market value of fooball players.

Models like this can assist clubs to assign market value to their own football players in order to negotiate in market transfers. Clubs also have the opportunity to make assign a fair market price for players they want to buy and avoid paying too much for players.

## Workflow

* **1. Data Cleaning**
* **2. Exploratory Analysis** 
* **3. 10 facts about the FIFA 19 Complete Dataset** 
* **4. Data Mining: Manchester United Case Study**
* **5. Predicting Market Value**

---
#     1.  Data Cleaning     

#### Downloading Libraries 

```library(caret)
library(DMwR)
library(parallel)
library(doParallel)
library(dplyr)
library(mlbench)
library(dummies)
library(ggpubr)
library(xgboost)
```

Setting working directory, libraries and loading data

```setwd("~/Downloads/")
FIFA19 <- read.csv("FIFA19.csv")
load("FIFA19_ML")
```

#### Our dataset 

```View(FIFA19)```

![Screen Shot 2019-11-05 at 16 03 45](https://user-images.githubusercontent.com/44293686/68247002-ef5eb380-ffe7-11e9-9cd5-b2293e9190c1.png)

#### Dimension

```dim(FIFA19)```

* 18207  observations  
* 89 columns


#### Variables

```names(FIFA19)```

* There is a total of 87 variables:

```Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, Release Clause. ```


#### Data Completeness

```
total_na <- sum(is.na(FIFA19))
total_data <- 18207 * 89 # dim 18207 X 89
percentage_na <- total_na/total_data                
percentage_na
```
* Percentage of null values: 0.001133037% 

#### Getting rid of unuseful variables

```FIFA19[,"Photo"] <- NULL
FIFA19[,"Flag"] <- NULL
FIFA19[,"ID"] <- NULL
FIFA19[,"Club.Logo"] <- NULL
FIFA19[,29:54] <- NULL
```
###### Eliminated columns:
* Photo, Flag, ID, CLUB LOGO, REAL FACE
* LS : RB

#### Removing unuseful monetary and quantitative characters


```FIFA19[,'Value'] <- gsub("€","",FIFA19[,'Value'])
FIFA19[,'Value'] <- gsub("M","",FIFA19[,'Value'])

FIFA19[,'Wage'] <- gsub("€","",FIFA19[,'Value'])
FIFA19[,'Wage'] <- gsub("K","",FIFA19[,'Value'])

FIFA19[,'Release.Clause'] <- gsub("€","",FIFA19[,'Release.Clause'])
FIFA19[,'Release.Clause'] <- gsub("M","",FIFA19[,'Release.Clause'])

FIFA19[,'Weight'] <- gsub("lbs","",FIFA19[,'Value'])

FIFA19[,'Height'] <- gsub("'",".",FIFA19[,'Height'])

View(FIFA19)
```
###### Eliminated signs:
* "$"
* "M"
* "€"
* "K"
* "lbs"
* "."
* ","



#### Transforming text variables to numeric

```FIFA19$Release.Clause <- as.numeric(FIFA19$Release.Clause)
FIFA19$Contract.Valid.Until <- as.numeric(FIFA19$Contract.Valid.Until)
FIFA19$Wage <- as.numeric(FIFA19$Wage)
FIFA19$Value <- as.numeric(FIFA19$Value)
```
###### Transformed to numeric:
* Release.Clause
* Contract.Valid.Until
* Wage 
* Value

#### Modified columns

5 columns were modifies for technical purposes:
* **Value (1), Wage (2), Release.Clause (3)** → This variables contained money signs such as “$”,  “€”, “K”, or “M” that had to be removed to be analyzed statistically. 
* **Weight (4), Height (5)**  → This two features contained characters such as “lbs” that had to be removed in order to run algorithms.

#### New column

* **Potential Gap:** This is the difference between Potential and Overall. 

##### (This will be used to predict market value and will be used for footballer signing decisions in the next sections.)


---

#     2.  Exploratory Analysis     

Intro blah blah blah

graphs



```mean(FIFA19$Contract.Valid.Until, na.rm =TRUE)
mean(FIFA19$Release.Clause, na.rm =TRUE)
mean(FIFA19$Value, na.rm =TRUE)
```


```range(FIFA19$Overall)
quantile(FIFA19$Overall)
```


```FIFA19 %>%
  ggplot(aes(x= Overall)) +
  geom_histogram(color = "white", fill = "darkgrey") +
  ggtitle("Player ratings Are Normally Distributed", subtitle = "The mean can be used as a measure of central tendancy")
```

### Overall
Player ratings are normally distributed

![Screen Shot 2019-11-05 at 16 06 50](https://user-images.githubusercontent.com/44293686/68247316-95aab900-ffe8-11e9-9c54-2879fb71132e.png)

**Summary statistics**

 | Min. | 1st Qu. | Median |   Mean | 3rd Qu. |   Max.| 
 |:----:|:-------:|:------:|:------:|:-------:|:-----:|
 |46.00 |  62.00  | 66.00  | 66.24  | 71.00   |  94.00|


```FIFA19 %>%
  ggplot(aes(x= Age)) +
  geom_histogram(color = "white", fill = "darkgrey") +
  ggtitle("Player ages is not normally distributed", subtitle = "Left side distribution")
```
### Ages
Player ages **are not** normally distributed.

![Screen Shot 2019-11-05 at 16 16 00](https://user-images.githubusercontent.com/44293686/68246873-b4f51680-ffe7-11e9-91bd-b2e9e47953a0.png)

**Summary statistics**

| Min. | 1st Qu. | Median |   Mean | 3rd Qu. |   Max.| 
|:----:|:-------:|:------:|:------:|:-------:|:-----:|
|16.00 |  21.00  | 25.00  | 25.12  | 28.00   |  45.00|


```FIFA19 %>%
  group_by(Age) %>%
  summarise(Rating = mean(Overall)) %>%
  ggplot(aes(x= Age, y= Rating, group = 1)) +
  geom_line(color = "grey50", size = 1) +
  ggtitle("The Age Curve Flattens Off", subtitle = "Player ratings tend not to get better after the age of 30")
```
### Rating vs Age
Players reach their peak level at low thirties and then decline.


![Screen Shot 2019-11-05 at 16 24 58](https://user-images.githubusercontent.com/44293686/68247460-e5898000-ffe8-11e9-8697-7a1a761c7108.png)

Plotting the youngest teams


```age_avg <- mean(FIFA19$Age)
age_sd <- sd(FIFA19$Age)
```

```team_age <- FIFA19 %>%
  group_by(Club) %>%
  summarise(AvgAge = mean(Age)) %>%
  mutate(AgeZ_score = (AvgAge - age_avg) / age_sd)
```

```team_age <- team_age %>%
  mutate(AgeType = ifelse(AgeZ_score <0, "Below", "Above"))
```

```team_age <- team_age %>%
  arrange(desc(AgeZ_score)) %>%
  head(20) %>%
  rbind(team_age %>% arrange(desc(AgeZ_score)) %>% tail(20))
```

```team_age %>%
  ggplot(aes(x= reorder(Club,AgeZ_score), y= AgeZ_score)) +
  geom_bar(stat = 'identity', aes(fill = AgeType), colour = "white") +
  geom_text(aes(label = round(AvgAge,1))) +
  scale_fill_manual(values = c("purple", "green")) +
  coord_flip() +
  ggtitle("Nordic Clubs Are Younger Than South American Clubs", subtitle = "Ranking the 20 oldest playing lists vs the 20 youngest playing lists") +
  
  theme(legend.position = "none", axis.text.x = element_blank())
  ```
 ### Oldest vs Youngest Teams
 Nordic Clubs Are Younger Than South American Clubs

![Screen Shot 2019-11-05 at 16 27 30](https://user-images.githubusercontent.com/44293686/68247647-4749ea00-ffe9-11e9-999b-712af86bdfc8.png)


 **Oldest Clubs**
 
| Club        | Average Age  | Country   |         
|:-----------:|:------------:| :--------:| 
|Paraná       | 31.6         | Brazil    |
|Cruzeiro     | 30.6         | Brazil    |
|Chapecoense  | 30.4         | Brazil    |
|Botafogo     | 30.4         | Brazil    |
|Paranaense   | 30.4         | Brazil    |

**Youngest Clubs**


| Club                 | Average Age  | Country      |         
|:--------------------:|:------------:| :-----------:| 
|FC Admira Wacker      | 21.9         |    Austria   |
|Sochaux-Montebéliard  | 21.7         |   France     |
|Bohemian FC           | 21.5         |  Ireland     |
|FC Groningen          | 21.4         |  Netherlands |
|FC Nordsjaelland      | 20.3         |  New Zealand |


* Brazilian players tend to retire in Brazil after long careers in Europe, that fact is key to understand the extreme above average age for some of the clubs. 

* Young average age teams tend not to have a large budget to spend on mature players and as soon as their players gain experience they are sold to bigger clubs. 


The average overall rating of the 20 highest rated teams in descending order.

``` top_20_overall_clubs <- FIFA19 %>%
  group_by(Club) %>%
  summarise(AverageRating = mean(Overall, na.rm = T)) %>%
  arrange(desc(AverageRating)) %>%
  head(n = 20) %>% pull(Club) 
  ```


```FIFA19 %>%
  filter(Club %in% top_20_overall_clubs) %>%
  mutate(Top3 = ifelse(Club %in% c("Juventus", "Napoli", "Inter"), "Yes", "No")) %>%
  ggplot(aes(x= reorder(Club,Overall), y= Overall, fill = Top3)) +
  geom_boxplot(color = "black") +
  scale_fill_manual(values = c("lightgrey", "purple")) +
  ggtitle("Italian Teams Have The Highest Overall Ratings", subtitle = "The average overall rating of the 20 highest rated teams in the game, sorted in descending order") +
  coord_flip() +
  theme(legend.position = "none")
  ```
  ### Italian teams (purple) have the highest overall ratings
  
  ![Screen Shot 2019-11-05 at 16 29 31](https://user-images.githubusercontent.com/44293686/68247871-b3c4e900-ffe9-11e9-8c1a-4382f34fe51e.png)
  
 **Which teams have the most superstars?**  

```FIFA19 %>%
  mutate(ElitePlayers = ifelse(Overall >= 85, "Elite", "Not Elite")) %>%
  group_by(Club, ElitePlayers) %>%
  filter(ElitePlayers == "Elite") %>%
  summarise(NumberElitePlayers = n()) %>%
  filter(NumberElitePlayers >1) %>%
  mutate(Top3 = ifelse(Club %in% c("Juventus", "Napoli", "Inter"), "Yes", "No")) %>%
  arrange(desc(NumberElitePlayers)) %>%
  ggplot(aes(x= reorder(Club,NumberElitePlayers), y= NumberElitePlayers, fill = Top3)) +
  geom_col(color = "black") +
  scale_fill_manual(values = c("lightgrey", "purple")) +
  ggtitle("However If You Define Talent As Number Of Superstars", subtitle = "Plotted are clubs with more than one 'elite' player. Elite players being those with a rating greater than 85") +
  scale_y_continuous(breaks = seq(0,12,1))+
  coord_flip() +
  theme(legend.position = "none")
  ```
  Let's define that top player has to have an 85 or more score Overall. Just like the players below...
  
  
  ![Screen Shot 2019-11-07 at 17 51 09](https://user-images.githubusercontent.com/44293686/68434586-5797cb80-0187-11ea-968f-455b8815b25d.png)
  
  ### Explain blah blahh
  
![Screen Shot 2019-11-05 at 16 32 22](https://user-images.githubusercontent.com/44293686/68248023-04d4dd00-ffea-11e9-9cec-7ef7c9c7b351.png)


---

#     3.  10 facts about the FIFA 19 Dataset    



#### 1. Left footed players

``` table(FIFA19$Preferred.Foot)
left <- 4211/13948
left
right <- 1-left
right
```
###### Answer: 
30 percent

___

#### 2. Country with most players in the game

```most_nationalities <- summary(FIFA19$Nationality)
head(most_nationalities, 10)
```

###### Answer: 
1. England (1662)
2. Germany (1198)
3. Spain (1072)
4. Argentina(937)
5. France(914).


![Screen Shot 2019-11-06 at 15 54 13](https://user-images.githubusercontent.com/44293686/68337644-f13f7a00-00ae-11ea-8636-9ff4d172207a.png)
___

#### 3. Nationality with most players in the top 100

```top_100 <- FIFA19[0:100,]
nationalities_top_100 <- data.frame(table(top_100$Nationality))
top_100 <-(nationalities_top_100 %>% arrange(desc(Freq)))
head(top_100,10)
```
###### Answer: 
Spain (14)

![Screen Shot 2019-11-06 at 15 56 10](https://user-images.githubusercontent.com/44293686/68337615-e1c03100-00ae-11ea-9eed-cd3c0752e635.png)

___

#### 4. Oldest player

```oldest_player <- FIFA19 %>%
  arrange(desc(Age))

head(oldest_player[,2:5],1)
```
###### Answer:
Mexican goalkeeper Oscar Perez (45) 


![Screen Shot 2019-11-06 at 15 54 40](https://user-images.githubusercontent.com/44293686/68337636-ee448980-00ae-11ea-8aa4-134bab085720.png)

___

#### 5. Best 10 under 21 players in the game

```young_beasts<-subset(FIFA19, FIFA19$Age < 21 & FIFA19$Overall>75)
head(young_beasts[,2:5],10)
```
###### Answer:
1. K. Mbappé  (88)
2. M. de Ligt  (82)
3. G. Donnarumma  (82)
4. M. Rashford (81) 
5. L. Bailey (81)


![Screen Shot 2019-11-06 at 15 55 42](https://user-images.githubusercontent.com/44293686/68337629-e8e73f00-00ae-11ea-871a-900015a56fed.png)

___

#### 6. England's best rated player

```eng<-subset(FIFA19, FIFA19$Nationality == "England" & FIFA19$Overall>80)
head(eng[,2:5],5)
```
###### Answer:
Harry Kane (89)

![Screen Shot 2019-11-06 at 15 57 08](https://user-images.githubusercontent.com/44293686/68337610-de2caa00-00ae-11ea-8b30-52d9f44d8e11.png)

___

#### 7. Average player age in the game

```mean(FIFA19$Age)```
###### Answer:
25.12 years

___

#### 8. Total teams

```length(unique(FIFA19$Club))```
###### Answer:
652

___

#### 9. Best 3 Players

```head(FIFA19[,0:5],3)```
###### Answer:
1. Lionel Messi (94), Cristiano Ronaldo (94)
2. Neymar (92)

![Screen Shot 2019-11-06 at 15 58 09](https://user-images.githubusercontent.com/44293686/68337592-d240e800-00ae-11ea-88f1-3908eb4c54cd.png)

___

#### 10. 5 Players with most expectations (Overall- Potential)

```FIFA19$Potential.Gap <-  FIFA19$Potential - FIFA19$Overall

most_expectations <- FIFA19 %>%
  arrange(desc(Potential.Gap))

head(most_expectations[,2:8], 5)
```
###### Answer:
1. J. von Moos
2. D. Campbell
3. Y. Lenze
4. B. Mumba
5. K. Askildsen

---

#     4. Manchester United Case Study    


![Screen Shot 2019-11-05 at 15 59 17](https://user-images.githubusercontent.com/44293686/68248885-b1638e80-ffeb-11e9-91d8-ec3442b4628b.png)



### Context

Let’s pretend we are part of the Manchester United recruitment team. For next season (2019-20) the team wants to sign players in order to compete for the Premier League title. There are four players that expire contract this season and need to be replaced:

* Alexis Sánchez, right winger
* Matteo Darmian, left back
* Juan Mata, right midfielder
* Ander Herrera, center midfielder

Suppose the club wants to find one replacement for each of those four player, plus one central defender to make their defense stronger since last season they lacked of top defenders. 
The last seven years have not being good in terms of performance for the team. Since Sir Alex Ferguson, former manager leaved Manchester United, the club has not made the best managing decisions. Now with a new manager called Ole Gunnar Solskjaer the club wants to return to the previous signing strategy which contemplates the following points:
Sign players that are not at their peak level but they have potential to be elite players (like they did with Cristiano, Beckham or Diego Forlan).

Young players than can be part of the team for at least 4 years in a great level (less than 27 years old).
Avoid spending enormous amounts of money for one player, that may be too risky (there are many examples in the post Ferguson era like Angel di María or Radamel Falcao).
Players that are talented today and don’t need more than one season to give great performances.

###  Approach

We will make a subset of players from the whole dataset that have the following characteristics:

|Function| Interpretation |
|:------:|:---------------:|
| FIFA19$Age < 27|  Players with less than 27 years old|
| FIFA19$Overall>80|  Player with high levels of talent|
| FIFA19$Potential.Gap > 0| Players that are expected to improve in the following years|
| FIFA19$Value < 80|  The maximum price they will pay is $ 80 million USD to contract them|
| FIFA19$Release.Clause < 80|  The club wont sign footballers with high release clauses, the threshold selected is $ 80 million USD|
	
* **There is a pool of 123 players** that have all the conditions above. 
* Now that we have that pool, we **find players that have the same position as the players we want to replace**. 
* **The players with the highest “Overall”** score and that have the same positions as the players leaving the club (plus a centre back) are the ones that will be signed. 

The code lines are below:

Dimension of players that fulfill conditions:

```posible_man_united <- subset(FIFA19, FIFA19$Age < 27 & FIFA19$Overall>80 & FIFA19$Potential.Gap > 0 & FIFA19$Value < 80 & FIFA19$Release.Clause < 80)
dim(posible_man_united)
```
###### Result: 123

Alexis Sánchez's replacement

```sanchez_replacement <- subset(posible_man_united, Position =="RW")
dim(sanchez_replacement)
head(sanchez_replacement[,2:8],5)
new_sanchez <- subset(head(sanchez_replacement[,2:8],1))
```
###### Player:  

Juan Mata's replacement 

```mata_replacement <- subset(posible_man_united, Position =="RM")
dim(mata_replacement)
head(mata_replacement[,2:8],5)
new_mata <-subset(head(mata_replacement[,2:8],1))
```
###### Player:

Ander Herrera's replacement

```herrera_replacement <- subset(posible_man_united, Position =="CM")
dim(herrera_replacement)
head(herrera_replacement[,2:8],5)
new_herrera <- subset(head(herrera_replacement[,2:8],1))
```
###### Player:


New centreback

```new_centerback <- subset(posible_man_united, Position =="CB")
dim(new_centerback)
head(new_centerback[,2:8],5)
new_1_centerback <- subset(head(new_centerback[,2:8],1))
```
###### Player:

Matteo Darmian's replacement

```darmian_replacement <- subset(posible_man_united, Position =="LB")
dim(darmian_replacement)
head(darmian_replacement[,2:8],5)
new_darmian <- subset(head(darmian_replacement[,2:8],1))
```
###### Player:

Getting a table of Man U's new signings.

```man_u_new_signings <- rbind(new_sanchez, new_mata, new_herrera, new_1_centerback, new_darmian)
man_u_new_signings
```
### Manchester United's new signings (2018-2019) 

| Rank   | Name      | Age  | Nationality   | Overall       | Potential  | Club             | Value  |
| :----: |:---------:| :---:| :-----------: |:-------------:| :---------:| :---------------:| :-----:|
| 168    | T. Werner | 22   | Germany       | 83            | 87         |RB Leipzig        | 34.5 M |
| 123    | F. Thauvin|   25 | France        | 84            | 87         |O Marseille       | 39.0 M |
| 122    | Jorginho  |    26| Italy         | 84            | 87         |Chelsea           | 38.0 M |
| 116    |  N. Süle  |   22 | Germany       | 84            | 90         |FC Bayern München | 36.5 M |
| 86     | D. Alaba  |    26| Austria       | 85            | 87         |FC Bayern München | 36.5 M |


```total_expenditure <- sum(man_u_new_signings$Value)
total_expenditure
```
|          Total  |
|          :----: |
|186 million euros|

---

#     5. Predicting Market Value  

 
### Context

Suppose Manchester United wanted to make a model to predict the market value of footballers in order to assign a market price to their own players and want to know the market value of players outside the club so that they make the best decisions in the next in the market windows.

### Methodology
**Target variable:**  market value (in millions of dollars)

**Type:** regression

**Evaluation metric:** RMSE

**Models deployed:** 

* Linear Regression (4 models)
* Stochastic Gradient Boosting (2 models)
* eXtreme Gradient Boosting  (4 models)
* Random Forest (2 models)
* Support Vector Machines (1 model)
* **TOTAL MODELS**: 13

### Data Pre Processing

The following steps were made to the original dataset after cleaning the data. It is important to note that all models deployed only take numerical data.

Data preprocessing steps

I.	Dummification of the following variables:
Position
Prefered.Foot
Work.Rate      
II. 	Filtering to use only numerical data
I decided to do this in order to speed the modelling process, this lets me experiment more with tuning parameters and feature selection.
II. 	Drop near zero variance columns
III. 	Data center and scale data
IV.  Data imputation (mean imputation) 
V. 	Reduce skewness (Box-Cox Transformation)



#### Data Pre Processing

* guide https://machinelearningmastery.com/pre-process-your-dataset-in-r/


---
There will be two datasets:



**1. Non PCA dataset(model_data.csv)** →  A dataset without highly correlated values (the threshold was 0.7 of correlation coefficient).  The dimension is of 59 columns and 18207 observations.

**2. PCA dataset	(pca_data.csv)**  → The dimension is of 47 columns and 18207 observations.


---
#### Preprocessing Dataset 1 (Non PCA)

```dim(FIFA19)```

##### I. Near Zero Varince

```FIFA19 <- FIFA19[, -nearZeroVar(FIFA19)]  ## removed only one predictor

dim(FIFA19)
```
##### II. dummifing position, preferred foot and work rate

```FIFA19_1 <- cbind(FIFA19, dummy(FIFA19$Position, sep = "_"))
FIFA19_1 <- cbind(FIFA19_1, dummy(FIFA19$Preferred.Foot, sep = "_"))
FIFA19_1 <- cbind(FIFA19_1, dummy(FIFA19$Work.Rate, sep = "_"))
```

```names(FIFA19_1)
View(FIFA19_1)
```

```numeric_FIFA19 <- FIFA19_1[sapply(FIFA19_1,is.numeric)]


names(numeric_FIFA19)
dim(numeric_FIFA19)
```

##### III. center and scale

```preProc <- preProcess(numeric_FIFA19, method=c('center','scale')) 
FIFA19_transformed <- predict(preProc,numeric_FIFA19)
```
##### IV. normalization
* since it is centeres and scaled, we will its null values with "0"  (no need to do complex imputations since they are so few)

```FIFA19_transformed <- replace(FIFA19_transformed, is.na(FIFA19_transformed), 0)

names(FIFA19_transformed)
View(FIFA19_transformed)
summary(FIFA19_transformed)
```

##### V. Skewness treatment

```preprocessParams <- preProcess(FIFA19_transformed, method=c("BoxCox"))```
##### summarize transform parameters
```print(preprocessParams)```
##### transform the dataset using the parameters
```FIFA19_transformed_1 <- predict(preprocessParams, FIFA19_transformed)```
##### summarize the transformed dataset (note pedigree and age)
```summary(FIFA19_transformed_1)```

##### DATASET 1 -> NON CORRELATED VALUES

##### VI. remove high corr

```df2 = cor(FIFA19_transformed_1)
hc = findCorrelation(df2, cutoff=0.75) # putt any value as a "cutoff" 
hc = sort(hc)
model_data = FIFA19_transformed_1[,-c(hc)]

dim(model_data)
dim(FIFA19_transformed_1)
```
---

##### Preprocessing Dataset 2 (PCA) 
*Note:(PCA does not discard correlated values) 

```pca_process = preProcess(FIFA19_transformed_1, method=c("pca"))
print(pca_process)
pca_data <- predict(pca_process, FIFA19_transformed_1)
dim(pca_data)
```


```#write.csv(model_data, "model_data.csv")
#write.csv(pca_data, "pca_data.csv")
```


##### Remove Value + insert as non-scaled

```Value <- FIFA19$Value
Value <- replace(Value, is.na(Value), 0)
model_data_1 <- subset(model_data, select = -c(Value) )
model_data_1 <- cbind(Value, model_data_1)

pca_data_1 <- cbind(Value, pca_data)

View(model_data_1)
View(pca_data_1)
```

##### Predictor importance

```set.seed(7)


control <- trainControl(method="repeatedcv", number=10, repeats=3)
model_imp <- train(Value~., data=model_data_1, method="lm", trControl=control)
importance <- varImp(model_imp, scale=FALSE)
print(importance)
plot(importance)


dim(model_data)

model1 <- lm(Value ~., data = FIFA19_transformed)
model1_results <- summary(model1)
summary(model1)
```

--- 

## Modelling

### Training Control
10 fold cross validation was applied to all models.
Below is the formula used:

```tc <- trainControl(method = "cv", number = 10)```

### Linear Regression Models


##### (1st model) Linear regression
*contains all features

```tc <- trainControl(method = "cv", number = 10)
lm1_cv <- train(Value~., data = model_data_1, method = "lm",
                trControl = tc)
summary(lm1_cv)
results_lm1_cv <- summary(lm1_cv)
```
##### (2nd model) Linear regression
*contains 20 relevant features (less than 0.001 p - value)


```lm2_cv <- train(Value~Age+Potential+Wage+International.Reputation+Skill.Moves+Jersey.Number+Contract.Valid.Until+SprintSpeed+SprintSpeed+Reactions+Stamina+Strength+Composure+FIFA19_CAM+FIFA19_CB+FIFA19_CDM+FIFA19_CF+FIFA19_CM+FIFA19_LB+FIFA19_LCB, data = model_data_1, method = "lm",
                trControl = tc)

results_lm2_cv <- summary(lm2_cv)
```

##### (3rd model) Linear regression
*contains only 4 features that are really intuitive

```lm3_cv <- train(Value~Age+Potential+Wage+International.Reputation+Stamina, data = model_data_1, method = "lm",
                trControl = tc)

summary(lm3_cv)

results_lm3_cv <- summary(lm3_cv)
```

###### Best MSE is 3.533


##### (4th model) Linear regression
*contains all features with PCA dataset

```lm1_cv_PCA <- train(Value~., data = pca_data_1, method = "lm",
                trControl = tc)


summary(lm1_cv_PCA)

results_csv_PCA <- summary(lm1_cv_PCA)
```

###### RMSE 0.959 which is great!

---

##### (1st) Stochastic Gradient Boosting 
*with PCA dataset and all features


```set.seed(7)

gbmFit1_pca <- train(Value ~ ., data = pca_data_1, 
                 method = "gbm", 
                 trControl = tc, 
                 verbose = FALSE, 
                 ## Only a single model can be passed to the
                 ## function when no resampling is used:
                 tuneGrid = data.frame(interaction.depth = 4,
                                       n.trees = 100,
                                       shrinkage = c(.1,.2,.3),
                                       n.minobsinnode = 20),
                 metric = "RMSE")


results_gbmFit1_pca <- gbmFit1_pca

results_gbmFit1_pca
```
###### RMSE of 1.11


##### (2nd) Stochastic Gradient Boosting 
*increasing number of trees from 100 to 300

```set.seed(7)

gbmFit2_pca <- train(Value ~ ., data = pca_data_1, 
                     method = "gbm", 
                     trControl = tc, 
                     verbose = FALSE, 
                     ## Only a single model can be passed to the
                     ## function when no resampling is used:
                     tuneGrid = data.frame(interaction.depth = 4,
                                           n.trees = 300,
                                           shrinkage = .2,
                                           n.minobsinnode = 20),
                     metric = "RMSE")


results_gbmFit2_pca <- gbmFit2_pca

results_gbmFit2_pca
```
###### RMSE 0.98935

---

##### (1st model) eXtreme Gradient Boosting with PCA


```set.seed(7)


xgbGrid_1 <- expand.grid(nrounds = c(1, 10, 15),
                             max_depth =  4,
                             eta = .1,
                             gamma = 0,
                             colsample_bytree = .7,
                             min_child_weight = 1,
                             subsample = c(.5,.8,1))

xgbFit1_pca <- train(Value ~ ., data = pca_data_1, 
                     method = "xgbTree", 
                     trControl = tc, 
                     verbose = FALSE, 
                     ## Only a single model can be passed to the
                     ## function when no resampling is used:
                     tuneGrid = xgbGrid_1, 
                     metric = "RMSE")


results_xgbFit1_pca <- xgbFit1_pca
```



```set.seed(7)


xgbGrid_2 <- expand.grid(nrounds = c(100, 150, 200),
                         max_depth =  4,
                         eta = .1,
                         gamma = 0,
                         colsample_bytree = .7,
                         min_child_weight = 1,
                         subsample = c(.5,.8,1))


xgbFit2_pca <- train(Value ~ ., data = pca_data_1, 
                     method = "xgbTree", 
                     trControl = tc, 
                     verbose = FALSE, 
                     ## Only a single model can be passed to the
                     ## function when no resampling is used:
                     tuneGrid = xgbGrid_2, 
                     metric = "RMSE")


xgbFit2_pca_results <- xgbFit2_pca
xgbFit2_pca_results
```

##### third model


```set.seed(7)


xgbGrid_3 <- expand.grid(nrounds = c(250,400),
                         max_depth =  4,
                         eta = .1,
                         gamma = 0,
                         colsample_bytree = .7,
                         min_child_weight = 1,
                         subsample = c(.5,.8,1))


xgbFit3_pca <- train(Value ~ ., data = pca_data_1, 
                     method = "xgbTree", 
                     trControl = tc, 
                     verbose = FALSE, 
                     ## Only a single model can be passed to the
                     ## function when no resampling is used:
                     tuneGrid = xgbGrid_3, 
                     metric = "RMSE")

xgbFit3_pca_results <- xgbFit3_pca
xgbFit3_pca_results
```
##### fourth model

```set.seed(7)

xgbGrid_4 <- expand.grid(nrounds = 1000,
                         max_depth =  4,
                         eta = .1,
                         gamma = 0,
                         colsample_bytree = .7,
                         min_child_weight = 1,
                         subsample = c(.5,.8,1))


xgbFit4_pca <- train(Value ~ ., data = pca_data_1, 
                     method = "xgbTree", 
                     trControl = tc, 
                     verbose = FALSE, 
                     ## Only a single model can be passed to the
                     ## function when no resampling is used:
                     tuneGrid = xgbGrid_4, 
                     metric = "RMSE")


xgbFit4_pca_results <- xgbFit4_pca
xgbFit4_pca_results
```

---

##### Random   Forest   with   PCA

```set.seed(7)


rfFit_pca1 <- train(Value ~ ., 
                data = pca_data_1,
                method = 'ranger',
                # should be set high at least p/3
                tuneLength = 10, 
                trControl = tc,
                ## parameters passed onto the ranger function
                # the bigger the better.
                num.trees = 15,
                importance = "permutation")


rfFit_pca1_results <- rfFit_pca1

rfFit_pca1_results
```

##### (second model) Random   Forest   with   PCA


```set.seed(7)


rfFit_pca2 <- train(Value ~ ., 
                    data = pca_data_1,
                    method = 'ranger',
                    # should be set high at least p/3
                    tuneLength = 10, 
                    trControl = tc,
                    ## parameters passed onto the ranger function
                    # the bigger the better.
                    num.trees = 150,
                    importance = "permutation")


rfFit_pca2_results <- rfFit_pca2
rfFit_pca2_results
```

##### Support   Vector    Machines    with    PCA 

```svm_pca1 <- train(Value~., data=pca_data_1, method = "svmLinear", trControl = tc)

svm_pca1_results <- svm_pca1

svm_pca1_results
```
###### RMSE 1.01

---

##  Model Results


###### Linear   Regression

```results_lm1_cv
results_lm2_cv
lm3_cv
results_csv_PCA
```
###### GBM

```results_gbmFit1_pca
results_gbmFit2_pca
```
###### XGB

```results_xgbFit1_pca
xgbFit2_pca_results 
xgbFit3_pca_results
xgbFit4_pca_results
```
###### Random Forest 

```rfFit_pca1_results
rfFit_pca2_results
```
###### SVM 

```svm_pca1_results```

## Best Model

```xgbFit4_pca_results```


|          Type      |       eXtreme Gradient Boosting  |    
|:-------------:     |:-------------------------------: | 
| nrounds            |      1000                        | 
|max_depth           |      4                           | 
|eta                 |      0.1                         | 
| gamma              |      0                           | 
| colsample_bytree   |      0.7                         | 
| min_child_weight   |      1                           | 
| subsample          |      1                           |



|          RMSE  |       RSquared  |       MAE  |
|:-------------: |:--------------: | :--------: |
|0.7217564       |      0.9835732  | 0.3410680  |


### Interpretation

The best **RMSE is 0.7217**. This means that the model has an **average error of $ 721,700 USD** when estimating market value. This is a good model since some players can be worth up to $ 100 million USD, while the average player is worth $ 5 million USD.


