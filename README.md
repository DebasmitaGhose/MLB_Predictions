# MLB Hall of Fame Predictions: CPSC 537 Final Project
### Emmanuel Adeniran, Chris Allen, Nicholas Georgiou, Debasmita Ghose

#### Project Objective

Using data from Lahman's Baseball Database (http://www.seanlahman.com/baseball-archive/statistics), which contains MLB player statistics from 1871 to 2018, we develop a model that predicts whether a player will be inducted into the Baseball Hall of Fame. 

We store the data in our own local database and extract data using PostgresSQL. 

#### Background and Motivation

Major League Baseball (MLB) is a professional baseball league based in the United States, and has teams located in both the U.S. and Canada. The league in its current form was officially founded in 1903. The baseball season runs approximately between April and November every year, with a team eventually becoming the champion by winning a "best of seven" series at the end of the season. 

There are many intricate details that go into the rules of baseball, but to keep things simple: the objective of a baseball game is to score more runs (think points) than the opposing team. During a game, both teams have 9 players who play in the game - one player is classified as a "pitcher" and the remaining eight players are classified as "batters/fielders". The two teams facing each other alternate turns "batting" trying to score runs, analagous to offense in other sports, while the other team is both "pitching" and "fielding" to defend against the team scoring runs. As sequence of events involves the "pitcher" throwing a ball to the "batter", who tries to hit the ball with a bat into the field. The batter's ultimate objective is to hit the ball into the field and run through four "bases", which then counts as one "run".  During individual games and through the entire season, advanced statistics are recorded that help quantify how good the nine players on each team are at batting, fielding, and pitching - in other words, how good they are at scoring runs and defending points from being scored. 

A more detailed summary of the rules can be found here: https://en.wikipedia.org/wiki/Baseball_rules. In addition to this link as a starting point, there are extensive resources on the history of baseball and its rules if one is interested in reading more. 

We chose this topic as baseball is a data-oriented sport with a rich history. The sport also is very data-driven, with multiple sources having recorded comprehensive statistics going back decades. The Lahman's Baseball database is just one of many public resources for baseball data and statistics. 

The sport is incredibly popular within the United States, with the games drawing millions of television viewers. Analysis of the game on various media channels also provides high ratings. Every year, one of the biggest events in the MLB calendar is the Baseball Hall of Fame induction. Every year, a cohort of older players who have been retired from playing (for at least five years) are elected into the Hall of Fame, which is a museum that honors the greatest players who have ever played in the game based on the totality of their careers. 

Writing the SQL to extract and manipulate features and developing a model to predict which players will be recognized as one of the greatest in the history of the game is an exciting and very interesting task, and represents a great opportunity for us to help show our knowledge of database systems. In other words, this project provides us with an exciting opportunity to incorporate our knowledge of SQL and database systems and see how it fits into the big picture in a predictive modeling project. 

#### Code

The **SQL** folder in this repository contains our SQL code for data extraction/processing and to feed the website with predictions. 

Our Python code for model development is located in the **Code** folder in this repository.

#### Database Normal Form

The ER diagram located in the **Docs** folder of this repository helps visualize that this database schema is in sufficient normal form. An additional breakdown of the database schema and what the attributes mean are also located in the **Docs** folder. 

#### Model Details

In model development, our training/validation data included all players n = 14,171 from our database who made their MLB debut before 1995. For features, we utilize cumulative and average statistics of different metrics for players based on their first five years playing in the MLB. 

Within the data, some features have missing values. There are a number of reasons for why this is the case. For example, some players may play in positions where they don't record specific statistics throughout their career. In addition, it is also possible that some statistics are not recorded further back in the past and began being recorded later after the league was founded. When selecting features, we select only those features from the database where the percentage of missing values are less than 50% of the size of the training set. For the selected features, we replace the missing values with the mean value of that feature. 

We then train a Logistic Regression model to estimate the predicted probability that a player will make the hall of fame based on the statistics from their first five seasons. On our website, we utilize the predicted probabilities of the test set, which contain all players who made their MLB debut in 1995 or later - many of whom who are still active players and a few having just retired fairly recently. 

#### Model Results

Overall classification accuracy of the model was 87.6%. For the test set of players from 1995 and onward, we have the following distribution of predicted probabilities that a given player gets in the Hall of Fame:

- Minimum: 15.4%
- Q1: 20.2%
- Median: 21%
- Mean: 29.7%
- Q3: 26.4%
- Max: 99.8%

The full distribution of predicted probabilities is located in the **Docs** section of the repository. Based on the first five years of their career, most players tend to have pretty low chances of making the Hall of Fame. There is a small cluster of players, however, who have very high probabilities. At a high level, using a combination of both the predicted probabilities as well as intuition about the game of baseball, we would say that any player with a predicted probability of at least 90% is on a good trajectory, based on their first five years, to make the baseball Hall of Fame. Of course, it's definitely possible that not all of these current players will make the Hall of Fame, as some may see their performance decrease later in their career due to various factors. 

Given additional time and resources, there is definitely opportunities to improve on the model in the future if there is interest. However, the model used for predictions for this project is quite sufficient.  

#### Technical Challenges

The database had a significant amount of missing values, which was a hindrance in building the Machine Learning model. In order to work around that, we needed to remove features where the percentage of missing values were more than 50%. For rest of the features, we imputed the missing values with the mean value in the column. Given that the distributions of most of these statistics is skewed, with the best players often residing on the tail end of these distributions, imputation with a mean value is pretty safe here. 

The dataset also has a strong class imbalance towards players not reaching the hall of fame, so standard machine learning techniques fail to perform as well. That is because they do not have enough examples to learn the distribution of one of the classes. Therefore, we had to use a standard random oversampling technique to oversample positive examples from the training dataset. This does have a downside, as one could argue that the model potentially overfits in favor of players who make the Hall of Fame, and may inflate some of the predicted probabilities for players who are less likely to be elected into the baseball Hall of Fame. This is the lesser of two evils, however, as we can take into account that the absolute best players with the best chance of making the Hall of Fame will have very high probabilities, and that players with lower probabilities in the distribution will have substantially lower chances compared to those players with higher probabilities. 

In the backend, connecting JDBC and the database, and then JDBC in the webapp, also presented technical challenges. 

