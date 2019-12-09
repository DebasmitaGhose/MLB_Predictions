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

The sport is incredibly popular within the United States, with the games drawing millions of television viewers. Analysis of the game on various media channels also provides high ratings. Every year, one of the biggest events in the MLB calendar is the Baseball Hall of Fame induction. Every year, a cohort of older players who have been retired from playing are elected into the Hall of Fame, which is a museum that honors the greatest players who have ever played in the game based on the totality of their careers. Developing a model to predict which players will be recognized as one of the greatest in the history of the game, based on data from early in their careers, is an exciting task. 


#### Technical Challenges

- The database had a significant amount of missing values, which was a hindrance in building the Machine Learning model. In order to work around that, we needed to remove features where the percentage of missing values were more than 50%. For rest of the features, we imputed the missing values with the mean value in the column. 

#### Code

The code folder in this repository contains our SQL code for data extraction/processing and our Python code for model development.  

#### Normal Form

The ER diagram located in the documents folder of this repository helps visualize that this database schema is in the correct normal form. 
