# NLP_Covid_19_Twitter_Aggression_Study

## Basic Description

This project uses neural network based BERT model to analyze the trends of aggressive emotions on Twitter between 2019/1/1 and 2020/10/1. For this, we picked 3 common types of online aggression as our subject: Anger, Offensive Language(Offensive), and Hate Speech (Hate). During the 92 weeks in this period, there is data for the count and proportion of aggressive tweets for each of the 50 US states. Using this data, we applied the Difference in Differences (DID) analysis to estimate the causal affect between lockdown and increasing aggression. 

This research consists of 4 parts
1. Sample Data from Twitter 
2. Training the BERT model
3. Use the trained BERT model to analyze aggressive emotions 
4. Use STATA to perform DID causal analysis 

## Twitter Data Sampling

This folder includes all Twitter data sampled in this study. 

## Training the BERT model

Source code for this part is located in the "Model Training" folder

This includes standard datapreprocessing for BERT models, training procedures, and model evaluation. The training data has a "Split" label with 0 meaning training, 1 meaning validation, and 2 meaning testing. 

## Analyzing Trends of Aggressive Emotions

In this section, we used the trained BERT model, one for each of the three emotions, to analyze aggression in our spatiotemporal dataset of US tweets. The data itself is included in the 


