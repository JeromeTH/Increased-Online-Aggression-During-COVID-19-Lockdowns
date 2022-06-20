# NLP_Covid_19_Twitter_Aggression_Study

## Basic Description

This project uses neural network based BERT model to analyze the trends of aggressive emotions on Twitter between 2019/1/1 and 2020/10/1. For this, we picked 3 common types of online aggression as our subject: Anger, Offensive Language(Offensive), and Hate Speech (Hate). During the 92 weeks in this period, there is data for the count and proportion of aggressive tweets for each of the 50 US states. Using this data, we applied the Difference in Differences (DID) analysis to estimate the causal affect between lockdown and increasing aggression. 

This research consists of 4 parts
1. Sample Data from Twitter 
2. Training the BERT model
3. Use the trained BERT model to analyze aggressive emotions 
4. Use STATA to perform DID causal analysis 

## Twitter Data Sampling

This folder includes all Twitter data sampled in this study. Based on Twitter's privacy policy, we only provide the Tweet ID for each tweet. One can fetch the text and other information via a Twitter developer account.  

The "Tweets by states" subfolder contains Tweet samples from each state, along with the emotion analysis by our trained BERT model. 

The "Human_Annotated" File is used to evaluate our mode's performance by comparing predictions with 2 native English annotators. 

## Training the BERT model

Source code for this part is located in the "Model Training" folder

The "Training data" subfolder includes: labeled training data for each of the three aggressive emotions. The training data has a "Split" label with 0 meaning training, 1 meaning validation, and 2 meaning testing. 

The "Training Code" subfolder includes: Datapreprocessing for BERT models, training procedures, and model evaluation. 

## Analyzing Trends of Aggressive Emotions

In this section, we used the trained BERT model, one for each of the three emotions, to analyze aggression in our spatiotemporal dataset of US tweets. 

The main analysis happen in these three files:

"weekly_percentages.py" instructs the BERT model to analyze tweets in each state and save results in a separate folder. In this case, we already provided the analyzed version of the data in the "Twitter Data Samplings" folder. 

"Weekly_Lockdown.py" visualizes the weekly trends of aggression between groups under and not under lockdown. 

"From_Lockdown.py" aligned data based on weeks from the start of lockdown (different in each state). The starting week of lockdown is week 0, while one week into lockdown is week 1, one week before lockdown is week -1. 

Other files are supplementary files that are imported as a python extension for the main analysis files. 

## Causal Analysis using STATA

We performed our causal analysis using the STATA BE 17.0 software, to replicate this section, one need to install the STATA software. 
Each .dta includes 4600 datapoints that corresponds to one datapoint in each 50 states for 92 weeks. 

One can follow the STATA command file to replicate the causal analysis performed in this study. 

## References
1. Dorottya Demszky, D.M.-A., Jeongwoo Ko, Alan Cowen, Gaurav Nemade, Sujith Rav, Go Emotions: A Dataset of Fine-Grained Emotions. Association of Computational Linguistics (ACL), 2020.
2. Davidson, T., et al. Automated hate speech detection and the problem of offensive language. in Proceedings of the International AAAI Conference on Web and Social Media. 2017.
3. Founta, A.M., et al. Large scale crowdsourcing and characterization of twitter abusive behavior. in Twelfth International AAAI Conference on Web and Social Media. 2018.
4. Devlin, J., et al., Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018







