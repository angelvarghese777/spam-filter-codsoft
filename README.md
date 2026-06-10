# spam-filter-codsoft
Spam message detection using Multinomial Naive Bayes and CountVectorizer. 


## Overview
This project was developed as part of the CodSoft internship selection tasks. It focuses on building a machine learning model to classify SMS messages as either spam or ham (not spam).

Initially, I had limited understanding of machine learning concepts. After learning through tutorials (especially StatQuest videos) and experimenting with different approaches, I was able to build and refine the model.

At first, I used TF-IDF vectorization with Logistic Regression. However, the performance on unseen messages was not optimal for this task. After further learning, I switched to Multinomial Naive Bayes with CountVectorizer, which is more suitable for frequency-based text classification problems like spam detection.Basically the more the words we usually find in spam messages occur in a new message, the more likely it is to be spam message.

## Algorithm Used
- Multinomial Naive Bayes

## Techniques Used
- CountVectorizer (order of words was not important)
- Train-test split
- Text preprocessing and label encoding

## Workflow
1. Load and clean dataset
2. Convert labels into numerical form (spam = 1, ham = 0)
3. Convert text messages into numerical features using CountVectorizer
4. Train Multinomial Naive Bayes model
5. Evaluate model performance
6. Test on custom messages

## Results
- Model performs well on unseen test data
- Identifies common spam indicators such as "free", "win", "prize", and "urgent"

## Example Use Case
Input: "Congratulations! You won a free iPhone"
Output: SPAM

## Tools & Libraries
- Python
- Pandas
- Scikit-learn
