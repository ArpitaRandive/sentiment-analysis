# Sentiment Analysis

Sentiment analysis is the process of analyzing digital text to understand the emotional tone of a message. It helps an analyst determine whether the message has a positive, negative, or neutral tone and to understand text much like humans. 
This study compares the performance of five models—logistic regression, random forest, support vector machine (SVM), recurrent neural network (RNN), and BERT — on classifying sentiment from labeled feedback datasets.

# Data preparation

The first step is to import the dataset using the import function **[import... as...]** and also import the **Pandas library** to read and manipulate data.
After this, the following actions are performed on the dataset to clean the data:

 - Concatenating the provided datasets using pd.concat().
 - Removing all the NaN values.
 - Tokenization using Tensorflow and defining features like num words, etc.
 - Sequence Padding (ensuring consistency in lengths).
 

## Data Splitting

After cleaning the dataset, it can be split into training and testing sets: the training set is used to train the model, and the test set is used to evaluate its performance.
For this dataset, we have decided to split the dataset in an **8:2 ratio**, 8 being the training set and 2 being the test set.

# Model Building
Now, we initiate the development and training of machine learning models tailored to achieve our desired outcomes, ensuring efficiency and performance throughout the pipeline
## Logistic Regression
Now, a logistic regression model is designed and trained using TF-IDF vectorized features. The model was configured with class weight=’balanced’ to account for any potential class imbalance in the dataset.
The performance metrics are summarized below:
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.98      | 0.97   | 0.98     | 181     |
| 1             | 0.98      | 0.99   | 0.98     | 243     |
| **Accuracy**  |           |        | **0.98** | **424** |
| **Macro avg** | 0.98      | 0.98   | 0.98     | 424     |
| **Weighted avg** | 0.98   | 0.98   | 0.98     | 424     |

The model gives an outstanding performance with an accuracy of 98%.
Both classes achieved equally high precision, recall, and F1-scores, indicating that the model is rarely making any mistakes, whether the class is positive or negative, and it is also balanced, which indicates it is not biased towards one class.
Overall, we can conclude that the model is reliable and consistent.

To assess the model’s performance, a confusion matrix and ROC curve were generated for validation. The results of the confusion matrix depicts that out of 424 total samples, only 8 has been misclassified, indicating a balanced performance across classes, whereas the ROC curve shows the overall accuracy of the model.

## Random Forest 
Random Forest creates an ensemble of multiple decision trees to reach a singular, more accurate prediction or result. This model was also trained on TF-IDF-transformed data. Following is the performance metrics of the model:

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.99      | 0.91   | 0.95     | 181     |
| 1             | 0.94      | 1.00   | 0.97     | 243     |
| **Accuracy**  |           |        | **0.96** | **424** |
| **Macro avg** | 0.97      | 0.95   | 0.96     | 424     |
| **Weighted avg** | 0.96   | 0.96   | 0.96     | 424     |

The Random Forest model shows an overall accuracy of 96%.
It shows high precision, recall, F1-scores and accuracy, indicating that it performed well across both classes but sacrifices some interpretability for better handling of nonlinear patterns (compared to logistic regression).

The confusion matrix and ROC curve confirm the performance of the model, with only 17 samples misclassified.

## Support Vector Machine (SVM)
Support Vector Machine (SVM) is a robust supervised machine learning algorithm commonly used for classification tasks. The model is trained on TF-IDF-transformed data, making it suitable for high-dimensional text classification problems such as sentiment analysis.
Following are the performance metrics of the model:
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.98      | 0.99   | 0.99     | 181     |
| 1             | 0.99      | 0.99   | 0.99     | 243     |
| **Accuracy**  |           |        | **0.99** | **424** |
| **Macro avg** | 0.99      | 0.99   | 0.99     | 424     |
| **Weighted avg** | 0.99   | 0.99   | 0.99     | 424     |

The SVM model achieved an impressive accuracy of 99% , outperforming most of the other models. Both precision and recall are nearly perfect, especially for class 1 (positive sentiment), indicating that the model was highly effective in correctly classifying both sentiment classes.
The confusion matrix and ROC curve confirm this strong performance.


## Recurring Neural Network - RNN(LSTM)

RNNs are designed to handle sequential data by taking the output from previous steps and using it as input for the current step.
The RNN model was built using embedding layers instead of TF-IDF, enabling it to capture semantic and contextual word relationships more effectively.
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.94      | 0.72   | 0.81     | 181     |
| 1             | 0.84      | 0.97   | 0.90     | 243     |
| **Accuracy**  |           |        | **0.87** | **424** |
| **Macro avg** | 0.89      | 0.84   | 0.86     | 424     |
| **Weighted avg** | 0.88   | 0.87   | 0.86     | 424     |

The model shows a decent accuracy of 87% which, even though is lower as compared to the other models, still performs well in capturing the temporal dependencies. It especially performed well for class 1, with higher recall and F1 scores, indicating its ability to correctly identify positive sentiments.
The confusion matrix and ROC curve reveal areas where the RNN struggled slightly more with distinguishing between negative and positive sentiments, particularly for class 0. 

Overall, the model showcases the potential of deep learning approaches in text analysis tasks


# BERT (Bidirectional Encoder Representations from Transformers)

BERT (Bidirectional Encoder Representations from Transformers) is a deep learning model developed by Google that has revolutionized natural language processing tasks, reading text bidirectionally, which allows it to better understand the context of a word based on both its left and right surroundings.
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.99      | 0.99   | 0.99     | 181     |
| 1             | 0.99      | 1.00   | 0.99     | 243     |
| **Accuracy**  |           |        | **0.99** | **424** |
| **Macro avg** | 0.99      | 0.99   | 0.99     | 424     |
| **Weighted avg** | 0.99   | 0.99   | 0.99     | 424     |

The BERT model achieved an outstanding accuracy of 99%, the highest among all models evaluated. It performed exceptionally well across both classes, with almost perfect precision, recall, and F1 scores.
The results indicate that BERT rarely misclassifies reviews, regardless of their sentiment.
The confusion matrix and ROC curve confirm this performance, with only a few misclassified instances and an AUC close to 1.0.
![BERT Confusion Matrix](https://github.com/ArpitaRandive/sentiment-analysis/blob/main/assets/BERT%20Confusion%20Matrix.png)
The reason why BERT outperforms the other models is due to the following reasons:
 - As suggested by its name, BERT reads text from both directions, enabling better contextual understanding.
 - It’s pre-trained on large text corpora, giving it strong language understanding before fine-tuning.
 - Its rich embeddings help it capture language subtleties missed by other models.

## Conclusion

Overall, all the models achieved high accuracy, and BERT outperformed as it had better contextual understanding. While more straightforward models, such as logistic regression and SVM, provided good baseline performances, sophisticated models such as RNN and BERT provided better performances as they were capable of analyzing sequential and contextual patterns in the data.

