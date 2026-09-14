# PDAN8412_ST10500865

# Table of Contents
EXECUTIVE SUMMARY	3
1. INTRODUCTION	3
2. DATASET EVALUATION AND SUITABILITY	3
3. DATA STRUCTURE AND EXPLORATORY DATA ANALYSIS	4
4. DATA CLEANING AND PREPARATION	6
5. LSTM MODEL DEVELOPMENT	6
6. MODEL EVALUATION	7
7. INTERPRETATION OF FINDINGS	8
8. RECOMMENDATIONS	8
9. LIMITATIONS AND ETHICAL CONSIDERATIONS	9
10. CONCLUSION	9
REFERENCES	10


# Executive Summary
The present report describes a prototype designed to determine the authorship of an English text sample. The Blog-1K corpus contains 20,166 posts by 1,000 authors, with predefined training, validation, and testing sets available. The dataset allows one to perform supervised sequence classification where the task is to detect the author based on an input written in natural language and an author identifier as a target class. The dataset was inspected and analyzed for quality using Spark, while TensorFlow was employed to train the LSTM model. The modeling has been limited to ten authors, which have the greatest number of collected posts, in order to be able to perform the experiment on a student-grade computer. 
The baseline LSTM model had a test accuracy of 7.69% and a macro F1 score of 2.50%. Further tuning of the configurations has resulted in an increase of the macro F1 metric to 4.17%, although the test accuracy was reduced to 5.13%. Therefore, it can be stated that the prototype has not yet managed to achieve reliable authorship detection.
Key outcome: The experiment successfully demonstrates the end-to-end LSTM workflow, but the model should be treated as an exploratory prototype rather than an effective deployment model.

# 1. Introduction
In this project, the issue tackled is that of automated author detection: having a written work to analyse, the model tries to figure out the author among those in its database. The model can be employed for authorship verification, lineage studies, and other text mining activities. The classification task is a multi-class supervised one, where the sequence of words is the crucial input, and the author ID is the desired output.
The reason an LSTM model was considered is that text is a sequence of characters. Instead of considering words as independent data points, the LSTM algorithm sees tokens in sequence and learns how to detect context and style across positions in the text.
The notebook uses a model that follows this architecture: Embedding > LSTM > Dropout > Dense/Softmax.

# 2. Dataset Evaluation and Suitability
The notebook uploaded shows Blog-1K’s status as a redistributable authorship-identification corpus from Haining Wang. This corpus includes a total of more than 16,000 English prose posts issued by a total of 1,000 candidate authors, and it also has predefined partitioning for training, validation and testing samples. The data set provided includes 20,166 records containing three columns; id, text and split. Therefore, this data set meets the requirement of a minimum of 10,000 records needed for the assignment and allows for a successful supervised author classification.
LSTM is seen as a perfect model to use on this data set since the text column consists of inputs that can be classified using natural-language sequential observations, while the id column serves as the author labels. As for the splits, they help ensure that the training and test observations will not get mixed incidentally in the initial experiments. Nevertheless, a full version of this problem entails training in a situation with 1,000 classes, which might be too demanding for the ordinary student’s laptop. Thus, the notebook resorts to working with the ten authors with the most training posts.
3. Data Structure and Exploratory Data Analysis
The dataset has a total of 20,166 records and 1,000 distinct authors. The data includes 16,132 training records, 2,017 validation records and 2,017 test records. Missing values were not present in either id, text or in split. There are 51 duplicate rows, based on the full row comparison.
Split	Records	Percentage
Train	16 132	80%
Validation	2 017	10%
Test	2 017	10%

 
Figure 1: Distribution of records across the predefined dataset splits.

The number of characters varied between 1,000 and 17,107 with the average being 1,863.5 and the median 1,558. The variation in the number of words is slightly smaller as it ranges between 77 and 3,192. The average word count is 339.8 while the median is 286. The discrepancy among the lengths of posts should be taken into account in terms of sequence-based processing as long documents will require either truncation or the usage of a more complex technique of document presentation.

 
Figure 2: Distribution of post character length.

The number of training posts per author differs considerably. The report states that authors typically create an average 16.13 training posts, ranging from the least of 13 to a most of 30 posts. The small number arising from the presence of only 10 authors is critical for the training of the neural network model.


 
Figure 3: Distribution of training posts per author.

# 4. Data Cleaning and Preparation
The cleaning procedure got rid of whitespaces, eliminated records with fewer than 200 characters, and discarded exact duplicates based on id, text and split. A total of 20,114 rows were left in the data frame, which means that 52 records were lost from the data frame that initially had 20,166 rows. This method was deliberately conservative in order to not distort meaningfully written content.
When conducting modelling, the top ten authors having the greatest training counts were chosen. This gave rise to a total of 332 records for modelling: 264 training records, 29 validation records, and 39 testing records. To encode author labels, Scikit-learn's LabelEncoder was used. A Keras tokenizer was used only on the training texts in order not to cause a vocabulary leak. Hence, the vocabulary was limited to 20,000 tokens, and each post was turned into a sequence and padded/truncated to 250 tokens long.
Preparation element	Configuration
Target	Author ID
Classes	10 selected authors
Vocabulary	20,000 tokens
Maximum sequence length	250 tokens
Training matrix	264 x 250
Validation matrix	29 x 250
Test matrix	39 x 250

# 5. LSTM Model Development
The baseline architecture consisted of an Embedding layer with 64 dimensions, a 64-unit LSTM layer, 30% dropout and a ten-class softmax output layer. The model used the Adam optimiser and sparse categorical cross-entropy loss.
Training was configured for up to ten epochs with batch size 64 and early stopping based on validation loss.
The baseline training curve shows training accuracy increasing to approximately 42.05% by epoch 9, while validation accuracy remained much lower and reached 20.69% at epoch 10. The gap between training and validation performance indicates weak generalisation and suggests that the small modelling sample does not provide sufficient evidence for stable learning.
  Figure 4: Baseline LSTM training and validation accuracy.

  Figure 5: Baseline LSTM training and validation loss.

# 6. Model Evaluation
Model	Accuracy	Macro Precision	Macro Recall	Macro F1
Baseline LSTM	7.69%	1.50%	7.50%	2.50%
Tuned LSTM	5.13%	5.45%	7.50%	4.17%

The basic model attained an accuracy of 7.69% in the test set containing 39 observations. Considering that there were ten categories, a balanced random classifier would have had an anticipated accuracy of about 10%. Thus, baseline accuracy does not indicate positive author detection. The macro precision was registered at 1.50%, the macro recall equaled 7.50%, while the macro F1 was equal to 2.50%. The classification report indicated that the model assigned the author class excessively to a single author while other classes had zero precision, recall, and F1 values.
The second version of the model changed the embedding dimension and the LSTM dimension to 128, raised up the dropout rate to 40%, introduced LSTM dropout and recurrent dropout, decreased the batch size to 32, and trained for up to 12 epochs. The improved model lowered the accuracy from 7.69% to 5.13%, while the macro F1 increased from 2.50% to 4.17%. This contradictory outcome indicates that changes made parameters enhanced the classification performance for some authors but made no improvements in classification accuracy in general.

#7. Interpretation of Findings
Finding 1 – Dataset suitability: The corpus is appropriate for the stated LSTM task because it contains sequential prose and author labels and exceeds the required dataset size. The predefined split structure is also useful for reproducible evaluation.
Finding 2 – Data quality: No missing values were identified in the three core fields. However, 52 rows were removed during cleaning, mainly because of the minimum text-length rule and duplicate handling. Post lengths were highly variable, making sequence truncation an important modelling consideration.
Finding 3 – Model effectiveness: The current LSTM prototype was ineffective for reliable author prediction. Both models performed close to or below the ten-class chance level on the test sample, and the baseline classification report showed severe class-level imbalance in predictions.
Finding 4 – Generalisation: Training accuracy rose substantially while validation accuracy remained low. This pattern is consistent with overfitting and/or inadequate sample size. The model had only 264 training examples for ten classes, which is a particularly restrictive setting for an LSTM.
Finding 5 – Tuning: The tuned model improved macro F1 but lowered accuracy. Further tuning alone is unlikely to solve the problem if the principal limitation is the small modelling dataset and 250-token truncation.

# 8. Recommendations
•	Increase the modelling sample. The most important next step is to use substantially more posts per author. The current ten-author subset contains only 264 training examples, so a larger and more balanced subset should be used before drawing conclusions about LSTM effectiveness.
•	Address class imbalance. Where the full author set is used, class frequencies should be examined and appropriate balancing, class weights or sampling strategies considered.
•	Improve document representation. A 250-token limit discards information from longer posts. Future work could use longer sequences, hierarchical modelling, document chunking and aggregation of predictions across multiple chunks.
•	Benchmark against simpler models. TF-IDF with logistic regression or linear SVM should be established as a baseline. Character n-grams can also be valuable for stylometry because punctuation, spelling and character-level patterns may capture author-specific writing habits.
•	Strengthen evaluation. The current test set contains only 39 observations. A larger evaluation set and per-class support are needed. Repeated experiments or cross-validation should be considered where compatible with the predefined author-identification protocol.
•	Only deploy after validation. The present results should not be used for consequential authorship decisions. The model is a prototype demonstrating the workflow rather than a production-ready classifier.

# 9. Limitations and Ethical Considerations
The main drawback of the research is the decrease from 1,000 authors to only 10 authors due to the limitations of computation. Although this allows for a deployable experiment, it completely alters the classification problem at hand. Moreover, the modelling data set is also small; the test data set comprises only 39 entries, while the constant length of 250 tokens could also lose the useful stylistic features of longer messages.
The performance of authorship prediction should be considered with caution, as the model might rely on such features as topics, named entities, formatting peculiarities and other empirical associations. Therefore, it is necessary to treat the results as some sort of probability, rather than proof of authorship. Any application in real life is only possible after appropriate validation, recording of model error rates, and after taking care of privacy and governance issues.

# 10. Conclusion
The main drawback of the research is the decrease from 1,000 authors to only 10 authors due to the limitations of computation. Although this allows for a deployable experiment, it completely alters the classification problem at hand. Moreover, the modeling data set is also small; the test data set comprises only 39 entries, while the constant length of 250 tokens could also lose the useful stylistic features of longer messages.
The performance of authorship prediction should be considered with caution, as the model might rely on such features as topics, named entities, formatting peculiarities and other empirical associations. Therefore, it is necessary to treat the results as some sort of probability, rather than proof of authorship. Any application in real life is only possible after appropriate validation, recording of model error rates, and after taking care of privacy and governance issues.

# References
Wang, H. (2022) Blog-1K. Zenodo. Available at: https://doi.org/10.5281/zenodo.7455623 (Accessed: 14 September 2026).
