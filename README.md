# nlp-website-sentiment-prediction

This is an NLP project which implements 3 different types of classifiers
to attempt to predict the category of a website based upon the text within

The data for the project was found online and is given in the spreadsheet
below:
[website_classification.csv](https://github.com/woods0813/nlp-website-sentiment-prediction/files/11804422/website_classification.csv)

The classifiers used were Random Forest Classifier, Multinomial Naive Bayes, and 
Xtreme Gradient Boost. The performance between the gradient boost and naive bayes
were similar, but naive bayes was significantly quicker and easier to train, 
up to 4 orders of magnitude faster. Given more time and more parameters to explore,
its possible the gradient boost could have provided better performance, but the increase
would be nearly negligible given the loss of efficiency

The produced an accuracy score of 0.884, with an alpha of 0.1 with the given parameters and 
cross validation attempts
