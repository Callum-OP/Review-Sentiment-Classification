# Import pickle Python’s built-in persistence mode
import pickle
# Import numpy for finding certain ratings when pre-processing
import numpy as np
# Import pandas for reading and editing the dataset
import pandas as pd
import matplotlib.pyplot as plt
# Import metrics for showing the f1 score, loss and the confusion matrix for both models
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn import metrics
# Import count vectorizer for feature extraction
from sklearn.feature_extraction.text import CountVectorizer
# Import the first model to be used for sentiment analysis
from sklearn.naive_bayes import MultinomialNB
# Import the second model to be used for sentiment analysis
from sklearn.neural_network import MLPClassifier
# Import plotter
from mlxtend.plotting import plot_confusion_matrix

if __name__ == '__main__':

    # Reading dataset and preprocessing it
    print("Preparing dataset...")
    data = pd.read_csv("Amazon_reviews.csv")
    print(data.head())
    # Remove missing values
    data.dropna(inplace=True)
    # Add sentiment to the data, positive, neutral and negative
    data["Sentiment"] = np.where(data["Rating"] < 3, 0,
                                 (np.where(data["Rating"] == 3, 1, (np.where(data["Rating"] > 3, 2, 1)))))
    pd.crosstab(index=data["Sentiment"], columns="Total count")
    print(data.head())

    print()
    print("Splitting data...")
    X = data["Reviews"]
    y = data["Sentiment"]
    from sklearn.model_selection import train_test_split
    # Split data into 60% training and 40% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)
    # Count vectorize used to change reviews training data
    vect = CountVectorizer().fit(X_train)
    # Transform training data into a matrix
    X_train_vector = vect.transform(X_train)
    print('Matrix', X_train_vector.shape)
    print(X_train_vector.toarray()[1])

    print()
    print("Defining Multinomial NaiveBayes model...")
    # Define the first model as Multinomial NaiveBayes
    mnbModel = MultinomialNB()
    mnbModel.fit(X_train_vector, y_train)
    # Make predictions with the MultinomialNB model
    mnbModelPredictions = mnbModel.predict(vect.transform(X_test))

    print()
    print("Evaluating MultinomialNB model...")
    # Calculate accuracy of the MultinomialNB model
    print("Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, mnbModelPredictions)))
    # Get confusion matrix of the MultinomialNB model
    print(confusion_matrix(y_test, mnbModelPredictions))
    # Get classification report of the MultinomialNB model
    print(classification_report(y_test, mnbModelPredictions))
    # Plots the confusion matrix in a more visual friendly format
    plot_confusion_matrix(conf_mat=confusion_matrix(y_test, mnbModelPredictions),
                          colorbar=True, show_absolute=False, show_normed=True)
    plt.show()

    print()
    print("Defining Multilayer Perceptron model...")
    mlpModel = MLPClassifier(max_iter=10, learning_rate_init=0.001, activation="relu", verbose=True)
    mlpModel.fit(X_train_vector.toarray(), y_train)
    # Make predictions with the second model
    mlpPredictions = mlpModel.predict(vect.transform(X_test))

    print()
    print("Evaluating MLP model...")
    # Calculate accuracy of the MLP model
    print("Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, mlpPredictions)))
    # Show the loss of the MLP model using log loss
    print("Loss: ", log_loss(y_test, mlpModel.predict_proba(vect.transform(X_test))))
    # Get confusion matrix of the MLP model
    print(confusion_matrix(y_test, mlpPredictions))
    # Get classification report of the MLP model
    print(classification_report(y_test, mlpPredictions))
    # Plots the confusion matrix of the MLP model in a more visual friendly format
    plot_confusion_matrix(conf_mat=confusion_matrix(y_test, mlpPredictions),
                          colorbar=True, show_absolute=False, show_normed=True)
    plt.show()

    # Save the model
    bestModel = pickle.dumps(mlpModel)

    print()
    print("Selecting best performing model...")

    # Load the best performing model
    model = pickle.loads(bestModel)
    print(model)

    print()
    print("Predicting new reviews...")

    # Predict new reviews to see the results from the model!
    review1 = "We bought this watch 2 months ago and it already works no longer"
    print(review1)
    if model.predict(vect.transform([review1])) == 0:
        print(model.predict(vect.transform([review1])), "Negative")
    elif model.predict(vect.transform([review1])) == 1:
        print(model.predict(vect.transform([review1])), "Neutral")
    elif model.predict(vect.transform([review1])) == 2:
        print(model.predict(vect.transform([review1])), "Positive")

    review2 = "This product is perfect, definitely great!"
    print(review2)
    if model.predict(vect.transform([review2])) == 0:
        print(model.predict(vect.transform([review2])), "Negative")
    elif model.predict(vect.transform([review2])) == 1:
        print(model.predict(vect.transform([review2])), "Neutral")
    elif model.predict(vect.transform([review2])) == 2:
        print(model.predict(vect.transform([review2])), "Positive")

    review3 = "Didn't work but refunded without an issue."
    print(review3)
    if model.predict(vect.transform([review3])) == 0:
        print(model.predict(vect.transform([review3])), "Negative")
    elif model.predict(vect.transform([review3])) == 1:
        print(model.predict(vect.transform([review3])), "Neutral")
    elif model.predict(vect.transform([review3])) == 2:
        print(model.predict(vect.transform([review3])), "Positive")

    review4 = "Good phone. I had a problem where the time and date etc"
    print(review4)
    if model.predict(vect.transform([review4])) == 0:
        print(model.predict(vect.transform([review4])), "Negative")
    elif model.predict(vect.transform([review4])) == 1:
        print(model.predict(vect.transform([review4])), "Neutral")
    elif model.predict(vect.transform([review4])) == 2:
        print(model.predict(vect.transform([review4])), "Positive")

    review5 = "The key pad is not very user friendly. Otherwise it's an ok phone for a low volume user"
    print(review5)
    if model.predict(vect.transform([review5])) == 0:
        print(model.predict(vect.transform([review5])), "Negative")
    elif model.predict(vect.transform([review5])) == 1:
        print(model.predict(vect.transform([review5])), "Neutral")
    elif model.predict(vect.transform([review5])) == 2:
        print(model.predict(vect.transform([review5])), "Positive")