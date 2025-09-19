import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained classification model on the test set using accuracy and ROC AUC. 
    Prints the classification report, confusion matrix, accuracy, and ROC AUC score. 
    Also displays the confusion matrix visually.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Confusion matrix display
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc:.2f}")


def evaluate_pca_pipeline_with_cv(X, y, classifier, n_components=14, cv=5, scoring='accuracy'):
    """
    Builds a pipeline (StandardScaler + PCA + classifier) and evaluates it using cross-validation.

    Parameters:
        X: Feature matrix
        y: Target labels
        classifier: Any sklearn-compatible classifier (e.g., RandomForestClassifier, XGBClassifier)
        n_components (int): Number of PCA components
        cv (int): Number of cross-validation folds
        scoring (str): Metric to evaluate (default: 'accuracy')

    Returns:
        mean_score: Mean cross-validation score
        std_score: Standard deviation of scores
        scores: All individual CV scores
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('clf', classifier)
    ])

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    mean_score = scores.mean()
    std_score = scores.std()

    print(f"{scoring.capitalize()} with PCA: {mean_score:.4f} ± {std_score:.4f}")

    return mean_score, std_score, scores
