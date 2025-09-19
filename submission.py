import pandas as pd
import numpy as np

def save_submission(y_pred, filename='submission.csv', start_id=101):
    """
    Save predictions to a CSV file in the required submission format.
    
    Parameters:
        y_pred (array-like): Array or list of predicted categories.
        filename (str): Name of the output CSV file.
        start_id (int): Starting ID for the test samples (default is 101).
    
    Raises:
        ValueError: If the number of predictions is not 50.
    """
    y_pred = np.array(y_pred)
    
    if len(y_pred) != 50:
        raise ValueError("Predictions must contain exactly 50 elements.")
    
    # Create a DataFrame with the correct IDs and predicted categories
    submission_df = pd.DataFrame({
        'Id': list(range(start_id, start_id + len(y_pred))),
        'Category': y_pred.astype(int)  # Ensure integer format
    })
    
    # Save the DataFrame to a CSV file
    submission_df.to_csv(filename, index=False)
    print(f"{filename} saved successfully.")
