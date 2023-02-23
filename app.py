import numpy as np

import pandas as pd
from compute import compute


def main(video1_path, video2_path, base_directory_path, video1_type, video2_type):
    # Call the main function to get the dataframes for the two videos
    dataframe1, percentage1 = compute(video1_path, base_directory_path, video1_type)
    dataframe2, percentage2 = compute(video2_path, base_directory_path, video2_type)
    
    # Compute the absolute difference between the dataframes
    comparison = dataframe1.subtract(dataframe2).abs()
    print("Dataframe no 1 percentage value is ", percentage1)
    print("Dataframe no 2 percentage value is ", percentage2)

    # Return a new dataframe with the same columns and index as the input dataframes
    return pd.DataFrame(comparison.values, columns=comparison.columns, index=comparison.index), dataframe1, dataframe2


res, df1, df2 = main(r'C:\Users\moaaz\3D Objects\vid2.mp4',r'C:\Users\moaaz\3D Objects\vid3.mp4', 'D:/abc', 1, 0)