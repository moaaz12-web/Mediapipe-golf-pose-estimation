import numpy as np
import math
import pandas as pd
from compute import compute
from writeDataframe import writeDataframe


def main(video1_path, video2_path, base_directory_path, video1_type, video2_type):
    # Call the main function to get the dataframes for the two videos
    dataframe1 = compute(video1_path, base_directory_path, video1_type)
    dataframe2 = compute(video2_path, base_directory_path, video2_type)

    writeDataframe(dataframe1, dataframe2, base_directory_path)
    # print("Similarity is ", math.fabs(percentage1-percentage2))

    
    
    return dataframe1, dataframe2


df1, df2 = main(r'C:\Users\moaaz\3D Objects\vid2.mp4',r'C:\Users\moaaz\3D Objects\vid3.mp4', 'D:/finale_last', 1, 0)
                        # |                                  # |
                                                             # |
                        # |
                # video1_path,                             video2_path

                                                # video2_type = 0 for user uploaded video
                                                # video2_type = 1 for etalon video
                                                # video1_type = 0 for user uploaded video
                                                # video1_type = 1 for etalon video
                                                                
