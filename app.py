import numpy as np
import math
import pandas as pd
from compute import compute
from writeDataframe import writeDataframe
import sys


def main(video1_path, video2_path, base_directory_path, video1_type, video2_type, framework_length_seconds):

    dataframe1 = compute(video1_path, base_directory_path, video1_type, framework_length_seconds)
    dataframe2 = compute(video2_path, base_directory_path, video2_type, framework_length_seconds)

    writeDataframe(dataframe1, dataframe2, base_directory_path)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python main.py <video1_path> <video2_path> <base_directory_path> <video1_type> <video2_type> <framework_length_seconds>")
        sys.exit(1)

    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    base_directory_path = sys.argv[3]
    video1_type = int(sys.argv[4])
    video2_type = int(sys.argv[5])
    framework_length_seconds = float(sys.argv[6])

    main(video1_path, video2_path, base_directory_path, video1_type, video2_type, framework_length_seconds)



# df1, df2 = main(r'C:\Users\moaaz\3D Objects\vid2.mp4',r'C:\Users\moaaz\3D Objects\vid3.mp4', 'D:/finale_last', 1, 0, 3)
                        # |                                  # |                                                         |
                                                             # |                                                         |
                        # |                                  # |                                                         |
                # video1_path                             video2_path                                              framework_length_seconds

                                              
                                                # 0 for user uploaded video                                         Specifies the length for each image split, example, 1 sec, 0.5 sec, 0.25 sec etc
                                                # 1 for etalon video
                                                                