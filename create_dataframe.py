import pandas as pd

def create_dataframe(elapsed_time, left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, 
                     right_knee_angle, left_hip_angle, right_hip_angle, left_wrist_angle, right_wrist_angle):
    # Create an empty dataframe
    df = pd.DataFrame(columns=['time', 'left_elbow', 'right_elbow', 'left_shoulder','right_shoulder', 
                               'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_wrist', 'right_wrist'])
    
        
    # Append the angles and time to the dataframe
    df = pd.concat([df, pd.DataFrame({'time': [elapsed_time], 
                                  'left_elbow': [left_elbow_angle], 
                                  'right_elbow': [right_elbow_angle], 
                                  'left_shoulder': [left_shoulder_angle], 
                                  'right_shoulder': [right_shoulder_angle], 
                                  'left_knee': [left_knee_angle], 
                                  'right_knee': [right_knee_angle], 
                                  'left_hip': [left_hip_angle], 
                                  'right_hip': [right_hip_angle], 
                                  'left_wrist': [left_wrist_angle], 
                                  'right_wrist': [right_wrist_angle]})], ignore_index=True)
    
    return df
