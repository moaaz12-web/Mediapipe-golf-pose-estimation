
import mediapipe as mp
import numpy as np
import cv2
import os
import pandas as pd
import time
from save_images import save_data
from show_angle import show_angle
from calculate_and_append_angle import calculate_and_append_angle
from calculate_and_print_average_angle import calculate_and_print_average_angle
from create_dataframe import create_dataframe
from get_video_length import get_video_length

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def compute(vid_path, base_directory, vid_type):
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # count = 0

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        df = pd.DataFrame(columns=['time', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                          'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_wrist', 'right_wrist'])

        start_time = time.time()
        left_elbow_angle_list = []
        right_elbow_angle_list = []
        left_knee_angle_list = []
        right_knee_angle_list = []
        left_wrist_angle_list = []
        right_wrist_angle_list = []
        left_shoulder_angle_list = []
        right_shoulder_angle_list = []
        left_hip_angle_list = []
        right_hip_angle_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_elbow_angle = calculate_and_append_angle(
                    left_shoulder, left_elbow, left_wrist, left_elbow_angle_list)
                right_elbow_angle = calculate_and_append_angle(
                    right_shoulder, right_elbow, right_wrist, right_elbow_angle_list)
                left_shoulder_angle = calculate_and_append_angle(
                    left_elbow, left_shoulder, left_hip, left_shoulder_angle_list)
                right_shoulder_angle = calculate_and_append_angle(
                    right_elbow, right_shoulder, right_hip, right_shoulder_angle_list)
                left_knee_ankle = calculate_and_append_angle(
                    left_hip, left_knee, left_ankle, left_knee_angle_list)
                right_knee_ankle = calculate_and_append_angle(
                    right_hip, right_knee, right_ankle, right_knee_angle_list)
                left_hip_angle = calculate_and_append_angle(
                    left_knee, left_hip, left_shoulder, left_hip_angle_list)
                right_hip_angle = calculate_and_append_angle(
                    right_knee, right_hip, right_shoulder, right_hip_angle_list)
                # Wrist hinge angle: The left_wrist_angle and right_wrist_angle are the angles between the lines connecting the elbow, wrist, and hip on each side,
                # which give an indication of the positioning and movement of the wrists during the swing.

                left_wrist_angle = calculate_and_append_angle(
                    left_elbow, left_wrist, left_hip, left_wrist_angle_list)
                right_wrist_angle = calculate_and_append_angle(
                    right_elbow, right_wrist, right_hip, right_wrist_angle_list)

                # Visualize angles
                show_angle(image, left_elbow_angle,
                           left_elbow, width, height, "LE")
                show_angle(image, right_elbow_angle,
                           right_elbow, width, height, "RE")
                show_angle(image, left_shoulder_angle,
                           left_shoulder, width, height, "LS")
                show_angle(image, right_shoulder_angle,
                           right_shoulder, width, height, "RS")
                show_angle(image, left_knee_ankle,
                           left_knee, width, height, "LK")
                show_angle(image, right_knee_ankle,
                           right_knee, width, height, "RK")

                show_angle(image, left_hip_angle,
                           left_hip, width, height, "LH")
                show_angle(image, right_hip_angle,
                           right_hip, width, height, "RH")
                show_angle(image, left_wrist_angle,
                           left_wrist, width, height, "LW")
                show_angle(image, right_wrist_angle,
                           right_wrist, width, height, "RW")
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=1)
                                      )

            # Render detections
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:

                left_elbow_angle_list, LE_avg_angle = calculate_and_print_average_angle(
                    left_elbow_angle_list, "LEFT_ELBOW_ANGLE")
                right_elbow_angle_list, RE_avg_angle = calculate_and_print_average_angle(
                    right_elbow_angle_list, "RIGHT_ELBOW_ANGLE")
                left_shoulder_angle_list, LS_avg_angle = calculate_and_print_average_angle(
                    left_shoulder_angle_list, "LEFT_SHOULDER_ANGLE")
                right_shoulder_angle_list, RS_avg_angle = calculate_and_print_average_angle(
                    right_shoulder_angle_list, "RIGHT_SHOULDER_ANGLE")
                left_knee_angle_list, LK_avg_angle = calculate_and_print_average_angle(
                    left_knee_angle_list, "LEFT_KNEE_ANGLE")
                right_knee_angle_list, RK_avg_angle = calculate_and_print_average_angle(
                    right_knee_angle_list, "RIGHT_KNEE_ANGLE")
                left_hip_angle_list, LH_avg_angle = calculate_and_print_average_angle(
                    left_hip_angle_list, "LEFT_HIP_ANGLE")
                right_hip_angle_list, RH_avg_angle = calculate_and_print_average_angle(
                    right_hip_angle_list, "RIGHT_HIP_ANGLE")
                left_wrist_angle_list, LW_avg_angle = calculate_and_print_average_angle(
                    left_wrist_angle_list, "LEFT_WRIST_ANGLE")
                right_wrist_angle_list, RW_avg_angle = calculate_and_print_average_angle(
                    right_wrist_angle_list, "RIGHT_WRIST_ANGLE")

                # start_time = time.time()
                temp = create_dataframe(start_time, LE_avg_angle, RE_avg_angle, LS_avg_angle, RS_avg_angle, LK_avg_angle,
                                        RK_avg_angle, LH_avg_angle, RH_avg_angle, LW_avg_angle, RW_avg_angle)

                
                df = pd.concat([df, temp])

                save_data(image, base_directory, count, vid_type)

                start_time += 1
                # count += 1

            # cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        percentage /= count

        return df, percentage




import mediapipe as mp
import numpy as np
import cv2
import os
import pandas as pd
import time
from save_images import save_data
from show_angle import show_angle
from calculate_and_append_angle import calculate_and_append_angle
from calculate_and_print_average_angle import calculate_and_print_average_angle
from create_dataframe import create_dataframe
from get_video_length import get_video_length

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def compute(vid_path, base_directory, vid_type):
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 1

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        df = pd.DataFrame(columns=['time', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                          'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_wrist', 'right_wrist'])

        start_time = time.time() + 1
        left_elbow_angle_list = []
        right_elbow_angle_list = []
        left_knee_angle_list = []
        right_knee_angle_list = []
        left_wrist_angle_list = []
        right_wrist_angle_list = []
        left_shoulder_angle_list = []
        right_shoulder_angle_list = []
        left_hip_angle_list = []
        right_hip_angle_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_elbow_angle = calculate_and_append_angle(
                    left_shoulder, left_elbow, left_wrist, left_elbow_angle_list)
                right_elbow_angle = calculate_and_append_angle(
                    right_shoulder, right_elbow, right_wrist, right_elbow_angle_list)
                left_shoulder_angle = calculate_and_append_angle(
                    left_elbow, left_shoulder, left_hip, left_shoulder_angle_list)
                right_shoulder_angle = calculate_and_append_angle(
                    right_elbow, right_shoulder, right_hip, right_shoulder_angle_list)
                left_knee_ankle = calculate_and_append_angle(
                    left_hip, left_knee, left_ankle, left_knee_angle_list)
                right_knee_ankle = calculate_and_append_angle(
                    right_hip, right_knee, right_ankle, right_knee_angle_list)
                left_hip_angle = calculate_and_append_angle(
                    left_knee, left_hip, left_shoulder, left_hip_angle_list)
                right_hip_angle = calculate_and_append_angle(
                    right_knee, right_hip, right_shoulder, right_hip_angle_list)
                # Wrist hinge angle: The left_wrist_angle and right_wrist_angle are the angles between the lines connecting the elbow, wrist, and hip on each side,
                # which give an indication of the positioning and movement of the wrists during the swing.

                left_wrist_angle = calculate_and_append_angle(
                    left_elbow, left_wrist, left_hip, left_wrist_angle_list)
                right_wrist_angle = calculate_and_append_angle(
                    right_elbow, right_wrist, right_hip, right_wrist_angle_list)

                # Visualize angles
                show_angle(image, left_elbow_angle,
                           left_elbow, width, height, "LE")
                show_angle(image, right_elbow_angle,
                           right_elbow, width, height, "RE")
                show_angle(image, left_shoulder_angle,
                           left_shoulder, width, height, "LS")
                show_angle(image, right_shoulder_angle,
                           right_shoulder, width, height, "RS")
                show_angle(image, left_knee_ankle,
                           left_knee, width, height, "LK")
                show_angle(image, right_knee_ankle,
                           right_knee, width, height, "RK")

                show_angle(image, left_hip_angle,
                           left_hip, width, height, "LH")
                show_angle(image, right_hip_angle,
                           right_hip, width, height, "RH")
                show_angle(image, left_wrist_angle,
                           left_wrist, width, height, "LW")
                show_angle(image, right_wrist_angle,
                           right_wrist, width, height, "RW")
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=1)
                                      )

            # Render detections
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:

                left_elbow_angle_list, LE_avg_angle = calculate_and_print_average_angle(
                    left_elbow_angle_list, "LEFT_ELBOW_ANGLE")
                right_elbow_angle_list, RE_avg_angle = calculate_and_print_average_angle(
                    right_elbow_angle_list, "RIGHT_ELBOW_ANGLE")
                left_shoulder_angle_list, LS_avg_angle = calculate_and_print_average_angle(
                    left_shoulder_angle_list, "LEFT_SHOULDER_ANGLE")
                right_shoulder_angle_list, RS_avg_angle = calculate_and_print_average_angle(
                    right_shoulder_angle_list, "RIGHT_SHOULDER_ANGLE")
                left_knee_angle_list, LK_avg_angle = calculate_and_print_average_angle(
                    left_knee_angle_list, "LEFT_KNEE_ANGLE")
                right_knee_angle_list, RK_avg_angle = calculate_and_print_average_angle(
                    right_knee_angle_list, "RIGHT_KNEE_ANGLE")
                left_hip_angle_list, LH_avg_angle = calculate_and_print_average_angle(
                    left_hip_angle_list, "LEFT_HIP_ANGLE")
                right_hip_angle_list, RH_avg_angle = calculate_and_print_average_angle(
                    right_hip_angle_list, "RIGHT_HIP_ANGLE")
                left_wrist_angle_list, LW_avg_angle = calculate_and_print_average_angle(
                    left_wrist_angle_list, "LEFT_WRIST_ANGLE")
                right_wrist_angle_list, RW_avg_angle = calculate_and_print_average_angle(
                    right_wrist_angle_list, "RIGHT_WRIST_ANGLE")

                start_time = time.time()
                temp = create_dataframe(count, LE_avg_angle, RE_avg_angle, LS_avg_angle, RS_avg_angle, LK_avg_angle,
                                        RK_avg_angle, LH_avg_angle, RH_avg_angle, LW_avg_angle, RW_avg_angle)

                df = pd.concat([df, temp])

                save_data(image, base_directory, count, vid_type)

                start_time += 2
                count += 1

            # cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        return df

