def calculate_and_print_average_angle(angle_list, joint_name, start_time, time_threshold=2.0):
    if len(angle_list) > 0:
        # Calculate the average of the angle list
        average_angle = sum(angle_list) / len(angle_list)
        # Print the average angle
        # print("List is ", angle_list)
        print("Average {} angle: {:.2f}".format(joint_name, average_angle))
        # Reset the angle list and start time
        angle_list = []
        # start_time = time.time()
    else:
        average_angle = 0


    return angle_list, average_angle



