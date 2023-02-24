def calculate_and_print_average_angle(angle_list, joint_name):
    if len(angle_list) > 0:
        # Calculate the average of the angle list
        average_angle = sum(angle_list) / len(angle_list)
        # print("Average {} angle: {:.2f}".format(joint_name, average_angle))
        # Reset the angle list
        angle_list = []
    else:
        average_angle = 0


    return angle_list, round(average_angle,2)



