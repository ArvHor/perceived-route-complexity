def get_turn_type(bearing_origin_to_intermediate, bearing_intermediate_to_destination):
    print("----------NEW------------")
    print(f"(uv): {bearing_origin_to_intermediate}")
    print(f"(vw): {bearing_intermediate_to_destination}")


    bearing_difference = (bearing_origin_to_intermediate - bearing_intermediate_to_destination)
    print(f"(uv)-(vw) = diff:{bearing_difference}")


    bearing_difference = bearing_difference % 360
    print(f"diff % 360 = {bearing_difference}")

    if bearing_difference > 180:
        bearing_difference -= 360  # Normalize to -180 to 180
        print(f"if diff > 180: diff - 360 = {bearing_difference}")

    if -45 < bearing_difference < 45:
        print(f"STRAGIHT -45 < diff < 45: {bearing_difference}")
        if -45 <= bearing_difference < -10:
            return 'slight_right_turn'
        elif 10 < bearing_difference <= 45:
            return 'slight_left_turn'
        elif -10 <= bearing_difference <= 10:
            return 'straight'
    elif 45 <= bearing_difference:
        print(f"RIGHT TURN diff < 45: {bearing_difference}")
        return 'right_turn'
    elif bearing_difference <= -45:
        print(f"LEFT TURN diff < -45: {bearing_difference}")
        return 'left_turn'


def old_get_turn_type(bearing_origin_to_intermediate, bearing_intermediate_to_destination):
    print("----------OLD------------")
    print(f"(uv): {bearing_origin_to_intermediate}")
    print(f"(vw): {bearing_intermediate_to_destination}")

    bearing_difference = (bearing_origin_to_intermediate - bearing_intermediate_to_destination)
    print(f"(uv)-(vw) = diff:{bearing_difference}")

    bearing_difference = bearing_difference % 360
    print(f"diff % 360 = {bearing_difference}")

    if bearing_difference > 180:
        bearing_difference -= 360  # Normalize to -180 to 180
        print(f"if diff > 180: diff - 360 = {bearing_difference}")

    if -45 < bearing_difference < 45:
        print(f"STRAGIHT -45 < diff < 45: {bearing_difference}")
        return 'straight'
    elif 45 <= bearing_difference:
        print(f"LEFT TURN diff > 45: {bearing_difference}")
        return 'left_turn'
    elif bearing_difference <= -45:
        print(f"RIGHT TURN diff < -45: {bearing_difference}")
        return 'right_turn'
outputs = []


def is_t_turn(azimuth_in, azimuth_out1, azimuth_out2):

    intersection_turns = []
    intersection_turns.append(get_turn_type(azimuth_in, azimuth_out1))
    intersection_turns.append(get_turn_type(azimuth_in, azimuth_out2))
    print(intersection_turns)
    if 'left_turn' in intersection_turns and 'right_turn' in intersection_turns:
        print("T-Turn")
        return True
    else:
        print("Not T-Turn")
        return False



# left turn, going north then west
outputs.append(get_turn_type(90, 180))

# slight left turn, going north then northwest
outputs.append(get_turn_type(90, 110))

# left turn, going south then east
outputs.append(get_turn_type(-90, 5))

# sharp left turn, going northwest then southwest
outputs.append(get_turn_type(100, -100))

# straight, going north then north
outputs.append(get_turn_type(90, 90))

# straight going south then south
outputs.append(get_turn_type(-90, -90))

# straight going east then east
outputs.append(get_turn_type(5, -5))

# straight going west then west
outputs.append(get_turn_type(175, -175))

# slight left, going west then southwest
outputs.append(get_turn_type(175, -155))

# crazy turn, going north then south
outputs.append(get_turn_type(90, -95))



old_outputs = []

# left turn, going north then west
old_outputs.append(old_get_turn_type(90, 180))

# slight left turn, going north then northwest
old_outputs.append(old_get_turn_type(90, 110))

# left turn, going south then east
old_outputs.append(old_get_turn_type(-90, 5))

# sharp left turn, going northwest then southwest
old_outputs.append(old_get_turn_type(100, -100))

# straight, going north then north
old_outputs.append(old_get_turn_type(90, 90))

# straight going south then south
old_outputs.append(old_get_turn_type(-90, -90))

# straight going east then east
old_outputs.append(old_get_turn_type(5, -5))

# straight going west then west
old_outputs.append(old_get_turn_type(175, -175))

# slight left, going west then southwest
old_outputs.append(old_get_turn_type(175, -155))

# crazy turn, going north then south
old_outputs.append(old_get_turn_type(90, -95))
print(outputs)
print(old_outputs)


print(old_get_turn_type(-5, -155))

print(get_turn_type(-5, -155))

print(old_get_turn_type(150, 50))

print(get_turn_type(150, 50))

print(is_t_turn(90, 180, 0))

print(is_t_turn(175, 90, -90))


print(old_get_turn_type(-145, -145))