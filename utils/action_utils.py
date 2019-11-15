def decode_action(a):
    speed_action = a//3
    turn_action = a%3
    return speed_action, turn_action

def encode_action(speed_action, turn_action):
    return speed_action*3+turn_action