import string

path_video_train = 'hand_capture/pre_processed/train/'
path_video_test = 'hand_capture/pre_processed/test/'



static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]
dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']


train_video_sequence = 30
test_video_sequence = 6
frame_sequence = 30