import string

path_data_train = 'pre_processed/train/'
path_data_test = 'pre_processed/test/'


static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]
dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']


train_video_sequence = 30
test_video_sequence = 6
frame_sequence = 30
