import string

path_data_train = 'dataset/lstm/train'
path_data_test = 'dataset/lstm/test'


static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]
dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']


train_video_sequence = 80
test_video_sequence = 20
frame_sequence = 30
