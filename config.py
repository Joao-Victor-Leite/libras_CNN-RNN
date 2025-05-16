import string

cnn_path_data_train = 'dataset/cnn/train/'
cnn_path_data_test = 'dataset/cnn/test/'

rnn_path_data_train = 'dataset/rnn/train'
rnn_path_data_test = 'dataset/rnn/test'

path_data_train = 'pre_processed/train/'
path_data_test = 'pre_processed/test/'


static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]
dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']


train_video_sequence = 30
test_video_sequence = 6
frame_sequence = 30
