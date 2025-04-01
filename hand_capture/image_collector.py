import cv2
import os
import string

image_x, image_y = 64, 64
dataset_train_size = 800
dataset_test_size = 200
dataset_size = dataset_train_size + dataset_test_size

dir_img_train = './pre_processed/train/'
dir_img_test = './pre_processed/test/'

cap = cv2.VideoCapture(0)

def get_next_index(directory):
    existing_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    if not existing_files:
        return 0

    return max(int(f.split('.')[0]) for f in existing_files) + 1

def capture_images(letter):
    class_dir_train = os.path.join(dir_img_train, letter)
    class_dir_test = os.path.join(dir_img_test, letter)

    os.makedirs(class_dir_train, exist_ok=True)
    os.makedirs(class_dir_test, exist_ok=True)

    print(f'Coletando dados para a letra {letter}')
    print('Pressione "s" para iniciar a captura.')

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        roi = frame[100:300, 425:625]
        roi_resized = cv2.resize(roi, (image_x, image_y))

        cv2.imshow('ROI', roi_resized)

        if cv2.waitKey(5) == ord('s'):
            break

    train_index = get_next_index(class_dir_train)
    test_index = get_next_index(class_dir_test)

    count = 0

    while count < dataset_size:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        roi = frame[100:300, 425:625]
        roi_resized = cv2.resize(roi, (image_x, image_y))

        cv2.imshow('ROI', roi_resized)
        cv2.waitKey(25)

        if count < dataset_train_size:
            cv2.imwrite(os.path.join(class_dir_train, f'{train_index}.jpg'), roi_resized)
            train_index += 1
        else:
            cv2.imwrite(os.path.join(class_dir_test, f'{test_index}.jpg'), roi_resized)
            test_index += 1

        count += 1

    print(f'Captura para {letter} concluída.')
    print('Pressione "n" para continuar.')

    while True:
        if cv2.waitKey(1) == ord('n'):
            break


os.makedirs(dir_img_train, exist_ok=True)
os.makedirs(dir_img_test, exist_ok=True)

static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]

mode = input('Digite 1 para escolher uma letra ou 2 para capturar todas as letras automaticamente: ')
if mode == '1':
    letter = input(f'Escolha uma letra dentre {static_letters}: ').upper()

    if letter in static_letters:
        capture_images(letter)
    else:
        print('Letra inválida.')

elif mode == '2':
    for letter in static_letters:
        capture_images(letter)
else:
    print('Opção inválida.')

cap.release()
cv2.destroyAllWindows()
