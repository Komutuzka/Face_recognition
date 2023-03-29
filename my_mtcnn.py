from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from os import listdir
from os.path import isdir
from numpy import savez_compressed


def extract_face(filename, required_size=(160, 160)):
    # print("Начало extract_face")
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()

    results = detector.detect_faces(pixels)

    arr_face = list()
    for i in results:
        if i['confidence'] >= 0.97:
            x1, y1, width, height = i['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize(required_size)
            # pyplot.imshow(image)
            # pyplot.show()
            arr_face.append(asarray(image))
    return arr_face


def load_faces(directory):
    # print("Начало load_faces")
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        for i in face:
            faces.append(i)
    return faces


def load_dataset(directory):
    # print("Начало load_dataset. Dir: ", directory)
    X, y = list(), list()
    lll = 0
    for subdir in listdir(directory):
        print("Subdir: ", subdir)
        path = directory + subdir + '\\'
        # print("Path: ", path)
        if not isdir(path):
            print("IS NOT ISDIR", path)
            continue
        try:
            faces = load_faces(path)
            labels = [subdir for _ in range(len(faces))]
            print(lll, ' >loaded %d examples for class: %s' % (len(faces), subdir))
            # Соединяем лица с меткой
            X.extend(faces)
            y.extend(labels)
        except:
            print("Except")

        lll = lll + 1
    return asarray(X), asarray(y)


def main():
    print("Начало")
    trainX, trainy = load_dataset('Folders_faces\\')
    savez_compressed('Npz_files\\faces-dataset.npz', trainX, trainy)
    print("Конец")


# if __name__ == '__main__':
#     main()
