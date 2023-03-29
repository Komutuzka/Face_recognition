import numpy as np
from numpy import load
from numpy import asarray
from numpy import expand_dims
from numpy.lib.npyio import savez

from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from my_mtcnn import load_faces
from processing import get_embedding

from tensorflow.keras.models import load_model


def load_newFaces(path):
    model = load_model('facenet_keras.h5')
    my_faces = asarray(load_faces(path))
    newFaces = list()
    for face_pixels in my_faces:
        embedding = get_embedding(model, face_pixels)
        newFaces.append(embedding)
    newFaces_emb = asarray(newFaces)
    return newFaces_emb, my_faces

def main():
    # Загрузка новых лиц
    my_faces_emb, my_faces = load_newFaces('My_faces\\')

    print("Начало обучения")
    # Загрузка датасета
    data = load('Npz_files\\faces-embeddings.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    # trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print("Загрузка датасета")

    # Нормализация входных векторов
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    print("Нормализация закончена")

    # кодирование категориальных признаков
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    print("Кодирование закончено")

    # обучение модели
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    print("Начало тестирования")
    # testX = in_encoder.transform(testX)
    # testy = out_encoder.transform(testy)

    ind = 0
    for i in my_faces:
        selection = ind

        face_pixels = my_faces[selection]  # <class 'numpy.ndarray'>
        face_emb = my_faces_emb[selection]  # <class 'numpy.ndarray'>
        # face_class = testy[selection]
        # face_name = out_encoder.inverse_transform([face_class])

        # предсказание
        samples = expand_dims(face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        # получение имени
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        # вывод
        if (class_probability < 60):
            print('Неизвестен')
        else:
            print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

        pyplot.imshow(face_pixels)

        if (class_probability < 60):
            title = 'Неизвестен'
        else:
            title = '%s (%.3f)' % (predict_names[0], class_probability)

        pyplot.title(title)
        plt.savefig('Name_my_faces/' + str(ind) + '_face.jpg')
        ind += 1


if __name__ == '__main__':
    main()
