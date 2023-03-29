from numpy import load
from numpy import asarray
from numpy import expand_dims
from numpy import savez_compressed

from tensorflow.keras.models import load_model


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def main():
    print("Начало")
    data_faces = load('Npz_files\\faces-dataset.npz')
    trainX, trainy = data_faces['arr_0'], data_faces['arr_1']
    model = load_model('facenet_keras.h5')
    print("1")
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print("2")
    savez_compressed('Npz_files\\faces-embeddings.npz', newTrainX, trainy)
    print("Конец")


# if __name__ == '__main__':
#     main()