# Распознавание лиц и их идентификация
Выполнена с помощью метода обнаружения и выравнивания лиц (MTCNN) и сиамской нейросети FaceNet.\
Используется Python 3.7.
## Библиотеки
1.	Numpy
2.	Tensorflow 
3.	Sklearn
4.	Pillow
5.	Keras
6.	MTCNN
## Начало работы
1.	Убедитесь, что используется Python версии 3.7.
2.	Установите необходимые библиотеки. (Предпочтительна среда Conda)
3.	Скачайте файл “facenet_keras.h5” и поместите его в папку проекта.
## Создание базы данных
1.	Поместите в папку “Folder_faces” папки с названиями имен людей, хранящую их фотографии соответственно.
2.	(При необходимости) Запустите файл “rename_faces.py” для переименования названий фотографий, если в них имеются пробелы или символы, которые не дадут их прочитать программе правильно. 
3.	Запустите файл “my_mtcnn.py”, который найдет все лица в папке “Folder_faces”, соединит их с именами и поместит данные в файл “faces-dataset.npz” в папке “Npz_files”. Займет некоторое время.
База данных создана. Файлы папки “Folder_faces” больше не потребуются.
## Обработка данных
1.	Запустить “processing.py”, в процессе которого создастся файл “faces-embeddings.npz” в папке “Npz_files”. Займет некоторое время.
## Запуск
1.	Поместить фотографии людей, которых необходимо идентифицировать в папку “My_faces”
2.	Запустить “main.py”.
3.	Фотография с именами людей будут загружены в папку “Name_my_faces”.
## Работа программы
Загруженная фотография\
![oop_1](https://user-images.githubusercontent.com/62021182/229915347-694dce25-0655-457b-aa69-62e3a9609bf6.png)

Сохраненный результат\
![oop_1](https://user-images.githubusercontent.com/62021182/229915450-8a86a4d3-7f7f-41d2-be46-289dcad5d0c4.png)