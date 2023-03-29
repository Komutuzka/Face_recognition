import pathlib

directory = pathlib.Path("Folders_faces\\")
for subdir in directory.iterdir():
    i = 0
    for path in subdir.glob('*.jpg'):
        new_name = str(i) + '.jpg'
        path.rename(subdir / new_name)
        print(f'Renamed `{path.name}` to `{new_name}`')
        i = i + 1
