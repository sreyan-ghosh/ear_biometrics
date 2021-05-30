string = "A point and shoot camera does just what the name implies - the photographer selects an automatic o semiautomatic mode from the available presets, points the camera at the subject and presses the shutter button to create an image."
l = list(string)
count = 0
for i in l:
    if i == 'o':
        count += 1
print(count)
