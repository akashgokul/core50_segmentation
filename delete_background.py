from matplotlib import pyplot as plt
from skimage import io, morphology
import numpy
import PIL
import math

from skimage.morphology import binary_dilation

# C_05_17_022
# List of images: C_02_01_207, C_02_42_003, C_03_01_004, C_03_22_182, C_03_47_177, C_05_17_022, C_07_36_005, C_09_45_031,
# C_10_12_007, C_11_22_030


name='C_11_22_030'
image_path = 'depth_images/'+name+'.png'
depth_name = list(name)
depth_name[0] ='D'
depth_name=''.join(depth_name)
depth_image_path = 'depth_images/'+str(depth_name)+'.png'
intermedio = 'intermedio.jpg'


# SET THESE VALUES BEFORE STARTING
background_depth = 200
svm_threshold = -2.2
dilated = 0

depth_image = numpy.asarray(PIL.Image.open(depth_image_path).convert('LA'))
depth_image.setflags(write=1)

rgb_image = PIL.Image.open(image_path)

hand_segmentation = True

# questo ciclo invece lavora solo sulla prima colonna dell'array 2D
# print(new_image.flags)
for i in range(len(depth_image)):
    for j in range(len(depth_image[i])):
        #dove la depth è < 226, la azzeriamo.
        if(depth_image[i][j][0]) < background_depth:
            depth_image[i][j][0] = 0
        #questa parte colora di nero i pixel dove manca il canale alpha
        if (depth_image[i][j][1]) == 0:
            depth_image[i][j][1] = 255
            #new_image[i][j][0] = 255


img = PIL.Image.fromarray(depth_image)
img = img.convert('RGB')
img.save("intermedio.png")


# Coloro i pixel del background della immagine RGB (128x128)
pixels=rgb_image.load()
width, height = img.size
for x in range(width):
    for y in range(height):
        r,g,b = img.getpixel((x, y))
        if r == 0 & g == 0 & b == 0:
            pixels[x, y] = (0, 0, 0)
        if r is None or g is None or b is None:
            print('pixel nullo')
            pixels[x, y] = (0, 0, 0)

rgb_image.save('risultato.png')

# applico SVM preaddestrato

if hand_segmentation:

    import pickle as cPickle

    with open('trained_SVM.pkl', 'rb') as fid:
        svm = cPickle.load(fid)
        print('SVM loaded correctly')

    list = []
    for x in range(width):
        for y in range(height):
            print(str(x) + ' ' + str(y))
            rgb_list = []
            r, g, b = rgb_image.getpixel((x, y))
            rgb_list.append(r)
            rgb_list.append(g)
            rgb_list.append(b)
            list.append(rgb_list)

    print('starting predictions..')
    #pred=svm.predict(list)
    pred = svm.decision_function(list)
    print('MIN VALUE')
    print(pred.min())
    i = 0
    pixels = rgb_image.load()
    for x in range(width):
        for y in range(height):
            if pred[i] >= svm_threshold:
                print('è la mano')
                pixels[x, y] = (0, 0, 0)
            else:
                print('non è la mano')
            i = i + 1

    # i = 0
    # pixels = rgb_image.load()
    # for x in range(width):
    #     for y in range(height):
    #         if pred[i] == 1:
    #             print('è la mano')
    #             pixels[x, y] = (0, 0, 0)
    #         else:
    #             print('non è la mano')
    #         i = i + 1

    rgb_image.save('results/FINE_'+name+'.png')


    from skimage import morphology

    # Convertire immagine in binaria
    gray = rgb_image.convert('L')
    binary_image = gray.point(lambda x: 0 if x > 128 else 1, '1')
    binary_image.save('predilation/'+name+'.png')

    print(binary_image.getpixel((127, 127)))

    # Applicare operatori morfologici
    pippo=numpy.array(binary_image)
    dilated_image = morphology.binary_dilation(numpy.array(binary_image), morphology.diamond(dilated)).astype(numpy.uint8)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.imsave('dilations/'+name+'.png', numpy.array(dilated_image).reshape(128, 128), cmap=cm.gray)


    print('done')




