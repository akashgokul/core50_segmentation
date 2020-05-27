import numpy
import PIL.Image
import PIL
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type = str, default = '/home/akash/core50/data/core50_128x128/', help='Directory of scenes')
args = parser.parse_args()

def save_seg(img_path):
    # Creating a list of all the images to process
    # img_path = '/home/akash/core50/data/core50_128x128/'
    root_path = '/home/akash/core50/data/core50_128x128/'
    images_to_process = []
    for (root, dirname, filenames) in os.walk(img_path):
        for file in filenames:
            if(file not in ['labels.pkl',  'LUP.pkl',  'paths.pkl']):
                relative_path = os.path.relpath(root + '/' + file)[30:]
                if('seg' not in relative_path):
                    #Handles old seg data and avoids saving as img
                    images_to_process.append(relative_path)
    # images_to_process = [f for f in os.listdir(img_path) if os.path.isfile(f) and f not in ['labels.pkl',  'LUP.pkl',  'paths.pkl']]
    print(len(images_to_process))
    print(images_to_process[0])

    # SET THESE VALUES BEFORE STARTING
    background_depth = 218
    svm_threshold = 0.5 # one class = -2.2
    dilated = 1

    svm_classes = 2

    dilated_image_lst = []
    missing_ct = 0
    for image in tqdm(sorted(images_to_process)):
        print('Start processing '+image)
        image_path = root_path  + image
        image_d = image.replace('C','D')
        depth_image_path = '/home/akash/core50/data/home/martin/core50_128x128_DepthMap/' + image_d
        print("DEPTH PATH:")
        print(depth_image_path)
        if(not os.path.exists(depth_image_path)):
            missing_ct += 1
            continue
        depth_image = numpy.asarray(PIL.Image.open(depth_image_path).convert('LA'))
        depth_image.setflags(write=1)
        rgb_image = PIL.Image.open(image_path)

        # use this boolean to decide wheater or not delete the holding hand
        hand_segmentation = True

        # qui devo crearmi una immagine di 1
        binary_image = numpy.zeros([128, 128], dtype=numpy.uint8)
        binary_image.fill(1)

        # deleting pixels belonging to the background
        for i in range(len(depth_image)):
            for j in range(len(depth_image[i])):
                if (depth_image[i][j][0]) < background_depth:
                    depth_image[i][j][0] = 0
                if (depth_image[i][j][1]) == 0:
                    depth_image[i][j][1] = 255

        img = PIL.Image.fromarray(depth_image)
        img = img.convert('RGB')


        # Taking the respective RGB image and deleting the background
        pixels = rgb_image.load()
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                if r == 0 & g == 0 & b == 0:
                    pixels[x, y] = (0, 0, 0)
                    #new_image[x,y] = 0
                else:
                    binary_image[x, y] = 0
        # rgb_image.save('temp'+image)



        # Using pre-trained SVM model for detecting pixels belonging to the hand
        if hand_segmentation:

            import pickle as cPickle

            if svm_classes == 1:
                with open('/home/akash/core50/core50_segmentation/trained_SVM.pkl', 'rb') as fid:
                    svm = cPickle.load(fid)
                    print('SVM one class loaded correctly')
            elif svm_classes == 2:
                with open('/home/akash/core50/core50_segmentation/trained_two_classes_SVM.pkl', 'rb') as fid:
                    svm = cPickle.load(fid)
                    print('SVC loaded correctly')


            list_SVM = []
            for x in range(width):
                for y in range(height):
                    # print(str(x) + ' ' + str(y))
                    rgb_list = []
                    r, g, b = rgb_image.getpixel((x, y))
                    rgb_list.append(r)
                    rgb_list.append(g)
                    rgb_list.append(b)
                    list_SVM.append(rgb_list)

            print('starting SVM prediction for '+image)
            pred = svm.decision_function(list_SVM)


            i = 0
            pixels = rgb_image.load()
            for x in range(width):
                for y in range(height):
                    if pred[i] >= svm_threshold:
                        #print('this pixel belongs to the hand')
                        pixels[x, y] = (0, 0, 0)
                        binary_image[x, y] = 1
                    i = i + 1

            # rgb_image.save('results/' + image)

            from skimage import morphology

            # l'immagine risultante la passo direttamente alla morfologia
            # plt.imsave('predilation/' + image, binary_image, cmap=cm.gray)


            # Applying morphological operators
            dilated_image = morphology.binary_dilation(binary_image, morphology.diamond(dilated)).astype(numpy.uint8)
            dilated_image_lst.append(dilated_image)
            # print("PATH: " + image_path[:-4] + '_seg.png')
            # plt.imsave(image_path[:-4] + '_seg.png',dilated_image, cmap=cm.gray)

           # applying dilation to the rgb
            i = 0
            pixels = rgb_image.load()
            for x in range(width):
                for y in range(height):
                    if dilated_image[x,y] == 1:
                        # print('this pixel belongs to the hand')
                        pixels[x, y] = (0, 0, 0)
                    i = i + 1

            # # rgb_image.save('results/' + image + '__dilated.png')

            # # saving the segmented image

            i = 0
            pixels = rgb_image.load()
            for x in range(width):
                for y in range(height):
                    r, g, b = img.getpixel((x, y))
                    ciccio = pixels[x, y]
                    if pixels[x, y] != (0,0,0):
                        # se il colore non è nero devo fare un leggero overlay
                        r, g, b = img.getpixel((x, y))
                        pixels[x,y] = (int(r/2), g, int(b/2))
                    i = i + 1
            
            print("Saved PATH: " + image_path[:-4] + '_seg.png')
            seg = rgb_image.convert("L")
            seg.save(image_path[:-4] + '_seg.png')
            print("--------\n")
            #plt.imsave(image_path[:-4] + '_newseg.png',seg, cmap=cm.gray)
            # rgb_image.save('colors/' + image + '__dilated.png')
    print("------------")
    print("Skipped Files: " + str(missing_ct))


if __name__ == "__main__":
    save_seg(args.img_path)









