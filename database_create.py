## Convertir les images jpg en png
#!/usr/bin/python
from PIL import Image
import os, sys
import re


## Classes
classes = ['stop', 'speedlimit']

path=r'C:/Users/lcsgl/pt4/dataset_raw/'
dirs = os.listdir( path )
final_size = 244;

def resize_aspect_fit():
    for cl in classes:
        i = 0
        cpath = path+classes
        for item in dirs:
             if os.path.isfile(cpath+item):
                 im = Image.open(cpath+item)
                 f, e = os.path.splitext(cpath+item)
                 size = im.size
                 ratio = float(final_size) / max(size)
                 new_image_size = tuple([int(x*ratio) for x in size])
                 im = im.resize(new_image_size, Image.ANTIALIAS)
                 new_im = Image.new("RGBA", (final_size, final_size))
                 new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
                 new_im.save(f + '_.png', 'png', quality=90)
                 print("Conversion", ((i/len(dirs))*100), "%")
                 i += 1


## supprimer tout les fichiers autre que png
def clean_folder():
    i = 0
    for item in dirs:
        if os.path.isfile(path+item):
            if re.match(".*\_.png$", item) == None:
                os.remove(path+item)
            print("Nettoyage", ((i/len(dirs))*100), "%")
            i += 1


if __name__ == "__main__":
    resize_aspect_fit()      
    clean_folder()
    print("Travail terminé.")