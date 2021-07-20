
import os
import cv2
import shutil
import numpy as np
if __name__ == '__main__':
    root_path = './datasets'
    files = os.listdir(os.path.join(root_path, 'trash_images'))
    for name in files:
        gar_class = name.split('_')[0]
        if not os.path.exists(os.path.join(root_path, 'garbage_databases', gar_class)):
            os.makedirs(os.path.join(root_path, 'garbage_databases', gar_class))
        images = os.listdir(os.path.join(root_path, 'trash_images', name))
        for image in images:
            image_path = os.path.join(root_path, 'trash_images', name, image)
            if image.endswith('.jpeg'):
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                cv2.imencode('.jpeg', img)[1].tofile(os.path.join(root_path, 'trash_images', name, image))
            else:
                img = cv2.imread(image_path)
                cv2.imwrite(os.path.join(root_path, 'trash_images', name, image), img)

