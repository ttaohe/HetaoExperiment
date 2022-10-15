import os
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 
            'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 
            'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
            'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']

if __name__ == '__main__':
    f = open('labels.txt', 'w+')
    for class_name in classes:
        file_path = os.path.join('dataset', class_name)
        images = os.listdir(file_path)
        for image in images:
            f.write('%s %s\n'%(os.path.join(file_path,image),classes.index(class_name)))

        
            