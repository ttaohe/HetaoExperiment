import os
def getFlist(path):
    root_dirs = []
    for root, dirs, files in os.walk(path):
        print('root_dir:', root)
        print('sub_dirs:', dirs)
        print('files:', files)
        root_dirs.append(root)
        print('root_dirs:', root_dirs[1:])
    root_dirs = root_dirs[1:]
    return root_dirs
def getChildList(root_dirs):
    j = 0
    f = open('dataset/cow_jpg.lst', 'w')#生成文件路径和类别索引
if __name__ == '__main__':
    resDir = 'dataset'
    f2 = open('dataset/object_list.txt', 'w')#生成类别和索引的对应表
    root_dirs = getFlist(resDir)
    k = 0
    for root_dir in root_dirs:
        f2.write('%s %s\n'%(root_dir,k))
        k = k+1
    f2.close()
    getChildList(root_dirs)
    print(root_dirs)