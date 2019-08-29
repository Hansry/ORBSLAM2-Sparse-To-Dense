import os
import os.path
import glob




if __name__ == '__main__':
   timestamp = open('./RGB-D/kitti/timestamps.txt')
   rgb_set = glob.glob('./RGB-D/kitti/rgb/*.png')
   depth_set = glob.glob('./RGB-D/kitti/depth/*.png')
   rgb_set = sorted(rgb_set)
   depth_set = sorted(depth_set)
   lines = timestamp.readlines()
   count = 0
   path_set = []
   for i in lines:
       ps = i.split(' ')
       date = ps[1].split('\n')
       print()
       if count<len(rgb_set):
          rgb = rgb_set[count]
          depth = depth_set[count] 
          write_line = str(date[0])+' '+str(rgb)+' '+str(depth)+'\n'
          path_set.append(write_line)
       count = count + 1
   
   with open('./kitti.txt','w') as f:
       for i in path_set:
           f.write(i)
        
