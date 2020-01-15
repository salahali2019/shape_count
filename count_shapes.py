import numpy as np
import numpy as np 
import sys
sys.setrecursionlimit(150000)


def read_image(binary_image, image_w, image_h):
  data = open(binary_image, "rb").read()
  mutable_bytes=bytearray(data)
  image_size=len(mutable_bytes)

  image=[]
  for i in range(image_size):
    image.append(mutable_bytes[i])
  image=np.array(image).reshape(image_w,image_h)
  return image

def reduce_image_size(arr_in,kernel):
  from numpy.lib.stride_tricks import as_strided

  block_shape = np.array(kernel)

  arr_shape = np.array(arr_in.shape)
  new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
  new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

  arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

  reduced_image=np.zeros([arr_out.shape[0],arr_out.shape[1]])

  for i in range(arr_out.shape[0]):
    for j in range(arr_out.shape[1]):
      reduced_image[i,j]=np.min(arr_out[i,j])
  return reduced_image

  
def isValid(Image,visited, row, col, c, h, w) : 
                                          
 
    return ((row >= 0 and row < h) and 
            (col >= 0 and col < w) and 
            (Image[row][col] == c and not 
             visited[row][col]));  
  
def DFS(Image,visited, row, col, c, h, w) :  
  
    
    row_move = [ -1, 1, 0, 0 ];  
    col_move = [ 0, 0, 1, -1 ];  
  
    # Update as visited  
    visited[row][col] = True;  
  
    # Recur for all connected neighbours  
    for k in range(4) : 
        if (isValid(Image, visited,row + row_move[k],  
                   col + col_move[k], c, h, w)) : 
  
            DFS(Image,visited, row + row_move[k],  
                col + col_move[k], c,h,w);  
  
 
def count_areas(Image,visited,Q, h) : 
  
    connectedComp = 1;  
    w = len(Image[0]);  
  
    for i in range(h) : 
        for j in range(w) : 
            if (not visited[i][j]) :  
                c = Image[i][j];  
                DFS(Image,visited, i, j, c, h,w); 
                Q[i,j]= connectedComp

                connectedComp += 1;  

          
    return connectedComp,Q;    

def output(image,segmented, contours):
  A=np.zeros([256])
  for i in range(1,contours):
    if ( A[int(image[np.where(segmented==i)])]==0):
         A[int(image[np.where(segmented==i)])]=1
    else:
         A[int(image[np.where(segmented==i)])]+=1
  return A

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='passing')



    parser.add_argument('--file_name', required=False,
                        metavar="image name",
                        help='Binary image')
    
    parser.add_argument('--shape_h', type=int,required=False,)
    parser.add_argument('--shape_w', type=int,required=False,)

    
    args = parser.parse_args()      

    image=read_image(args.file_name, args.shape_h, args.shape_w)
    image=np.array(image)

    image=reduce_image_size(image,(2,2))

    Q=np.zeros_like(image)
    visited = np.zeros_like(image)

    n=len(image)
    contours, segmented=connectedComponents(image,visited,Q, n)
    output=output(image,segmented, contours)
    print(output)



