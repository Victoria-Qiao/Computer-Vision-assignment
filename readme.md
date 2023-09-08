The coursework was about doing image stitching from multiple images.
![cycwarp_resize_480_320](https://github.com/Victoria-Qiao/Computer-Vision-assignment/assets/135839176/23ab9479-402f-436d-8c33-84d443d4afaf)

![cycwarp_resizing_default_size](https://github.com/Victoria-Qiao/Computer-Vision-assignment/assets/135839176/57177443-7bf1-4557-89e3-bb3e820c675a)

To run the program: 

```
python stitch.py files1.txt(files2.txt or files3.txt) 1 1 480 320 
```

For example, 
```
python stitch.py files1.txt 1 1 480 320
```
The first 1 is whether to use cylindrical warping it can also be 0. 

The second 1 is whether to resize images. For faster computation, use resizing to change to smaller image size. 

If there are no options for the cylindrical warping and resizing options or the two options are inputing wrongly except 1 or 0, the default is to use cylindrical warping and not resize.

480 represents the resized height and 320 represents the resized width for input images if resize is chosen


If the path is not exist, it automatically stitches images in files1.txt

The filename is files1.txt in default if there is no input or the input is wrong

files1.txt is used to save the paths for input images

The input order should be images taken from left to right
