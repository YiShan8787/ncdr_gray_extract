# ncdr_gray_extract
 extract the front gray image 

# How to use

 ## source code (recommend)
 ```
   1. modify the path in the .ini

   2. put the sample front image in the ./weather_image (default in .ini) folder

   3. use the .exe file
```

# Config
```
origin_path =  weather_image             # where you put the weather image files
destination_path =  extract_image        # where you hope to save the output images
suffix = _gray_extract                   # the suffix of the output images
image_type = jpg                         # the image type of output images (recommend jpg or png)

cut_x = 100                              # we need to crop the image to cut the redundant word
cut_y = 200                              # we need to crop the image to cut the redundant word

y_len = 1000                             # the size of the image after we crop
x_len = 600                              # the size of the image after we crop

# flipped from descending to ascending   
lonRange = [90,160]                      # the longtitude range of the cropped image 
latRange = [10,60]                       # the latitude range of the cropped image 
```

# Sample image

we provide some image [there!](https://drive.google.com/drive/folders/1o6EcvwDACSFmxB4M3C2vVPyyqjBxo0E-)

# Exe file

Due to the github file size limit, we provide the exe file [there!](https://drive.google.com/drive/folders/1_dSlG-W-W66lBJEQMgiVOhqV4VdVxS3J)
