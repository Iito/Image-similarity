# Introduction
As an amater photographer, I take a lot of pictures, often from the same subject with different exposure, iso etc...\
As such I end up with many photos to parse and sorts.\
So to help you out sorting this solution should give you the best of similar shots.

## Dependencies
* Python 3 >= 3.9
* NumPy >= 1.21.1
* OpenCV >= 4.5.3
* OpenCV >= 4.5.3
* RawPy >= 0.16.0
* tqdm >= 4.62.0
### CPU only
You can use the CPU only version quite easily by simply install the requirements defined in the requirements.txt, nothing more to do.
### CUDA (Unix, Windows, WSL2)
This is a little bit more tricky, as it is required to build the library from source depending on your environment and configuration (which GPU generation). \

#### Unix & WSL2*
Here is some explanations : https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7

#### Windows
I followed the following instructions to get mine working on Windows (not WSL, native windows):\
https://www.yotec.net/guide-how-to-install-opencv-cuda-on-windows/

#### WSL2 only:
According to NVIDIA's official documentation: You need to be registered on Windows Insider Program, subscribe to the Dev channel and have Windows 10 OS build is 20145. We recommend at least OS 21390 or higher for latest features and performance updates.

## How does it works:
The program will compare one by one the photos in the indicated directory.
Photos in the directory should have a similar context: taken on the same day or taken different days but same location. \
(If you try to compare the best photos between photos taken in different places expect odd results)
By comparing photos with each other the program will create group of similar photos, here in the example, image_1, image_2 and image_3 are "similar" they were taken the same day and discribe the same subject with a different angle or exposure, iso etc..\
image_4 for reference was taken another day at a total different location.\
The program will try to "match" using template matching (norm hamming) with all the others. 

### Step 1
Small lookup of what it will do:
- image_1 similar to image_2 ? yes
- image_1 similar to image_3 ? yes
- image_1 similar to image_4 ? no
- image_2 similar to image_3 ? yes
- image_2 similar to image_4 ? no
- image_3 similar to image_4 ? no

List of Sets = [ 
    group_1: { image_1, image_2, image_3 }
    group_2: { image_2, image_3 } 
]\
Since group_2 is contained in group_1 we can merge them.\
Final set would be :
group_1: {
        image_1,
        image_2,
        image_3
    }
### Step 2
From each group we will look for the one with less blur, for that we will use variance of Laplace formula.\
It will output a number for each image, the lower the number the blurier.\
Ideally the one with the highest number will be the best match from the group.

### Step 3 (Optional)
If you select the flag `--rename` the program will therefore move the 'rejected' photos into a folder named after the best match.