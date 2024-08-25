# Introduction

Cropping and adding watermarks to pictures could be really repetitive, tedious, and time consuming, especially when there are thousands of pictures to be edited. On the other hand, computers are well known to be faster, more efficient and accurate (no human errors) in running repetitive tasks. 

So instead of having to repetitively load each image to the photo editing app, crop & resize them to the right dimensions, save them to the right formats & directory location on the computer, why not have a computer automatically do it all in bulk for us?

Therefore, I developed a Bulk Image Cropper & Watermark Generator using Python. Implemented Computer Vision and Image Processing techniques to achieve the desired outputs.

Initially, this program is designed to edit pictures of multiple Real Estate listings simultaneously, with each listing's pictures separated by their own respective folders, both as an input and output. However, this program could be implemented in other businesses/industries too.

# Tutorial

1. Install all required Python libraries; PIL, CV2, and Numpy
2. Move the watermark icon (PNG image) to the main directory
3. Move all pictures to be edited to the 'images_raw' folder:
- If there are groups of pictures, separate them by their own respective folders within the directory. 
- Even if there are no groups of pictures, please also place them all in a single folder within the directory.
- Folder/s names could be anything
4. Make sure that all directory/path variables are correct (first cell of code)
5. There are 2 editing options for the program, which could be chosen depending on the needs:
- Square Crop, Blurred Border, & Watermark Images
- Square Crop, & Watermark Images
- Please refer to the 'examples' folder for each outcome example
- To chose an option, simply comment all the codes under the not-selected option
6. Run the 'main.ipynb' file. 
7. Once completed, the edited pictures will be present in the 'images_edited' folder, separated by their
own respective folders for each group within the directory.

Note: To avoid confusion, please look at the given example output folders & pictures in the 'images_edited' 
folder, and compare them with the original inputs in the 'images_raw' folder
