## Landslide susceptibility mapping (LSM)


### Data Description
- Influencing factors and labels are both in TIF format, and the width and height of the dimensions are the same
- Influencing factor is named as **xxx_5.tif** (5 is the number of reclassifications, which can be modified to the value of your own data)
- The label naming should include **label** (for easy automatic recognition by the program) See the image below
- The label needs to be made into a point with four types of 0, 1, 2, and 3, and specifically refers to the project example_label.tif


![image](https://user-images.githubusercontent.com/57258378/225853069-a1f1eefe-32d1-46ea-a1ea-13ae98c75581.png)

### Execution Description
- Clone the project into the same directory as the data folder (influencing factors and labels are both stored in the data folder)
- Modify the first parameter in config to the name of your own data folder. Detailed comments are provided for the remaining parameters.
- Run **start.py**


### Required Library Installation
- GDAL 2.4.1


