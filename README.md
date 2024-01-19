# Photobomber Eraser

## Motivation and Goal
Dealing with passersby in photos is a common and unavoidable issue. There are times when it's challenging to capture moments without any obstruction from people passing by. The motivation occured, to implement a solution that can eliminate photobombers from photos and restore them, resulting in clean and authentic images.

## Method
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/2505194c-e3c1-433c-94ce-9b05bc73548a)

- Human Mask Extraction 
  - Use PSPNet to extract the pixels without class “human”.
- Feature Detection
  - SIFT
- Feature matching
  - KNN
- Homography Matrix Calculation
  - The coordinate translations based on the feature pairs.
  - Applied RANSAC to mitigate the impact of wrong pairs.
- Image Warping
  - Multiply coordinates by the homography matrix.
- Image Blending
  - Linear blending
  - Reference the other image to fill up the occluded areas.
 
## Result
* Images with different angles
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/fdd6dffb-e625-4ef2-98b9-e60fe5df6cae)

* Night Scene
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/bd0d487d-cadd-4ea7-b766-1e15d45cd937)

* Image with larger angle differences
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/7f5fcbc4-1d4a-4946-9f70-dabecdb03524)


## How to Run
1. Download the pretrained weights of PSPNet
2. Modify the paths in model.py
3. Run `python model.py`


[Slide Link](https://docs.google.com/presentation/d/1_xSNHx6IEdSV8wERCFffddFSg87rtolcRObpJMDgy5Q/edit?usp=sharing)

Reference:
[Image Stitching](https://yungyung7654321.medium.com/python%E5%AF%A6%E4%BD%9C%E8%87%AA%E5%8B%95%E5%85%A8%E6%99%AF%E5%9C%96%E6%8B%BC%E6%8E%A5-automatic-panoramic-image-stitching-28629c912b5a), 
[PSPNet](https://github.com/hszhao/semseg)
