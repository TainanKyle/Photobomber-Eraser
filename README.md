# Photobomber Eraser

### Motivation and Goal
Taking photos is an essential part of modern life; however, dealing with passersby during photography is a common and unavoidable issue. There are times when it's challenging to capture moments without any obstruction from people passing by. Occasionally, we may only realize the interference of bystanders when reviewing the photos at home. Despite taking multiple shots of the same scene, it can be challenging to obtain a clean image without unwanted elements. Therefore, we were motivated to implement a solution that can eliminate photobombers from photos and restore them, resulting in clean and authentic images.

### Proposed Method
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
 
### Result
Images with different angles
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/fdd6dffb-e625-4ef2-98b9-e60fe5df6cae)

Night Scene
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/bd0d487d-cadd-4ea7-b766-1e15d45cd937)

Image with larger angle differences
![image](https://github.com/TainanKyle/Photobumber-Eraser/assets/150419874/7f5fcbc4-1d4a-4946-9f70-dabecdb03524)

[Slide Link] (https://docs.google.com/presentation/d/1_xSNHx6IEdSV8wERCFffddFSg87rtolcRObpJMDgy5Q/edit?usp=sharing)

Reference:
[Image Stitching](https://yungyung7654321.medium.com/python%E5%AF%A6%E4%BD%9C%E8%87%AA%E5%8B%95%E5%85%A8%E6%99%AF%E5%9C%96%E6%8B%BC%E6%8E%A5-automatic-panoramic-image-stitching-28629c912b5a)
[PSPNet](https://github.com/hszhao/semseg)
