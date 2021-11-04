1. This Model is the implementation of the Mask RCNN using torch as the framework
Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.

2. The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range. Different images can have different sizes.

3. The behavior of the model changes depending if it is in training or evaluation mode.

**TRAINGING PROCESS**
the model expects both the input tensors, as well as a targets (list of dictionary), containing:

```boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

labels (Int64Tensor[N]): the class label for each ground-truth box

masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
```
The model returns a ```Dict[Tensor]``` during training, containing the classification and regression losses for both the RPN and the R-CNN, and the mask loss.
-> RPN과 R-CNN모델에서의 classification, regression 손실 값과 masking한 결과에 대한 손실 값을 반환한다.


During inference, the model requires only the input tensors, and returns the post-processed predictions as a ```List[Dict[Tensor]]```, one for each input image. The fields of the Dict are as follows, where N is the number of detected instances:

boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

labels (Int64Tensor[N]): the predicted labels for each instance

scores (Tensor[N]): the scores or each instance

masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)
-> 만약에 뒤이어서 classifier을 이어 붙여서 cell 3종류와 배경을 구분하지 않는 다면 mask R-CNN만을 사용하는 경우에는 그냥 threshold를 0.5로 두고 mask output으로 나온 binary data 에 대해서 threshold를 넘는지 넘지않는지의 여부에 맞추어서 0과 1의 형태로 바꾸어 준다.

- 저번 MOAI대회에서도 중요했던 부분은 학습시키는 과정에서 사용하는 손실 함수 function(IOU score제대로 사용하기) 그리고 올바른 dataset이다.