// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved



namespace fsdet {

at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);


at::Tensor nms_rotated_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);


// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {

    return nms_rotated_cuda(dets, scores, iou_threshold);

    AT_ERROR("Not compiled with GPU support");

  }

  return nms_rotated_cpu(dets, scores, iou_threshold);
}

} // namespace fsdet
