// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved








namespace fsdet {


extern int get_cudart_version();


std::string get_cuda_version() {

  std::ostringstream oss;

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();

  return std::string("not available");

}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;


  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }




  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }



  { ss << "MSVC " << _MSC_FULL_VER; }

  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");

  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");

  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def(
      "deform_conv_backward_input",
      &deform_conv_backward_input,
      "deform_conv_backward_input");
  m.def(
      "deform_conv_backward_filter",
      &deform_conv_backward_filter,
      "deform_conv_backward_filter");
  m.def(
      "modulated_deform_conv_forward",
      &modulated_deform_conv_forward,
      "modulated_deform_conv_forward");
  m.def(
      "modulated_deform_conv_backward",
      &modulated_deform_conv_backward,
      "modulated_deform_conv_backward");

  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes");

  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");

  m.def(
      "roi_align_rotated_forward",
      &ROIAlignRotated_forward,
      "Forward pass for Rotated ROI-Align Operator");
  m.def(
      "roi_align_rotated_backward",
      &ROIAlignRotated_backward,
      "Backward pass for Rotated ROI-Align Operator");
}

} // namespace fsdet
