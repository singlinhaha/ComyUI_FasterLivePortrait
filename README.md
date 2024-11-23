# Comfyui-FasterLivePortrait
本仓库是对FasterLivePortrai的comfyui实现
**原仓库: [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait/tree/master)，感谢作者的分享**
**功能：**
* 支持将LivePortrait模型转为Onnx模型，使用onnxruntime-gpu运行
* 支持将LivePortrait模型转为TensorRT模型，使用TensorRT运行
* 支持Human模型、Animal模型
* 支持人脸选定、多人模式
## 环境安装
 环境依赖：
```bash
onnxruntime-gpu
tensorrt

# xpose安装
cd src/models/XPose/models/UniPose/ops
python setup.py build install
```
onnxruntime和tensorrt的版本取决电脑的cuda和cudnn版本。
官方项目中动物模型使用的XPose作为特征点检测器，更多详细信息请参考[XPose](https://github.com/IDEA-Research/X-Pose)

## 模型转换
下载FasterLivePortrait仓库作者转换好的[模型onnx文件](https://huggingface.co/warmshao/FasterLivePortrait): `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`。
##### Onnxruntime 推理
* 使用onnxruntime cpu推理的话，直接`pip install onnxruntime`即可，但是cpu推理超级慢。
* 由于最新的onnxruntime-gpu仍然无法支持grid_sample cuda，需要切换到一位大佬的分支进行重新编译，按照以下步骤源码安装`onnxruntime-gpu`:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks for liqun's grid_sample with cuda implementation!
  * 运行以下命令编译,`cuda_version`和`CMAKE_CUDA_ARCHITECTURES`根据自己的机器更改:
  ```shell
  ./build.sh --parallel \
  --build_shared_lib --use_cuda \
  --cuda_version 11.8 \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/ \
  --config Release --build_wheel --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="60;70;75;80;86" \
  --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  --disable_contrib_ops \
  --allow_running_as_root
  ```
  * `pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl`就可以了
##### TensorRT 推理
* 安装TensorRT，请记住[TensorRT](https://developer.nvidia.com/tensorrt)安装的路径。
* 安装 grid_sample的tensorrt插件，因为模型用到的grid sample需要有5d的输入,原生的grid_sample 算子不支持。
  * `git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin`
  * 修改`CMakeLists.txt`中第30行为:`set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")`
  * `export PATH=/usr/local/cuda/bin:$PATH`
  * `mkdir build && cd build`
  * `cmake .. -DTensorRT_ROOT=$TENSORRT_HOME`,$TENSORRT_HOME 替换成你自己TensorRT的根目录。
  * `make`，记住so文件的地址，将`scripts/onnx2trt.py`和`src/models/predictor.py`里`/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so`替换成自己的so路径
* 将onnx模型转为tensorrt，运行`sh scripts/all_onnx2trt.sh`和`sh scripts/all_onnx2trt_animal.sh`
## 参数
* driving_type：驱动类型，可选为["human", "animal"]
* batch_mode: 是否开启多人模式
* face_index: 选中人脸编号
* max number：允许同时驱动的最大数量
* config：推理后端选择
注：由于xpose每次只能返回一个结果，所以多人模式对于动物模型不生效
