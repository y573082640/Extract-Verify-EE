# PointerNet_Chinese_Information_Extraction
基于指针网络进行事件抽取的工作。效果略优于global pointer和序列标注方法。缺点是运行时间比二者长，因为模块更多。

主要思想：先基于传统指针网络方法识别事件。调低识别阈值，获得大量True Positive和False Positive，随后使用一个事件验证器将假论元剔除。

基于duee1.0数据集分别训练三个网络：
1. 多层双指针网络，识别不同事件类型的关键词。训练数据来源于duee。
2. 基于提示微调的双指针网络，识别事件论元。训练数据来源于duee。
3. 基于提示学习的论元验证，识别错误论元。训练数据从duee构建。
   
还有其他探索过的增强方法，不过效果一般：
1. 数据增强。通过论元替换等方式合成新数据进行训练。
2. 检索相似数据作为demo的方法。
   
# 主要参考
主要基于下面的开源仓库构建：
> [一种基于Prompt的通用信息抽取（UIE）框架_阿里技术的博客-CSDN博客](https://blog.csdn.net/AlibabaTech1024/article/details/127747678) （思想和大部分图片都来自这）
> [PointerNet_Chinese_Information_Extraction](https://github.com/taishan1994/PointerNet_Chinese_Information_Extraction)（事件抽取部分的基础代码）
