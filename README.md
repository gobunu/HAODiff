# Human Body Restoration with One-Step Diffusion Model and A New Benchmark

[Jue Gong](https://github.com/gobunu), [Jingkai Wang](https://github.com/jkwang28), [Zheng Chen](https://zhengchen1999.github.io/), Xing Liu, Hong Gu, [Yulun Zhang](http://yulunzhang.com/), and [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ), "A new benchmark and the first one-step diffusion model for human body restoration.", 2025

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoSvg)](
https://arxiv.org/abs/2502.01411)
[![supp](https://img.shields.io/badge/Supplementary_material-Paper-orange.svg)](https://github.com/gobunu/OSDHuman/releases/download/v1/supp.pdf)
[![releases](https://img.shields.io/github/downloads/gobunu/OSDHuman/total.svg)](https://github.com/gobunu/OSDHuman/releases)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=gobunu.OSDHuman&right_color=violet)](https://github.com/gobunu/OSDHuman)
[![GitHub Stars](https://img.shields.io/github/stars/gobunu/OSDHuman?style=social)](https://github.com/gobunu/OSDHuman)

#### 🔥🔥🔥 News

- **2025-2-5:** This repo is released.

---

> **Abstract:** Human body restoration, as a specific application of image restoration, is widely applied in practice and plays a vital role across diverse fields. However, thorough research remains difficult, particularly due to the lack of benchmark datasets. In this study, we propose a high-quality dataset automated cropping and filtering (HQ-ACF) pipeline. This pipeline leverages existing object detection datasets and other unlabeled images to automatically crop and filter high-quality human images. Using this pipeline, we constructed a person-based restoration with sophisticated objects and natural activities (PERSONA) dataset, which includes training, validation, and test sets. The dataset significantly surpasses other human-related datasets in both quality and content richness. Finally, we propose OSDHuman, a novel one-step diffusion model for human body restoration. Specifically, we propose a high-fidelity image embedder (HFIE) as the prompt generator to better guide the model with low-quality human image information, effectively avoiding misleading prompts. Experimental results show that OSDHuman outperforms existing methods in both visual quality and quantitative metrics. 

![](images/OSDHuman_overall.png)

---

[<img src="images/test-1.png" height="200"/>](https://imgsli.com/MzQ1ODk0) [<img src="images/test-2.png" height="200"/>](https://imgsli.com/MzQ1ODky)[<img src="images/Val-1.png" height="200"/>](https://imgsli.com/MzQ1ODk3) [<img src="images/Val-2.png" height="200"/>](https://imgsli.com/MzQ1ODk4)

---

## ⚒️ TODO

* [ ] Release code and pretrained models
* [ ] Release PERSONA dataset

## 🔗 Contents

- [ ] PERSONA Dataset
- [ ] Models
- [ ] Testing
- [ ] Training
- [ ] [Results](#Results)
- [x] [Citation](#Citation)
- [ ] [Acknowledgements](#Acknowledgements)

## <a name="results"></a>🔎 Results

[TBD]

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@article{gong2025osdhuman,
    title={Human Body Restoration with One-Step Diffusion Model and A New Benchmark},
    author={Jue Gong and Jingkai Wang and Zheng Chen and Xing Liu and Hong Gu and Yulun Zhang and Xiaokang Yang},
    journal={arXiv preprint arXiv:2502.01411},
    year={2025}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

[TBD]

<!-- ![Visitor Count](https://profile-counter.glitch.me/gobunu/count.svg) -->