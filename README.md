# XGAN: Generative Adversarial Networks Training Guided with XAI-based models

Training Generative Adversarial Networks (GANs) aims to achieve a balance that is often challenging because the discriminator (D) typically outperforms the generator (G), as only D accesses the image features. To address this, we introduce a novel approach using methods based on Explainable Artificial Intelligence (XAI) to provide G with crucial information during training. We enhance the learning process by identifying key features learned by D and transferring this knowledge to G. Our modified loss function uses a matrix of XAI explanations instead of just a single error value, resulting in improved quality and greater variability in the generated images.

## References
If you use this method on your research, please cite:

[1.](https://doi.org/10.3390/app14188125) Rozendo, G. B., Garcia, B. L. D. O., Borgue, V. A. T., Lumini, A., Tosta, T. A. A., Nascimento, M. Z. D., & Neves, L. A. (2024). Data Augmentation in Histopathological Classification: An Analysis Exploring GANs with XAI and Vision Transformers. Applied Sciences, 14(18), 8125.

[2.](https://doi.org/10.5220/0012618400003690) Rozendo, G., Lumini, A., Roberto, G., Tosta, T., Zanchetta do Nascimento, M. and Neves, L. (2024). X-GAN: Generative Adversarial Networks Training Guided with Explainable Artificial Intelligence. In Proceedings of the 26th International Conference on Enterprise Information Systems - Volume 1: ICEIS; ISBN 978-989-758-692-7; ISSN 2184-4992, SciTePress, pages 674-681.
