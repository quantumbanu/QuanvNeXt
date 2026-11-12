# QuanvNeXt

This repo contains the PyTorch implementation of **QuanvNeXt**, a lightweight quantum–classical hybrid architecture designed for time series analysis.

All trainable layers in QuanvNeXt are built using our **Quanv1D** layer, previously introduced at **ACM SIGKDD 2025 (KDD '25)**. Classical computations handle normalization, activation functions, and optimization.

No prior knowledge of quantum computing is required to use QuanvNeXt. All quantum-related operations are handled internally by the Quanv1D layers and simulated entirely in PyTorch, without requiring quantum hardware or external quantum computing frameworks.

## Paper

Our work on QuanvNeXt is currently under review. An early preprint is available on arXiv:

[Read the preprint on arXiv](https://arxiv.org/abs/2512.09517)

## Citation

If you find QuanvNeXt useful in your research, please consider citing our QuanvNeXt preprint (for now), along with the original Quanv1D paper published at KDD '25.

```bibtex
@article{orka2025quanvnext,
  title={QuanvNeXt: An end-to-end quanvolutional neural network for EEG-based detection of major depressive disorder},
  author={Orka, Nabil Anan and Haque, Ehtashamul and Jannat, Maftahul and Awal, Md Abdul and Moni, Mohammad Ali},
  journal={arXiv preprint arXiv:2512.09517},
  year={2025}
}
```

```bibtex
@inproceedings{10.1145/3711896.3736972,
  author = {Orka, Nabil Anan and Haque, Ehtashamul and Awal, Md. Abdul and Moni, Mohammad Ali},
  title = {Fully Quanvolutional Networks for Time Series Classification},
  year = {2025},
  isbn = {9798400714542},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3711896.3736972},
  doi = {10.1145/3711896.3736972},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages = {2210–2221},
  numpages = {12},
  location = {Toronto, ON, Canada},
  series = {KDD '25}
}
```
