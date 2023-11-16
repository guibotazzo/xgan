# X-GAN: Generative Adversarial Networks Training Guided with Explainable Artificial Intelligence

## 1 - Status de implementação

✅ - Implementado e funcionando <br>
❌ - Não implementado ou não funciona ainda <br>
🚫 - Modelo não é compatível com o dataset

### Modelos originais

|         | MNIST | FMNIST | CelebA | NHL256 |
|---------|:-----:|:------:|:------:|:------:|
| DCGAN   |   ✅   |   ✅    |   ✅    |   ❌    |
| WGAN-GP |   ✅   |   ✅    |   ✅    |   ❌    |

### Modelos novos

|          | MNIST | FMNIST | CelebA | NHL256 |
|----------|:-----:|:------:|:------:|:------:|
| XDCGAN   |   ✅   |   ✅    |   ✅    |   ❌    |
| XWGAN-GP |   ✅   |   ✅    |   ✅    |   ❌    |

## 2 - Treinamento não supervisionado

Para treinar os models GAN maneira não-supervisionada executar:

python train.py [PARAMETROS]

**Exemplo:**

Treinar a WGAN-GP no dataset MNIST com dimesões 32x32x3 por 50 epochs

```
python train.py --gan WGAN-GP --dataset MNIST --image_size 32 --channels 1 --epochs 50
```
Parametros padrão:

| Parameters | Values  |
|------------|---------|
| Model      | DCGAN   |
| Dataset    | CIFAR10 |
| Image size | 32      |
| Channels   | 3       |
| Batch size | 32      |
| Epochs     | 100     |



## 3 - Treinamento supervisionado para fins de aumento artificial e classificação

Para treinar os modelos GAN com o propósito de aumento artificial é necessário treinar um modelo para cada classe do dataset.

Para isso deve-se modificar o diretório do dataset para ser possível carregar uma classe de cada vez no Dataloader.

A organização padrão dos diretórios é, por exemplo:

```
xgan
└── datasets
    └── NHL256
        └── CLL
        │    │  CLL (1).png
        │    │  ...
        └── FL
        │    │  FL (1).png
        │    │  ...
        └── MCL
             │  MCL (1).png
             │  ...
```

E no arquivo lib/datasets.py o root é por padrão:

```
dataset = ImageFolder(root='./datasets/NHL256/',
                      transform=Compose([
                            Resize(img_size),
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
```
Para realizar o treinamento "supevisionado", a estrutura do diretório deve ser:
```
xgan
└── datasets
    └── NHL256
        └── CLL
        │    └── one_class
        │    │      CLL (1).png
        │    │      ...
        └── FL
        │    └── one_class
        │    │      FL (1).png
        │    │      ...
        └── MCL
             └── one_class
             │      MCL (1).png
             │      ...
```

E deve-se modificar o root para a classe desejada, por exemplo:

```
dataset = ImageFolder(root='./datasets/NHL256/CLL/',
                      transform=Compose([
                            Resize(img_size),
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
```