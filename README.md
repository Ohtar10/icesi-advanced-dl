# [Universidad Icesi] Deep Learning Avanzado - Material del Curso

Este repositorio contiene el material de apoyo técnico para el curso de Aprendizaje Profundo Avanzado de la Maestria de Inteligencia Artificial Aplicada de la Universidad Icesi, Cali - Colombia.

## Estructura del repositorio
```bash
icesi-advanced-dl/
├── Unidad 1 - Time Series
├── Unidad 2 - Graph Neural Networks
├── Unidad 3 - Transformers
├── environment.yaml
├── requirements.txt
├── assets
├── datasets
├── LICENSE
└── README.md
```
Este repositorio esta diseñado para servir como referencia funcional de las lecciones del curso. Cada unidad del curso tiene su propio directorio con notebooks de Jupyter con el material técnico visto en las lecciones y de donde el estudiante se puede valer para hacer sus propios entregables.

Los notebooks pueden ser ejecutados en [Google Colab](https://colab.research.google.com/) de forma individual y auto-suficiente. Además, se ofrecen los respectivos `environment.yaml` y `requirements.txt` para crear en local via [Anaconda](https://anaconda.org/anaconda/conda) o [virtualenv](https://virtualenv.pypa.io/en/latest/) un entorno de trabajo de python donde puedan ser ejecutados los ejercicios en forma local.

Dentro de [assets](./assets/) se encuentran recursos visuales utilizados en los notebooks.

Dentro de [datasets](./datasets/) se encuentran unos conjuntos de datos que se usan en algunos de los notebooks. Sin embargo, la mayoría de los notebooks descargarán los datasets de diferentes fuentes.

## Unidades

### [1. Series de Tiempo](./Unidad%201%20-%20Time%20Series/)
- [Estadística basica](./Unidad%201%20-%20Time%20Series/estadistica-basica.ipynb)
- [Moving Average Model](./Unidad%201%20-%20Time%20Series/moving-average-model.ipynb)
- [Autoregressive Model](./Unidad%201%20-%20Time%20Series/autoregressive-model.ipynb)
- [ARIMA Model](./Unidad%201%20-%20Time%20Series/arima.ipynb)
- [Sarima Model](./Unidad%201%20-%20Time%20Series/sarima-sarimax.ipynb)
- [MLP para series de tiempo](./Unidad%201%20-%20Time%20Series/mlp-time-series.ipynb)
- [LSTM para series de tiempo](./Unidad%201%20-%20Time%20Series/lstm-time-series.ipynb)
- [CNN para series de tiempo](./Unidad%201%20-%20Time%20Series/cnn-time-series.ipynb)
- [N-HiTS](./Unidad%201%20-%20Time%20Series/nhits-time-series.ipynb)

### [2. Graph Neural Networks](./Unidad%202%20-%20Graph%20Neural%20Networks/)
- [Grafos](./Unidad%202%20-%20Graph%20Neural%20Networks/graphs.ipynb)
- [Pytorch Geometric](./Unidad%202%20-%20Graph%20Neural%20Networks/pytorch-geometric.ipynb)
- [Vanilla Graph Neural Networks](./Unidad%202%20-%20Graph%20Neural%20Networks/vanilla-gnn.ipynb)
- [Graph Convolutional Networks](./Unidad%202%20-%20Graph%20Neural%20Networks/gcn.ipynb)
- [Graph Attention Networks](./Unidad%202%20-%20Graph%20Neural%20Networks/gat.ipynb)
- [Predicción de enlaces](./Unidad%202%20-%20Graph%20Neural%20Networks/link-prediction.ipynb)
- [Clasificación de gráfos](./Unidad%202%20-%20Graph%20Neural%20Networks/graph-classification.ipynb)

### [3. Transformers](./Unidad%203%20-%20Transformers/)
- [Transformers desde cero](./Unidad%203%20-%20Transformers/transformers-from-scratch.ipynb)
- [Clasificación de texto con HuggingFace](./Unidad%203%20-%20Transformers/text-classification-with-hf.ipynb)
- [Generación de texto](./Unidad%203%20-%20Transformers/text-generation.ipynb)
- [Chatbot, Retrieval Augmented Generation - RAG](./Unidad%203%20-%20Transformers/ollama-rag.ipynb)

### Datasets
- [Shampoo Sales (shampo_sales.csv)](https://www.kaggle.com/datasets/redwankarimsony/shampoo-saled-dataset?select=shampoo_sales.csv)
- [ADIDAS quarterly Sales (adidas_revenue1.csv)](https://www.kaggle.com/datasets/kofi2614/adidas-quarterly-sales)

