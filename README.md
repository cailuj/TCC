# Análise Comparativa de CNNs e XAI para Diagnóstico de Câncer de Mama em Mamografias

Repositório com o código-fonte desenvolvido para o Trabalho de Conclusão de Curso intitulado **“Análise Comparativa de Modelos de Aprendizado Profundo e Técnicas de IA Explicável (XAI) para a Otimização do Diagnóstico de Câncer de Mama em Mamografias”**.

O projeto investiga o uso de Redes Neurais Convolucionais (CNNs) para classificação de mamografias e aplica técnicas de Inteligência Artificial Explicável (XAI) para auditar se as decisões dos modelos estão alinhadas com regiões clinicamente relevantes das imagens.

## Objetivo

Realizar uma análise comparativa entre diferentes arquiteturas de CNN aplicadas à classificação de mamografias, avaliando tanto o desempenho preditivo quanto a qualidade das explicações visuais produzidas por técnicas de XAI.

O estudo compara as arquiteturas:

- ResNet50
- DenseNet121
- VGG16

E utiliza técnicas de explicabilidade, principalmente:

- Grad-CAM++
- LIME
- SHAP, em experimentos complementares presentes nos notebooks

## Contexto do estudo

O câncer de mama é um problema relevante de saúde pública, e o diagnóstico precoce é fundamental para aumentar as chances de tratamento efetivo. Modelos de aprendizado profundo podem auxiliar na triagem e classificação de exames de imagem, mas seu uso em contextos clínicos exige transparência, rastreabilidade e validação das regiões que influenciam a decisão do modelo.

Por esse motivo, este trabalho não avalia apenas métricas tradicionais, como acurácia, precisão, recall e F1-score. A pesquisa também analisa a coerência espacial das explicações geradas pelos modelos, comparando mapas de ativação e regiões destacadas com as regiões de interesse disponíveis no dataset.

## Aviso de uso

Este repositório tem finalidade exclusivamente acadêmica e experimental. Os modelos, códigos e resultados aqui apresentados não devem ser utilizados para diagnóstico médico real, tomada de decisão clínica ou substituição da avaliação de profissionais de saúde.

## Dataset

O projeto utiliza o dataset **MINI-DDSM**, derivado do Digital Database for Screening Mammography. O conjunto de dados contém mamografias organizadas em três classes:

- Normal
- Benign
- Cancer

Nas fases com análise espacial, é utilizado o arquivo `DataWMask.xlsx`, que contém metadados das imagens e caminhos para as máscaras de lesão quando disponíveis.

Estrutura esperada no Google Drive para as fases 3 e 4:

```text
/content/drive/MyDrive/MINI-DDSM/
├── DataWMask.xlsx
├── ... arquivos e subpastas do dataset
```

Colunas esperadas no arquivo `DataWMask.xlsx`:

```text
fullPath
Status
Tumour_Contour
```

Onde:

- `fullPath` indica o caminho relativo da imagem no dataset.
- `Status` indica a classe da imagem: `Normal`, `Benign` ou `Cancer`.
- `Tumour_Contour` indica o caminho relativo da máscara da lesão, quando existente. Para imagens sem máscara, o valor pode ser `-`.

Para as fases 1 e 2, os notebooks utilizam uma organização simplificada das imagens em pastas por classe:

```text
/content/drive/MyDrive/organized data/
├── benign/
├── cancer/
└── normal/
```

O notebook localizado em `organização/organizing_mini_ddsm.ipynb` realiza a cópia e organização das imagens do MINI-DDSM para essa estrutura.

## Estrutura do repositório

```text
TCC-main/
├── organização/
│   └── organizing_mini_ddsm.ipynb
│
├── Fase 1/
│   ├── fase_1_densenet121_grid_search.ipynb
│   ├── fase_1_resnet50_grid_search.ipynb
│   └── fase_1_vgg16_grid_search.ipynb
│
├── Fase 2/
│   ├── fase_2_densenet121_grid_search.ipynb
│   ├── fase_2_resnet50_grid_search.ipynb
│   └── fase_2_vgg16_grid_search.ipynb
│
├── Fase 3/
│   ├── fase_3_densenet121.ipynb
│   ├── fase_3_densenet121.py
│   ├── fase_3_resnet50.ipynb
│   ├── fase_3_resnet50.py
│   ├── fase_3_vgg16.ipynb
│   └── fase_3_vgg16.py
│
├── Fase 4/
│   ├── ajustes_explicabilidade_densenet121.ipynb
│   ├── ajustes_explicabilidade_resnet50.ipynb
│   ├── ajustes_explicabilidade_vgg16.ipynb
│   ├── fase_4_implementação_ajustada_densenet121_+_xai.py
│   ├── fase_4_implementação_ajustada_resnet50_+_xai.py
│   └── fase_4_implementação_ajustada_vgg16_+_xai.py
│
└── README.md
```

## Descrição das fases experimentais

### Organização dos dados

O notebook `organização/organizing_mini_ddsm.ipynb` organiza as imagens do MINI-DDSM em diretórios por classe, removendo arquivos de máscara da estrutura simplificada usada nos primeiros experimentos.

### Fase 1: Grid Search exploratório

A Fase 1 realiza uma busca inicial de hiperparâmetros para as arquiteturas ResNet50, DenseNet121 e VGG16.

Configurações avaliadas:

- Funções de ativação: `relu` e `swish`
- Taxas de aprendizado: `1e-3` e `1e-4`
- Otimizadores: `Adam` e `SGD`
- Imagens redimensionadas para `224x224`
- Batch size igual a `32`
- Treinamento com até `50` épocas

A seleção do melhor modelo é feita principalmente com base no recall de validação, considerando a importância de reduzir falsos negativos no contexto de detecção de câncer.

### Fase 2: Grid Search refinado

A Fase 2 aprofunda a busca de hiperparâmetros, mantendo a comparação entre as três arquiteturas. O objetivo é refinar as configurações selecionadas na Fase 1 e obter uma base mais consistente para os experimentos posteriores.

### Fase 3: Treinamento CNN + XAI

A Fase 3 realiza o treinamento das arquiteturas selecionadas e aplica técnicas de explicabilidade para avaliação das decisões dos modelos.

Principais componentes:

- Leitura do `DataWMask.xlsx`
- Separação entre treino, validação e teste
- Divisão por paciente usando `GroupShuffleSplit`, reduzindo o risco de vazamento entre conjuntos
- Mapeamento das classes:

```python
class_map = {
    "Normal": 0,
    "Benign": 1,
    "Cancer": 2
}
```

- Treinamento com fine-tuning das arquiteturas pré-treinadas na ImageNet
- Avaliação por acurácia, matriz de confusão e relatório de classificação
- Aplicação de Grad-CAM++, LIME e SHAP
- Cálculo de IoU quando há máscara de lesão disponível

### Fase 4: Ajustes de explicabilidade e atenção espacial

A Fase 4 testa estratégias para melhorar a atenção espacial dos modelos e verificar se as redes passam a considerar regiões mais próximas da lesão.

Estratégias avaliadas:

- Aumento de brilho em 20%
- Sobreposição de máscaras como mecanismo de atenção forçada
- Recortes das regiões de interesse, também chamados de crops
- Reavaliação com Grad-CAM++ e LIME

Essa fase é importante porque confronta o desempenho numérico dos modelos com a qualidade espacial das explicações geradas.

## Requisitos

O projeto foi desenvolvido em ambiente Google Colab, com uso de GPU. A execução local é possível, mas exige adaptação dos caminhos de arquivos e remoção das dependências específicas do Colab, como `google.colab.drive`.

Principais dependências:

```text
python
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
opencv-python
lime
shap
scikit-image
openpyxl
```

Instalação sugerida em ambiente Colab ou ambiente virtual local:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python lime shap scikit-image openpyxl
```

## Como executar

### 1. Preparar o dataset

Baixe o MINI-DDSM e coloque os arquivos no Google Drive, preferencialmente em:

```text
/content/drive/MyDrive/MINI-DDSM/
```

Garanta que o arquivo `DataWMask.xlsx` esteja presente na pasta principal do dataset.

### 2. Organizar os dados para as fases iniciais

Execute o notebook:

```text
organização/organizing_mini_ddsm.ipynb
```

Esse notebook gera a estrutura:

```text
/content/drive/MyDrive/organized data/
├── benign/
├── cancer/
└── normal/
```

### 3. Executar a Fase 1

Execute os notebooks da pasta `Fase 1` para realizar a busca exploratória de hiperparâmetros:

```text
Fase 1/fase_1_resnet50_grid_search.ipynb
Fase 1/fase_1_densenet121_grid_search.ipynb
Fase 1/fase_1_vgg16_grid_search.ipynb
```

### 4. Executar a Fase 2

Execute os notebooks da pasta `Fase 2` para o refinamento experimental:

```text
Fase 2/fase_2_resnet50_grid_search.ipynb
Fase 2/fase_2_densenet121_grid_search.ipynb
Fase 2/fase_2_vgg16_grid_search.ipynb
```

### 5. Executar a Fase 3

Execute os notebooks ou scripts da pasta `Fase 3` para treinamento dos modelos com análise XAI:

```text
Fase 3/fase_3_resnet50.ipynb
Fase 3/fase_3_densenet121.ipynb
Fase 3/fase_3_vgg16.ipynb
```

Os scripts `.py` correspondentes foram exportados a partir dos notebooks do Colab. Caso sejam executados localmente, será necessário adaptar:

```python
BASE_PATH = "/content/drive/MyDrive/MINI-DDSM/"
EXCEL_PATH = os.path.join(BASE_PATH, "DataWMask.xlsx")
```

### 6. Executar a Fase 4

Execute os notebooks ou scripts da pasta `Fase 4` para os testes com ajustes de explicabilidade:

```text
Fase 4/ajustes_explicabilidade_resnet50.ipynb
Fase 4/ajustes_explicabilidade_densenet121.ipynb
Fase 4/ajustes_explicabilidade_vgg16.ipynb
```

Ou os scripts ajustados:

```text
Fase 4/fase_4_implementação_ajustada_resnet50_+_xai.py
Fase 4/fase_4_implementação_ajustada_densenet121_+_xai.py
Fase 4/fase_4_implementação_ajustada_vgg16_+_xai.py
```

## Saídas geradas

Durante a execução, os notebooks e scripts podem gerar:

- Modelos treinados no formato `.keras`
- Curvas de acurácia e loss
- Relatórios de classificação
- Matrizes de confusão
- Mapas de ativação Grad-CAM++
- Explicações LIME
- Visualizações SHAP em experimentos complementares
- Métricas de IoU para comparação entre explicações e máscaras de lesão

Os arquivos de saída não estão necessariamente versionados no repositório, pois podem ser grandes e dependem da execução de cada experimento.

## Resultados resumidos

A tabela abaixo resume a evolução da acurácia nos principais experimentos:

| Arquitetura | Fase 3 | Fase 4 - Brilho +20% | Fase 4 - Máscara | Fase 4 - Crop |
|---|---:|---:|---:|---:|
| ResNet50 | 78,24% | 76,81% | 71,13% | 71,88% |
| DenseNet121 | 69,45% | 69,39% | 71,70% | 69,39% |
| VGG16 | 61,66% | 64,03% | 67,39% | 57,79% |

A ResNet50 apresentou a maior acurácia global na Fase 3. A DenseNet121 apresentou comportamento mais estável ao longo dos experimentos. A análise por XAI indicou que modelos com boa acurácia ainda podem produzir explicações espacialmente frágeis, com baixa sobreposição entre os mapas de ativação e as regiões reais de lesão.

O principal achado do estudo é que a acurácia isolada não é suficiente para validar modelos aplicados ao diagnóstico médico. Em cenários de saúde, a explicabilidade espacial deve ser considerada parte essencial da avaliação.

## Limitações

- O dataset MINI-DDSM não está incluído no repositório.
- A execução completa exige GPU e alto consumo de memória.
- Os caminhos dos arquivos foram definidos para Google Colab e Google Drive.
- Os scripts `.py` foram exportados dos notebooks e podem exigir ajustes para execução fora do Colab.
- Os resultados dependem da organização correta do dataset e da disponibilidade das máscaras de lesão.
- O projeto não possui validação clínica externa e não deve ser usado em ambiente médico real.

## Recomendações para reprodução

Para facilitar a reprodutibilidade:

- Use o mesmo caminho base do Google Drive indicado nos notebooks.
- Mantenha o arquivo `DataWMask.xlsx` na raiz do dataset.
- Preserve os nomes esperados das colunas de metadados.
- Execute as fases na ordem proposta.
- Utilize GPU no Colab ou ambiente equivalente.
- Registre versões das principais bibliotecas caso os experimentos sejam refeitos.
- Salve modelos treinados e resultados intermediários em uma pasta separada, como `outputs/` ou `results/`.

## Autoria

**Autora:** Júlia de Souza Calado  
**Curso:** Bacharelado em Ciência da Computação  
**Instituição:** CESAR School  
**Orientação:** Prof. Pamela Thays Lins Bezerra  
**Coorientação:** Prof. João Victor Tinoco de Souza Abreu  
**Ano:** 2026

## Como citar

Caso utilize este repositório como referência, cite o trabalho acadêmico associado:

```text
CALADO, Júlia de Souza. Análise Comparativa de Modelos de Aprendizado Profundo e Técnicas de IA Explicável (XAI) para a Otimização do Diagnóstico de Câncer de Mama em Mamografias. Trabalho de Conclusão de Curso, CESAR School, Recife, 2026.
```

## Licença

Este repositório não possui uma licença definida no momento. Até que uma licença seja adicionada, todos os direitos permanecem reservados à autora.
