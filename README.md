# FIAP — Visão Computacional e PDI
## Arquivos Python das Atividades Práticas
**Prof. Dr. Paulo Sérgio Rodrigues**

---

### Pré-requisitos
```
python -m venv venv
venv\Scripts\activate
pip install opencv-python numpy
```

---

### Lista de Arquivos

| Arquivo                              | Atividade | Conteúdo                               | Imagem necessária     |
|--------------------------------------|-----------|----------------------------------------|-----------------------|
| `atividade01_setup.py`               | 01        | Verificação do ambiente instalado      | nenhuma (sintética)   |
| `atividade02_leitura_imagem.py`      | 02        | Leitura, exibição e propriedades       | lena.jpg              |
| `atividade03_leitura_video.py`       | 03        | Leitura de vídeo frame a frame         | paisagem01.mp4        |
| `atividade04_binarizacao.py`         | 04        | Acesso a pixels e binarização          | lena.jpg              |
| `atividade05_gamma.py`               | 05        | Transformação gamma (power-law)        | vertebra_mri.jpg      |
| `atividade06_negativo.py`            | 06        | Transformação negativo                 | lena.jpg              |
| `atividade07_stretching.py`          | 07        | Stretching de contraste (piecewise)    | lena.jpg              |
| `atividade08_equalizacao_histograma.py` | 08     | Equalização global e CLAHE             | imagem_aerea.jpg      |
| `atividade09a_filtro_media.py`       | 09A       | Filtro de média (ruído gaussiano)      | lena.jpg              |
| `atividade09b_filtro_mediana.py`     | 09B       | Filtro de mediana (sal-e-pimenta)      | circuito.jpg          |
| `atividade09c_filtro_gaussiano.py`   | 09C       | Filtro gaussiano e variação de sigma   | lena.jpg              |
| `atividade10a_laplaciano.py`         | 10A       | Laplaciano e aguçamento                | lena.jpg              |
| `atividade10b_sobel.py`              | 10B       | Sobel Gx/Gy, magnitude e orientação   | lena.jpg              |
| `atividade10c_canny.py`              | 10C       | Detecção de bordas Canny               | lena.jpg              |

---

### Como executar cada atividade
```
# Certifique-se que o ambiente virtual está ativo: (venv) no terminal
python atividade01_setup.py
python atividade02_leitura_imagem.py
# ... e assim por diante
```

### Imagens clássicas necessárias
Disponíveis na plataforma FIAP ou no GitHub da disciplina:
- https://github.com/PauloSergioImmersiveScience/FIAP

Coloque todas as imagens na **mesma pasta** que os arquivos .py.

### Controles nas janelas OpenCV
- **Qualquer tecla** → avança para o próximo resultado
- **Q** → encerra (nos arquivos de vídeo)
- **P** → pausa/retoma (arquivo de vídeo)
- **S** → salva frame atual (arquivo de vídeo)
