# Previsão de geração solar com regressão linear e logística

Mini-projeto que aplica o conteúdo do **Curso 1 da Machine Learning Specialization** (Andrew Ng, Coursera: *Supervised Machine Learning: Regression and Classification*) em dados reais de uma usina solar.

O objetivo é mostrar, em dados reais, cada etapa do curso: regressão linear implementada à mão (custo, gradiente, gradient descent), normalização, escolha da taxa de aprendizado, features polinomiais, overfitting, regularização e regressão logística.

---

## Dados

- **Fonte:** [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) (Kaggle), duas usinas na Índia, leituras a cada 15 minutos, de 15/05/2020 a 17/06/2020 (34 dias).
- **Usada:** só a Usina 1 (`Plant_1_Generation_Data.csv` e `Plant_1_Weather_Sensor_Data.csv`).
- **Preparação:**
  - A potência (`AC_POWER`) dos 22 inversores foi somada por instante, para ter a potência total da usina.
  - A tabela de geração foi unida à do sensor de clima pela data e hora (`merge`), resultando em **3157 instantes**.
  - Foi criada a feature `HORA` (hora + minuto/60).

| Papel | Coluna | Descrição |
|---|---|---|
| Feature | `IRRADIATION` | irradiação solar |
| Feature | `AMBIENT_TEMPERATURE` | temperatura do ar (°C) |
| Feature | `MODULE_TEMPERATURE` | temperatura do painel (°C) |
| Feature | `HORA` | hora do dia (decimal) |
| Saída | `AC_POWER` (soma) | potência total gerada (kW) |

Pico observado da potência total: **29.150 kW** (30/05/2020, 12h00). A relação entre irradiação e potência é quase linear (correlação de 0,996).

---

## Estrutura

```
Projeto-solar/
├── data/                  # CSVs do Kaggle (não versionados)
├── notebooks/
│   └── previsao_geracao_solar.ipynb   # todo o código do projeto, em ordem
├── requirements.txt
└── README.md
```

Para reproduzir:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Baixe os CSVs do Kaggle para `data/`, abra o notebook e use **Run All**.

---

## 1. Regressão linear implementada à mão

Implementei com NumPy, sem biblioteca de ML:

- normalização z-score: $x' = (x - \mu)/\sigma$;
- função de custo: $J(\vec w, b) = \frac{1}{2m}\sum_i (f_{\vec w,b}(\vec x^{(i)}) - y^{(i)})^2$;
- gradientes $\partial J/\partial w_j$ e $\partial J/\partial b$;
- gradient descent, com histórico do custo.

As funções foram validadas primeiro em um exemplo pequeno calculado à mão.

**Taxa de aprendizado ($\alpha$):**

| $\alpha$ | Comportamento |
|---|---|
| 0,01 | converge muito devagar |
| 0,1 | converge, mas ainda longe do mínimo após 1000 iterações |
| 0,5 | converge rápido e estável (escolhido) |
| 0,8 | **diverge**: o custo sobe a cada iteração |

**Resultado (α = 0,5, 1000 iterações, todos os dados):**

| Métrica | Valor |
|---|---|
| Custo final | ≈ 299.150 |
| RMSE | 773,5 kW (≈ 2,7% do pico) |
| $R^2$ | 0,9919 |

## 2. Validação com scikit-learn

O mesmo problema com `StandardScaler` + `LinearRegression`:

| | Gradient descent (à mão) | scikit-learn |
|---|---|---|
| Normalização | — | idêntica (`np.allclose` = `True`) |
| Peso irradiação | 8138,85 | 8141,31 |
| Peso temp. ambiente | −18,93 | −17,78 |
| Peso temp. módulo | 459,97 | 456,59 |
| Peso hora | −11,81 | −11,87 |
| $b$ | 6704,83 | 6704,83 |
| RMSE | 773,4986 kW | 773,4985 kW |

A implementação à mão reproduz o scikit-learn. A pequena diferença nos pesos de temperatura vem de o gradient descent parar perto do mínimo, numa direção onde o custo é quase plano (features correlacionadas).

## 3. Features polinomiais, overfitting e regularização

A partir daqui, os dados foram divididos **por dias** (não por linhas aleatórias, porque instantes vizinhos são quase iguais): os últimos 7 dias para teste e os outros 27 para treino. Normalização e features polinomiais foram ajustadas só no treino (`fit_transform`) e aplicadas no teste (`transform`).

**Graus do polinômio (teste = últimos 7 dias):**

| Grau | Colunas | RMSE treino | RMSE teste |
|---|---|---|---|
| 1 | 4 | 777,3 | 788,6 |
| 2 | 14 | 637,9 | 508,7 |
| 3 | 34 | 598,7 | **416,2** |
| 4 | 69 | 574,2 | 462,0 |
| 5 | 125 | 555,1 | 481,8 |

O erro de treino cai sempre; o de teste cai até o grau 3 e depois sobe: **overfitting**.

**Robustez: três janelas de teste de 7 dias** (RMSE de teste, kW):

| Grau | Últimos 7 | 7 anteriores | 7 antes dessas | Média |
|---|---|---|---|---|
| 1 | 788,6 | 947,4 | 803,9 | 846,6 |
| 2 | 508,7 | 810,8 | 619,8 | 646,4 |
| 3 | 416,2 | 802,6 | 626,3 | 615,0 |
| 4 | 462,0 | 797,7 | 671,4 | 643,7 |
| 5 | 481,8 | 792,0 | 639,7 | 637,8 |

O melhor grau muda de janela para janela, mas a melhora do grau 1 para o grau 2 aparece nas três. Ou seja, o ganho do modelo não linear é consistente; a escolha exata do grau, não.

**Regularização (`Ridge`, média das três janelas):**

| Modelo | λ = 0,01 | λ = 0,1 | λ = 1 | λ = 10 | λ = 100 |
|---|---|---|---|---|---|
| Grau 3 | 598,7 | **598,2** | 616,0 | 647,5 | 843,2 |
| Grau 5 | — | 609,6 | 607,2 | 640,1 | 766,4 |

λ pequeno melhora um pouco; λ grande força os pesos a zero e o modelo volta ao nível do linear (**underfitting**).

**Comparação com modelos ingênuos (últimos 7 dias):**

| Modelo | RMSE teste |
|---|---|
| Sempre a média do treino | 7890,9 kW |
| Média do treino por horário do dia | 2936,4 kW |
| Regressão linear (grau 1) | 788,6 kW |
| Grau 3 | 416,2 kW |

O modelo não aprende só "de dia gera mais": o ganho sobre a média por horário vem das variáveis de clima.

**Modelo de regressão final:** grau 3 + `StandardScaler` + `Ridge` (λ = 0,1). RMSE de teste médio de **~598 kW** (de ~419 a ~789 conforme a semana), cerca de **2,1% do pico**, contra ~847 kW do modelo linear. A diferença para o grau 2 ou o grau 5 + Ridge é de poucos por cento, menor que a variação entre semanas.

## 4. Regressão logística: geração boa ou baixa

- Só instantes diurnos (`IRRADIATION > 0`), 1754 linhas.
- **"Boa"** = potência ≥ mediana da potência diurna do treino (**12.695 kW**); **"baixa"** = abaixo. Esse corte é uma escolha do projeto, não existe nos dados.
- Mesmo corte treino/teste (últimos 7 dias).

| Métrica | Valor |
|---|---|
| Acurácia treino | 97,7% |
| Acurácia teste | **96,5%** |
| Chute da classe mais comum (teste) | 55,2% |

Matriz de confusão (teste; linhas = real, colunas = previsto):

| | prev. baixa | prev. boa |
|---|---|---|
| **real baixa** | 199 | 4 |
| **real boa** | 9 | 156 |

- A sigmoide calculada à mão ($1/(1+e^{-z})$, com $z = \vec w\cdot\vec x + b$) bate com `predict_proba` do scikit-learn.
- Os 13 erros estão todos perto do limiar (potência real entre 12.406 e 13.299 kW), onde a separação boa/baixa é arbitrária.
- **Fronteira de decisão (2 features):**
  - irradiação × temperatura ambiente: reta quase vertical em irradiação ≈ 0,42; a temperatura ambiente quase não influencia;
  - irradiação × temperatura do módulo: reta mais inclinada, mas a irradiação continua dominando. As duas variáveis têm correlação de 0,93, então a temperatura do módulo carrega boa parte da mesma informação.

---

## Limitações

- **Estimativa, não previsão do futuro:** as features são medidas no mesmo instante da potência. Prever a geração de amanhã exigiria previsão do tempo, com erro bem maior.
- **Poucos dados:** uma usina, 34 dias, só maio e junho. Nada garante o comportamento em outras estações.
- **Teste pequeno:** janelas de 7 dias. O erro varia bastante de semana para semana (de ~420 a ~810 kW), então os números são indicativos.
- **Escolhas feitas olhando o teste:** grau e λ foram escolhidos nas mesmas janelas usadas para avaliar, o que deixa o resultado um pouco otimista. O correto seria um conjunto de validação separado.
- **Janelas de teste no meio do período:** nas três janelas, o treino inclui dias depois do teste, o que não imita o uso real.
- **Pesos das temperaturas instáveis:** as features são muito correlacionadas, então o peso individual de cada uma não deve ser interpretado como efeito isolado.
- **Classificação "boa/baixa" é quase trivial:** como a potência depende quase toda da irradiação, o classificador responde basicamente "o sol está forte?".
- **Dados com falhas:** alguns instantes têm menos de 22 inversores registrados e há outliers não explicados; não foram removidos.
- **Não transfere para outra usina:** os pesos dependem do tamanho da usina. Para outro local (por exemplo, Goiânia), seria preciso retreinar com dados locais.

## Próximos passos

- Separar treino, validação e teste corretamente, ou usar validação cruzada por blocos de dias.
- Repetir na Usina 2 para ver se o padrão se mantém.
- Classificação mais útil: detectar instantes em que a usina gera **menos do que a irradiação justificaria** (falha de inversor, sujeira nos painéis).
- Prever o futuro usando previsão meteorológica como entrada.

## Ferramentas

Python 3.11, NumPy, pandas, Matplotlib, scikit-learn, Jupyter (VS Code).
