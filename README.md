# FedIris gRPC — Aprendizagem Federada Distribuída

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![gRPC](https://img.shields.io/badge/gRPC-Protocol%20Buffers%20v3-244c5a?logo=google&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE%20Real--Time-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LogisticRegression-F7931E?logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/Unit%20Tests-15%2F15%20Passing-059669)

Sistema distribuído de **Aprendizagem Federada (*Federated Learning*)** utilizando **gRPC**, **Protocol Buffers** e o algoritmo **FedAvg (*Federated Averaging*)** aplicado à classificação multiclasse do dataset clássico **Iris**, acompanhado de uma **interface web interativa** para visualização em tempo real da topologia distribuída, métricas de convergência, pesos do modelo e validação de protocolo.

---

## Informações Acadêmicas

| Campo | Detalhes |
| :--- | :--- |
| **Instituição** | Universidade Federal do Ceará (UFC) |
| **Disciplina** | Sistemas Distribuídos — Trabalho 1 |
| **Professor** | Filipe Maciel Roberto |
| **Aluno / Autor** | [Felipi Ribeiro dos Santos](https://github.com/felpsrib) |

---

## Visão Geral

No aprendizado de máquina centralizado tradicional, os dados produzidos por diferentes dispositivos ou organizações precisam ser transferidos para um servidor central antes do treinamento. Em cenários reais — como saúde, finanças ou aplicativos móveis — essa centralização enfrenta barreiras severas de **privacidade**, **regulamentação** (como LGPD e GDPR) e **custo de comunicação**.

A **Aprendizagem Federada** inverte esse paradigma: em vez de mover os dados até o processamento, **move-se o treinamento até os dados**. Os dados brutos nunca saem dos nós clientes; apenas os parâmetros atualizados do modelo trafegam pela rede até o servidor coordenador.

### Ciclo Federado Implementado

1. **Inicialização (Rodada 0):** o servidor cria o modelo global inicial (`LogisticRegression` com pesos zerados) e disponibiliza seus parâmetros.
2. **Distribuição (`GetGlobalModel`):** cada cliente consulta o modelo global atual via gRPC e verifica se há uma nova rodada disponível.
3. **Treinamento Local:** cada cliente treina localmente sobre sua partição IID do dataset **Iris** (80% treino local / 20% teste local) utilizando `warm_start=True` e `max_iter=1`.
4. **Envio de Atualizações (`SubmitUpdate`):** os clientes serializam `coef_` (matriz achatada + `shape`) e `intercept_` em mensagens Protobuf e enviam somente os parâmetros e métricas locais ao servidor.
5. **Validação e Agregação (FedAvg):** o servidor valida cada atualização (rejeitando rodadas inválidas, clientes duplicados ou envios pós-encerramento). Ao receber todos os $K$ clientes esperados, consolida o novo modelo global pela média ponderada pelo número de exemplos de cada cliente:

$$w^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{n} \, w_k^{(t+1)}, \qquad n = \sum_{k=1}^{K} n_k$$

6. **Repetição e Convergência:** o ciclo se repete até atingir o número total de rodadas configurado, sinalizando `done = True` aos clientes e encerrando a federação.

---

## Arquitetura do Sistema

O projeto foi organizado em duas camadas desacopladas para preservar **100% de compatibilidade** com a estrutura e a suíte oficial de testes do trabalho (`start/`), ao mesmo tempo em que oferece observabilidade e controle interativo (`app/`):

```mermaid
flowchart LR
    subgraph Clients["Nós Clientes gRPC (start/student_fl_client_grpc.py)"]
        C0["Cliente 0\nPartição IID local"]
        C1["Cliente 1\nPartição IID local"]
        CN["Cliente N-1\nPartição IID local"]
    end

    subgraph Core["Servidor Coordenador gRPC (start/student_fl_server_grpc.py)"]
        SRV["FederatedLearningServicer\n• GetGlobalModel\n• SubmitUpdate\n• _validate_update\n• FedAvg + Avaliação Global"]
    end

    subgraph UI["Camada de Observabilidade & Interface Web (app/)"]
        ORC["ObservableFederatedLearningServicer\n& FederatedOrchestrator"]
        API["FastAPI (REST + SSE /api/events)"]
        WEB["Painel Web FedIris gRPC\nTopologia SVG • Gráficos • Heatmap • Testes"]
    end

    C0 <-->|"Protobuf / HTTP2"| SRV
    C1 <-->|"Protobuf / HTTP2"| SRV
    CN <-->|"Protobuf / HTTP2"| SRV
    SRV -.->|"Telemetria em tempo real"| ORC
    ORC <--> API
    API <--> WEB
```

### Contrato gRPC (`start/federated.proto`)

```protobuf
service FederatedLearning {
  rpc GetGlobalModel (ClientHello) returns (GlobalModel);
  rpc SubmitUpdate (ClientUpdate) returns (UpdateAck);
}
```

- **`ModelParameters`**: transporta `coef_values` (`repeated double`), `coef_shape` (`repeated int32`) e `intercept_values` (`repeated double`), garantindo reconstrução exata dos arrays NumPy entre cliente e servidor.
- **`GlobalModel`**: informa `round`, `total_rounds`, `done`, `message` e `model`.
- **`ClientUpdate` / `UpdateAck`**: transporta `cid`, `round`, `num_examples`, `train_loss`, `train_acc` e o modelo treinado localmente, retornando a confirmação (`accepted`, `message`, `server_round`).

---

## Interface Web Interativa (`FedIris gRPC`)

A interface web foi construída de forma autocontida (HTML5, CSS3, JavaScript puro e gráficos/topologia em SVG — sem dependências de CDN, funcionando inclusive offline) e está organizada em **5 abas**:

1. **Visão Geral & Topologia:** configuração de clientes, rodadas, épocas locais, delay visual e **modo passo a passo**; linha do tempo das 6 etapas do ciclo federado; KPIs ao vivo; topologia gRPC animada com pacotes trafegando nas arestas; gráficos de **Acurácia** e **Log-Loss** comparáveis em tempo real com a **baseline manual** (`fl_sim_manual.py`).
2. **Clientes, Partições & Pesos:** visualização da distribuição IID das 3 espécies do Iris por cliente, peso relativo no FedAvg, **heatmap interativo** da matriz `coef_` e `intercept_` com seletor histórico por rodada e **matriz de confusão 3×3** no conjunto de teste global.
3. **Protocolo gRPC & Validações:** laboratório ao vivo para injetar RPCs inválidos reais (`rodada inválida`, `atualização duplicada` e `envio pós-encerramento`) contra o servidor ativo + tabela filtrável de todo o tráfego Protobuf (bytes, latência em ms e status).
4. **Testar Modelo (Iris):** laboratório de inferência com sliders e presets (Setosa, Versicolor, Virginica) consultando o modelo global agregado atual em tempo real.
5. **Testes Unitários:** executa os 5 módulos oficiais de `start/tests/` e exibe o status detalhado de cada caso de teste.

---

## Estrutura de Diretórios

```text
.
├── pyproject.toml                  # Metadados do projeto, dependências e config Pyright
├── run_ui.py                       # Launcher único da interface web (com fallback de porta livre)
├── fl_sim_manual.py                # Simulação manual em memória (referência didática)
├── README.md                       # Documentação do projeto
├── .gitignore                      # Arquivos ignorados pelo Git
├── start/                          # Núcleo oficial do Trabalho 1 (compatível com avaliação)
│   ├── federated.proto             # Definição do serviço e mensagens gRPC/Protobuf
│   ├── federated_pb2.py            # Stubs de mensagens gerados via grpc_tools.protoc
│   ├── federated_pb2.pyi           # Stubs de tipagem estática gerados via --pyi_out
│   ├── federated_pb2_grpc.py       # Stubs de serviço gRPC gerados via grpc_tools.protoc
│   ├── common_fl.py                # Particionamento Iris, LogisticRegression, treino local e FedAvg
│   ├── student_fl_client_grpc.py   # Implementação completa do cliente gRPC
│   ├── student_fl_server_grpc.py   # Implementação completa do servidor gRPC
│   ├── README.md                   # Roteiro original da disciplina
│   └── tests/                      # Suíte de testes unitários progressivos (15 testes)
│       ├── test_01_client_serialization.py
│       ├── test_02_client_loop.py
│       ├── test_03_server_serialization_and_get.py
│       ├── test_04_server_validation.py
│       ├── test_05_server_submit_and_consolidate.py
│       └── utils.py
└── app/                            # Orquestração gRPC observável e painel web
    ├── orchestrator.py             # Gerenciador de sessões gRPC, telemetria, inferência e testes
    ├── server.py                   # API FastAPI (REST + SSE)
    └── static/                     # Frontend web autocontido
        ├── index.html
        ├── styles.css
        └── app.js
```

---

## Instalação e Configuração

### Pré-requisitos
- Python **3.9 ou superior** (testado em Python 3.13)

### 1. Clonar o repositório e criar o ambiente virtual

**Windows (PowerShell):**
```powershell
git clone git@github.com:felpsrib/aprendizagem-federada-grpc.git
cd aprendizagem-federada-grpc

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install flwr-datasets numpy pandas scikit-learn grpcio grpcio-tools fastapi uvicorn
```

**Linux / macOS:**
```bash
git clone git@github.com:felpsrib/aprendizagem-federada-grpc.git
cd aprendizagem-federada-grpc

python3 -m venv .venv
source .venv/bin/activate
pip install flwr-datasets numpy pandas scikit-learn grpcio grpcio-tools fastapi uvicorn
```

### 2. (Opcional) Regerar os stubs Protobuf
Os arquivos `federated_pb2.py`, `federated_pb2.pyi` e `federated_pb2_grpc.py` já estão incluídos em `start/`. Caso altere `start/federated.proto`, regere-os executando dentro de `start/`:

```bash
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. federated.proto
```

---

## Como Executar

### Opção 1 — Interface Web Interativa (Recomendado)
Na raiz do projeto:
```bash
python run_ui.py
```
O servidor sobe em `http://127.0.0.1:8000` (ou seleciona automaticamente a próxima porta livre caso a 8000 esteja ocupada) e abre o painel no navegador.

### Opção 2 — Execução Distribuída via CLI (Múltiplos Terminais)

**Terminal 1 — Servidor gRPC:**
```bash
cd start
python student_fl_server_grpc.py --host 127.0.0.1 --port 50051 --num-clients 3 --rounds 5
```

**Terminais 2, 3 e 4 — Clientes gRPC:**
```bash
python student_fl_client_grpc.py --cid 0 --server-address 127.0.0.1:50051 --num-clients 3
python student_fl_client_grpc.py --cid 1 --server-address 127.0.0.1:50051 --num-clients 3
python student_fl_client_grpc.py --cid 2 --server-address 127.0.0.1:50051 --num-clients 3
```

### Opção 3 — Simulação Manual em Memória (Referência)
```bash
python fl_sim_manual.py
```

---

## Testes Unitários

Para executar a suíte oficial completa de 15 testes unitários:

```bash
cd start
python -m unittest discover -s tests -v
```

| Módulo de Teste | Escopo Verificado | Status |
| :--- | :--- | :---: |
| `test_01_client_serialization.py` | Serialização/desserialização Protobuf ↔ NumPy e montagem de `ClientUpdate` | Passou (3/3) |
| `test_02_client_loop.py` | Política `should_wait` e laço completo de `run_client` com encerramento em `done=True` | Passou (2/2) |
| `test_03_server_serialization_and_get.py` | Serialização no servidor e resposta de `GetGlobalModel` durante e após o treino | Passou (4/4) |
| `test_04_server_validation.py` | Regras de `_validate_update` (rodada válida, encerrada, inválida e duplicata) | Passou (4/4) |
| `test_05_server_submit_and_consolidate.py` | Armazenamento em buffer, consolidação FedAvg apenas ao receber todos os clientes, limpeza e `training_finished` | Passou (2/2) |

---

## Autor

Desenvolvido por **Felipi Ribeiro dos Santos** ([GitHub: @felpsrib](https://github.com/felpsrib)) como parte da avaliação da disciplina de **Sistemas Distribuídos** (UFC), ministrada pelo professor **Filipe Maciel Roberto**.
