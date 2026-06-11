# Tópicos Avançados em ES e SI I - Atividades 2 e 3

Projeto da Equipe 4 no domínio jurídico para a disciplina **Tópicos Avançados em Engenharia de Software e Sistemas de Informação**.

O repositório consolida duas etapas conectadas:

- **AV2:** implementação de um framework **LLM-as-a-Judge** com persistência em PostgreSQL, rubricas, painel de juízes, auditoria e consultas analíticas.
- **AV3:** evolução do ecossistema para avaliar o impacto de **Recuperação Aumentada por Geração (RAG)** nas respostas dos mesmos modelos candidatos da AV1/AV2, comparando o baseline **Sem_RAG** contra novas respostas **Com_RAG**.

## Links

- Aplicativo: https://topicos-av3.onrender.com/
- Tutorial: https://drive.google.com/file/d/1CyBay00Mu3DVn7Y7j2Ktu_-cfdgCN7Zv/view?usp=sharing
- Vídeo: https://www.vidline.com/share/V0YOHR53DG/a836f8ed0cc2f441ec1660d932177ddf
- Curadoria: https://atividade3-60692236585.us-west1.run.app/

## Objetivo da AV3

A Atividade 3 exige medir se uma base externa de conhecimento, recuperada por RAG, melhora ou piora as respostas dos modelos em um domínio de alta especialidade.

No nosso caso, o recorte é jurídico:

- **J1 / OAB-Bench:** questões abertas, discursivas e peças.
- **J2 / OAB Exams:** questões objetivas de múltipla escolha.

A comparação principal é pareada:

```text
Pergunta original
  -> recuperação top-k no índice RAG
  -> prompt Com_RAG
  -> mesmo modelo candidato da AV1/AV2
  -> resposta AV3
  -> avaliação LLM-as-a-Judge pós-RAG
  -> comparação contra baseline Sem_RAG
```

A AV3 não troca o problema nem avalia um benchmark novo. Ela reexecuta os mesmos modelos e os mesmos recortes da AV1/AV2 com contexto jurídico recuperado, permitindo medir impacto de forma controlada.

## Pergunta experimental

> A inclusão de contexto jurídico recuperado por RAG melhora a qualidade das respostas dos modelos candidatos em relação ao baseline Sem_RAG?

A resposta é avaliada por:

- variação de nota do juiz entre **Sem_RAG** e **Com_RAG**;
- identificação de ganhos, perdas e ruído;
- comparação por dataset, modelo, curador, especialidade jurídica e tipo de questão;
- análise estatística e visual no dashboard;
- inspeção dos contextos recuperados para verificar se o RAG ajudou ou confundiu o modelo.

## Principais requisitos da Atividade 3

| Exigência | Implementação no projeto |
|---|---|
| Curadoria de documentos externos | Uso das referências jurídicas cadastradas na curadoria e ingestão de conteúdo textual das fontes disponíveis. |
| Temporalidade e conhecimento atualizado | Preservação de metadados de fonte, documento e chunk para auditoria do contexto usado. |
| Chunking e embeddings | Pipeline de geração de chunks e embeddings persistidos no schema `av3`. |
| Base vetorial | PostgreSQL 18 com `pgvector`. |
| Recuperação top-k | Busca por similaridade para selecionar os fragmentos mais relevantes por pergunta. |
| Re-inferência dos modelos | Comando `run-candidates-rag` gera respostas AV3 com prompt Com_RAG. |
| LLM-as-a-Judge pós-RAG | Reuso/adaptação do pipeline de juízes da AV2 para avaliar respostas Com_RAG. |
| Comparação Sem_RAG vs Com_RAG | Views e dashboards no schema `analytics` para análise pareada. |
| Backup auditável | `backup_atividade_2.sql` promovido apenas quando contém AV2 + AV3 RAG. |

## Arquitetura conceitual

```text
Curadoria J1/J2
  -> documentos jurídicos externos
  -> extração textual
  -> chunks rastreáveis
  -> embeddings
  -> PostgreSQL + pgvector
  -> retrieval top-k
  -> prompt Com_RAG
  -> modelos candidatos
  -> respostas AV3
  -> juiz AV3
  -> analytics Sem_RAG vs Com_RAG
  -> dashboard e apresentação
```

## Organização do banco

O projeto mantém o PostgreSQL como fonte auditável dos dados.

| Schema | Papel |
|---|---|
| `public` | Estrutura consolidada da AV2: datasets, perguntas, respostas Sem_RAG, modelos, prompts e avaliações de juiz. |
| `av3` | Estruturas novas de RAG: documentos, chunks, embeddings, retrieval runs, contextos recuperados, respostas Com_RAG e avaliações AV3. |
| `analytics` | Views comparativas para dashboard e análise: notas Sem_RAG vs Com_RAG, deltas, ganhos, perdas, ruído e correlações. |

A estrutura da AV2 não deve ser reescrita. A AV3 é uma extensão controlada sobre a base existente.

## Fluxo da AV3

### 1. Curadoria e base RAG

A equipe parte das questões e metadados já curados na AV1. Para cada questão, a base RAG usa documentos jurídicos externos associados à legislação, tema ou especialidade.

Na Web UI, o botão **Gerar embeddings** materializa a base vetorial a partir da curadoria ativa quando ainda não existe uma `retrieval_run` ativa para o dataset.

Durante essa etapa, o sistema:

- consulta URLs de fonte cadastradas na curadoria;
- extrai conteúdo textual quando disponível;
- ignora fontes inacessíveis, vazias, não textuais ou acima do limite configurado;
- gera chunks rastreáveis;
- calcula embeddings;
- grava o índice vetorial no PostgreSQL.

### 2. Recuperação top-k

Para cada pergunta, o pipeline busca os chunks mais similares no índice vetorial.

A recuperação deve ser auditável: cada resposta Com_RAG precisa permitir identificar quais fragmentos foram enviados ao modelo candidato.

### 3. Geração Com_RAG

Os modelos candidatos são reexecutados com a pergunta original e os fragmentos recuperados pelo RAG.

A regra metodológica central é preservar comparabilidade: cada candidato deve responder apenas o mesmo recorte que já era de sua responsabilidade na AV1/AV2.

### 4. Avaliação pós-RAG

As novas respostas são avaliadas pelo pipeline LLM-as-a-Judge adaptado da AV2.

O juiz deve avaliar não apenas a resposta final, mas também se o RAG:

- corrigiu erro jurídico;
- reduziu alucinação;
- trouxe fundamentação útil;
- inseriu ruído;
- confundiu o modelo;
- melhorou ou piorou a aderência ao gabarito/rubrica.

### 5. Análise Sem_RAG vs Com_RAG

A análise principal compara, para a mesma pergunta e o mesmo modelo:

```text
Nota_Juiz_Sem_RAG -> Nota_Juiz_Com_RAG -> Delta
```

Os resultados devem destacar:

- maiores ganhos;
- maiores pioras;
- estabilidade por modelo;
- impacto por especialidade jurídica;
- concordância entre juízes;
- correlação e ranking antes/depois do RAG.

## Estrutura do repositório

```text
src/atividade_2/        Código Python importável da AV2/AV3
tests/                  Suíte pytest
resources/              Entradas estáveis e fixtures
outputs/                Artefatos locais gerados
outputs/backup/         Backups SQL timestampados
outputs/audit/          Logs locais de execução
scripts/                Automações locais de banco
backup_atividade_2.sql  Backup canônico AV2 + AV3 RAG, promovido explicitamente
backup_atividade_2_reset.sql
                        Backup fixo para bootstrap/reset da AV2
```

## Requisitos

- Python 3.11+
- Docker com Docker Compose v2
- `make`

## Setup Python

Instale o projeto em modo editável com dependências de desenvolvimento:

```bash
make install
```

Execute os testes:

```bash
make test
```

Os comandos Python usam explicitamente `.venv/bin/python`.

## Banco local

O banco local usa PostgreSQL 18 com `pgvector` via Docker Compose, para compatibilidade com o backup existente e com a base RAG vetorial.

Suba o PostgreSQL:

```bash
make db-up
```

Conexão local padrão:

```text
postgresql://postgres:postgres@localhost:5432/app_dev
```

Restaure o backup inicial somente quando o banco estiver vazio:

```bash
make db-migrate-or-create
```

Para atualizar apenas a estrutura do banco ativo e os seeds determinísticos de AV3, sem restore nem limpeza:

```bash
make db-ensure-schema
```

Valide o restore:

```bash
make db-restore-validate
```

## Web UI local

Suba PostgreSQL e Web UI pelo Docker Compose:

```bash
make web-up
```

Acesse:

```text
http://127.0.0.1:8000
```

A Web UI apoia:

- configuração de endpoints e modelos;
- execução auditável de juízes;
- visualização de progresso;
- edição/versionamento de prompts de juiz;
- geração de embeddings RAG;
- dashboards AV2, AV3 e comparação Sem_RAG vs Com_RAG.

Comandos úteis:

```bash
make web-up
make web-logs
make web-down
```

Por segurança, o serviço Web é publicado apenas em `127.0.0.1:${WEB_PORT:-8000}`.

## Configuração de endpoints

`.env` é local e não deve ser commitado. `.env.example` é apenas o template.

Se ainda não existe `.env`:

```bash
cp .env.example .env
```

Variáveis principais para juízes remotos:

| Variável | Uso |
|---|---|
| `REMOTE_JUDGE_BASE_URL` | URL base do endpoint do juiz 1. |
| `REMOTE_JUDGE_API_KEY` | Chave/token local do endpoint. |
| `REMOTE_JUDGE_MODEL` | Modelo do juiz 1 e do modo `single`. |
| `REMOTE_SECONDARY_JUDGE_MODEL` | Modelo do juiz 2. |
| `REMOTE_ARBITER_JUDGE_MODEL` | Modelo árbitro. |
| `JUDGE_PANEL_MODE` | `single`, `primary_only` ou `2plus1`. |
| `JUDGE_EXECUTION_STRATEGY` | `sequential`, `parallel` ou `adaptive`. |

Exemplo mínimo:

```env
JUDGE_PROVIDER=remote_http
REMOTE_JUDGE_BASE_URL=https://seu-endpoint.example.com/v1
REMOTE_JUDGE_API_KEY=sua-chave-local
JUDGE_PANEL_MODE=single
REMOTE_JUDGE_MODEL=gpt-oss-120b
REMOTE_SECONDARY_JUDGE_MODEL=llama-3.3-70b-instruct
REMOTE_ARBITER_JUDGE_MODEL=m-prometheus-14b
REMOTE_JUDGE_OPENAI_COMPATIBLE=true
```

## Modelos juízes

Aliases aceitos no `.env` e na CLI:

| Alias | Provider model id | Papel |
|---|---|---|
| `gpt-oss-120b` | `openai/gpt-oss-120b` | primário |
| `llama-3.3-70b-instruct` | `meta-llama/Llama-3.3-70B-Instruct` | primário |
| `m-prometheus-14b` | `Unbabel/M-Prometheus-14B` | árbitro/calibração |

## Execução dos juízes

Validar configuração sem chamar banco nem endpoint:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --judge-model openai/gpt-oss-120b \
  --dataset J2 \
  --batch-size 1 \
  --dry-run
```

Smoke test real com uma questão objetiva:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --dataset J2 \
  --batch-size 1
```

Smoke test real com uma questão discursiva/peça:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --dataset J1 \
  --batch-size 1
```

Rodar o modo `2plus1`:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode 2plus1 \
  --dataset J2 \
  --batch-size 10
```

## Execução dos candidatos Com_RAG

Os candidatos AV3 são executados com o comando `run-candidates-rag`.

Exemplo para modelo local `llama.cpp` / Jurema:

```bash
LLAMA_CPP_URL=http://localhost:8080/v1 \
.venv/bin/python -m atividade_2.cli run-candidates-rag \
  --dataset J1 \
  --candidate-model jurema-7b-q4_k_m \
  --provider remote_http \
  --batch-size 1 \
  --question-sequence-start 95 \
  --question-sequence-end 95 \
  --candidate-execution-strategy sequential \
  --no-audit-animation
```

Exemplo para modelo local `llama.cpp` / Curió:

```bash
LLAMA_CPP_URL=http://localhost:8080/v1 \
.venv/bin/python -m atividade_2.cli run-candidates-rag \
  --dataset J1 \
  --candidate-model curio-edu-7b-gguf \
  --provider remote_http \
  --batch-size 1 \
  --question-sequence-start 95 \
  --question-sequence-end 95 \
  --candidate-execution-strategy sequential \
  --no-audit-animation
```

Restrições operacionais para modelos locais:

- modelos `llama.cpp` devem ser executados manualmente antes do comando;
- execuções locais são sequenciais;
- não misturar candidatos locais com lotes remotos OpenRouter/Featherless;
- rodar modelos locais separadamente para reduzir risco de erro e consumo de memória.

## Backup

Gere um backup SQL auditável do banco local:

```bash
make db-backup
```

O dump é gerado em:

```text
outputs/backup/atividade_2_YYYYmmdd_HHMMSS.sql
```

O arquivo `backup_atividade_2.sql` é o backup canônico protegido para restore compartilhável de AV2 + AV3 RAG.

Promova o backup canônico apenas de forma explícita:

```bash
make db-backup-promote
```

Antes de promover, o script valida que estas contagens sejam maiores que zero no banco atual:

- `public.respostas_atividade_1`
- `public.avaliacoes_juiz`
- `av3.rag_chunks`
- `av3.rag_embeddings`
- `av3.retrieval_runs` com `ativo = TRUE`

## Logs e auditoria

Toda execução de juiz grava log detalhado em arquivo e mostra progresso no terminal.

- Logs padrão: `outputs/audit/judge_run_YYYYmmdd_HHMMSS.log`
- Logs locais em `outputs/audit/*.log` são ignorados pelo Git.
- O log registra configuração resolvida sem segredos, seleção de respostas, chamadas por resposta/modelo, parsing, persistência, skips e resumo final.

Para saída estável em terminal:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --dataset J2 \
  --batch-size 1 \
  --no-audit-animation
```

## Entregáveis da AV3

A entrega final deve conter:

1. **Tutorial em PDF consolidado** com arquitetura RAG, critérios de inclusão dos documentos, queries SQL e resultados estatísticos.
2. **Pasta Atividade_3** ou documentação equivalente com scripts, prompts e artefatos novos da AV3.
3. **Backup SQL ou dump** refletindo dados antes/depois do RAG.
4. **Vídeo demonstrativo** de 10 a 20 minutos, com participação dos membros e execução/explicação das partes implementadas.
5. **README atualizado** com visão geral, links e instruções de reprodução.

## Critérios de avaliação da AV3

| Critério | Peso | Como o projeto responde |
|---|---:|---|
| Arquitetura RAG e justificativa | 30% | Base vetorial com chunks, embeddings, fontes rastreáveis e recuperação top-k. |
| Pipeline e evolução no banco | 25% | Schemas `av3` e `analytics` preservando `public` como baseline AV2. |
| Meta-avaliação por juiz | 20% | LLM-as-a-Judge pós-RAG para medir ganho, falha ou ruído. |
| Análise estatística e de erros | 15% | Deltas, correlações, rankings, especialidades e casos de ganho/piora. |
| Apresentação e documentação | 10% | README, tutorial, vídeo e dashboard voltados à defesa presencial. |

## Fluxo recomendado do zero

```bash
make install
make db-up
make db-migrate-or-create
make db-ensure-schema
make db-restore-validate
make web-up
make test
```

Depois de gerar embeddings, candidatos Com_RAG e avaliações pós-RAG:

```bash
make db-backup
make db-backup-promote
```

## Troubleshooting

- `REMOTE_JUDGE_BASE_URL is required`: copie as variáveis novas de `.env.example` para `.env`.
- `REMOTE_JUDGE_API_KEY is required`: defina uma chave local para o endpoint. Não commite chaves reais.
- `Remote judge response did not contain model text`: confirme se o endpoint responde no formato OpenAI `/chat/completions` ou defina `REMOTE_JUDGE_OPENAI_COMPATIBLE=false`.
- `Judge response contains invalid JSON`: o modelo não seguiu o contrato; rode com `--batch-size 1`, ajuste endpoint/modelo e repita.
- Avaliações duplicadas não são regravadas para o mesmo conjunto resposta/modelo/papel/modo.
- Use sempre `.venv/bin/python`, nunca `python` ou `python3`, para comandos do projeto.

## Comandos Make

```bash
make venv
make install
make test
make db-up
make db-migrate-or-create
make db-ensure-schema
make db-restore-validate
make db-backup
make db-backup-promote
make db-status
make db-psql
make db-logs
make web-up
make web-logs
make web-down
make db-down
make db-reset
make clean
```

## Fora de escopo nesta documentação

- reexplicar toda a AV1;
- substituir o tutorial em PDF exigido na entrega;
- documentar segredos, tokens ou endpoints privados;
- promover backup canônico sem validação explícita;
- alterar schema, código ou dados de produção por meio deste README.
