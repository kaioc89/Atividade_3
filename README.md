# Atividade 2

Implementação de framework **LLM-as-a-Judge** com persistência em PostgreSQL para a disciplina de Tópicos Avançados em Engenharia de Software e Sistemas de Informação.

Este repositório contém a fundação local do projeto e a pipeline inicial de julgamento por LLM: ambiente Python, testes, PostgreSQL local, restore do backup inicial, execução de juízes remotos HTTP, persistência das avaliações e geração de backups auditáveis.

## Links
- Aplicativo: https://topicos-av2.onrender.com/
- Tutorial: https://drive.google.com/file/d/1zn6FkIB2tEC8DrpiqqPARJn5utK5VGru/view?usp=sharing
- Vídeo: https://www.vidline.com/share/V0O6HJYW6N/baeaf4f0df82e55101bb35a1edeca104

## Requisitos

- Python 3.11+
- Docker com Docker Compose v2
- `make`

## Estrutura

- `src/atividade_2/`: código Python importável.
- `tests/`: suíte pytest.
- `resources/`: entradas estáveis e fixtures.
- `outputs/`: artefatos gerados localmente.
- `outputs/backup/`: backups SQL gerados por `make db-backup`.
- `outputs/audit/`: logs locais gerados por `run-judge`.
- `scripts/`: automações locais de banco.
- `backup_atividade_2_reset.sql`: backup SQL fixo usado para iniciar/resetar o banco AV2.
- `backup_atividade_2.sql`: última versão compartilhável gerada por `make db-backup` quando `APP_ENV=prod`.

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

## Banco Local

O banco local usa PostgreSQL 18 com pgvector via Docker Compose, para compatibilidade com o backup existente e com a base RAG vetorial.

Crie o arquivo `.env` automaticamente e suba o PostgreSQL:

```bash
make db-up
```

O comando:

- copia `.env.example` para `.env` se necessário;
- baixa `pgvector/pgvector:0.8.2-pg18` se a imagem não existir localmente;
- sobe o container `topicos-av2-postgres`;
- valida conexão com `app_dev`;
- cria `app_test` se ainda não existir.

Conexão local padrão:

```text
postgresql://postgres:postgres@localhost:5432/app_dev
```

## RAG Vetorial

Na Web UI, o botão `Gerar embeddings` materializa automaticamente a base vetorial a partir da curadoria ativa quando ainda não existe uma `retrieval_run` ativa para o dataset.

Durante a geração, a aplicação consulta as URLs de fonte cadastradas na curadoria e adiciona o texto recuperado como chunks extras da base vetorial. URLs inacessíveis, vazias, com conteúdo não textual ou acima de 5 MB são reportadas no status da tela e no comando CLI.

## Restore Inicial

Restaure o backup inicial somente quando o banco estiver vazio:

```bash
make db-migrate-or-create
```

Esse comando usa `backup_atividade_2_reset.sql`. Se o banco já tiver tabelas públicas, o restore é ignorado para evitar sobrescrever dados locais.

Para forçar o restore sobre um banco já populado, limpando os schemas `public` e `av3` antes de restaurar:

```bash
make db-migrate-or-create FORCE=1
```

O script também aceita o argumento diretamente:

```bash
./scripts/db_migrate_or_create.sh --force
```

`make db-migrate-or-create --force` não é suportado pelo GNU Make local, porque `--force` é interpretado como opção do próprio `make`.

Valide o restore:

```bash
make db-restore-validate
```

A validação confirma as tabelas centrais:

- `datasets`
- `modelos`
- `perguntas`
- `respostas_atividade_1`
- `avaliacoes_juiz`

## Backup

Gere um backup SQL auditável do banco local:

```bash
make db-backup
```

O arquivo gerado segue o formato:

```text
outputs/backup/atividade_2_YYYYmmdd_HHMMSS.sql
```

Cada execução grava um arquivo timestampado em `outputs/backup/`. Quando `APP_ENV=prod`, a execução também atualiza `backup_atividade_2.sql` na raiz com a última versão do backup. Em `dev` e `test`, a raiz não é atualizada. Os arquivos timestampados mantêm o histórico local e são ignorados pelo Git; o arquivo da raiz é único, compartilhável e permanece versionado. O baseline de reset fica separado em `backup_atividade_2_reset.sql` e não é sobrescrito por esse fluxo.

## Running the LLM-as-a-Judge pipeline with a remote model

O banco e a pipeline rodam localmente. O modelo juiz roda em um endpoint HTTP.

```text
PostgreSQL local -> Python local -> endpoint do juiz -> avaliação salva no PostgreSQL
```

O endpoint pode ser Colab, Hugging Face Inference Endpoint, vLLM, llama.cpp, LM Studio, proxy do Ollama ou qualquer servidor compatível com OpenAI.

`.env` é local e não deve ser commitado. `.env.example` é só o template.

## Web UI local para execução auditável

Além da CLI, o projeto inclui uma console Web local para configurar, validar e acompanhar execuções do `run-judge`.

Suba o PostgreSQL e a Web UI pelo Docker Compose:

```bash
make web-up
```

Acesse:

```text
http://127.0.0.1:8000
```

A Web UI:

- carrega defaults do `.env`;
- mostra configuração efetiva sem exibir API keys;
- valida configuração por dry-run;
- inicia execução real sem chamar a CLI por subprocesso;
- mostra progresso percentual do batch;
- exibe o comando CLI equivalente e o caminho do audit log;
- inclui a aba `Prompt Juizes` para manter prompt, persona, contexto, rubrica e saida por `dataset`.

Na aba `Prompt Juizes`, cada salvamento cria uma nova versao imutavel em `prompt_juizes`. A avaliacao persistida em `avaliacoes_juiz` passa a apontar para `id_prompt_juiz`, preservando qual versao exata do prompt foi usada em cada execucao. A propria UI mostra o preview com uma questao de exemplo e a lista de versoes disponiveis.

#### Atualizando o prompt J1 (discursiva/peca)

Mudancas no prompt default do codigo afetam apenas bases novas/restores. Para aplicar o prompt J1 recalibrado em um banco ja existente:

1. Suba a Web UI: `make web-up`
2. Abra `http://127.0.0.1:8000` e acesse a aba `Prompt Juizes`.
3. Selecione o dataset `J1` (alias de `OAB_Bench`), ajuste `Prompt/Persona/Contexto/Rubrica/Saida` e clique em `Salvar`.
4. Confirme que a nova versao ficou `ativa` e que o preview renderiza corretamente.

Esse fluxo cria uma nova versao no Postgres sem migrations e sem alterar automaticamente versoes ativas.

Por segurança, o serviço Web é publicado apenas em `127.0.0.1:${WEB_PORT:-8000}`. Se o endpoint do juiz roda no host da máquina e a Web roda em container, `localhost` dentro do container aponta para o próprio container; em macOS/Windows, use `host.docker.internal` quando precisar acessar LM Studio, llama.cpp, Ollama proxy ou serviço similar no host.

Comandos úteis:

```bash
make web-up
make web-logs
make web-down
```

`make web-up` libera automaticamente a porta configurada em `WEB_PORT` antes de subir a Web UI: para containers Docker que publicam essa porta, executa `docker stop`; para processos locais escutando na porta, tenta `SIGTERM` e depois `SIGKILL` se necessário.

### Configuração rápida

Se ainda não existe `.env`:

```bash
cp .env.example .env
```

Se o `.env` já existe, não sobrescreva. Copie apenas as variáveis novas de `.env.example`.

#### Variáveis que você precisa configurar

Estas dependem do endpoint de cada pessoa:

| Variável | O que colocar |
|---|---|
| `REMOTE_JUDGE_BASE_URL` | URL base do endpoint. Ex.: `https://.../v1`. |
| `REMOTE_JUDGE_API_KEY` | Chave/token default do endpoint. Não commit. |
| `REMOTE_JUDGE_MODEL` | Juiz 1. Também é usado no modo `single`. |
| `REMOTE_SECONDARY_JUDGE_MODEL` | Juiz 2 do painel. |
| `REMOTE_ARBITER_JUDGE_MODEL` | Juiz árbitro. |

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

#### Endpoint e chave por juiz

`REMOTE_JUDGE_BASE_URL` e `REMOTE_JUDGE_API_KEY` configuram o endpoint do juiz 1. Se cada juiz roda em um provedor diferente, defina também uma URL e uma key para os outros slots do painel.

Formato:

```env
REMOTE_JUDGE_BASE_URL=https://endpoint-do-juiz-1.example.com/v1
REMOTE_JUDGE_API_KEY=chave-do-juiz-1

REMOTE_SECONDARY_JUDGE_BASE_URL=https://endpoint-do-juiz-2.example.com/v1
REMOTE_SECONDARY_JUDGE_API_KEY=chave-do-juiz-2

REMOTE_ARBITER_JUDGE_BASE_URL=https://endpoint-do-arbitro.example.com/v1
REMOTE_ARBITER_JUDGE_API_KEY=chave-do-arbitro
```

O modo `single` usa `REMOTE_JUDGE_MODEL`, o mesmo modelo do juiz 1. Se só a URL ou só a key específica for configurada, a validação falha antes da execução. Se não houver endpoint específico para o slot, a chamada usa o endpoint global.

#### Variáveis que normalmente ficam no padrão

| Variável | Padrão recomendado | Quando mudar |
|---|---|---|
| `APP_ENV` | `dev` | Use `prod` para publicar a última versão do backup em `backup_atividade_2.sql`; `dev` e `test` salvam apenas em `outputs/backup/`. |
| `BACKUP_ROOT_FILE` | `backup_atividade_2.sql` | Caminho do arquivo único de última versão. No container Web, é sobrescrito para `/workspace/backup_atividade_2.sql` para escrever na raiz do repositório host. |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/app_dev` | Se seu banco usa outra porta, usuário ou database. |
| `JUDGE_PROVIDER` | `remote_http` | Não mude por enquanto. |
| `JUDGE_PANEL_MODE` | `2plus1` | Use `single` para smoke test barato. |
| `REMOTE_JUDGE_MODEL` | `gpt-oss-120b` | Juiz 1 e modelo do modo `single`. |
| `REMOTE_SECONDARY_JUDGE_MODEL` | `llama-3.3-70b-instruct` | Juiz 2 do painel. |
| `REMOTE_ARBITER_JUDGE_MODEL` | `m-prometheus-14b` | Juiz árbitro. |
| `JUDGE_ARBITRATION_MIN_DELTA` | `2` | Para arbitragem mais ou menos sensível. |
| `JUDGE_ALWAYS_RUN_ARBITER` | `false` | Use `true` só para auditoria/amostra. |
| `JUDGE_EXECUTION_STRATEGY` | `sequential` | Use `parallel` para concorrência fixa ou `adaptive` para ajuste por endpoint/modelo. |
| `JUDGE_BATCH_SIZE` | `10` | Quantas respostas pendentes buscar por execução incremental. |
| `JUDGE_ADAPTIVE_INITIAL_CONCURRENCY` | `1` | Concorrência inicial por grupo no modo `adaptive`. |
| `JUDGE_ADAPTIVE_MAX_CONCURRENCY` | `4` | Teto local de concorrência por grupo no modo `adaptive`. |
| `JUDGE_ADAPTIVE_SUCCESS_THRESHOLD` | `5` | Sucessos consecutivos antes de aumentar concorrência em `+1`. |
| `JUDGE_ADAPTIVE_MAX_RETRIES` | `3` | Retentativas para `429`, `5xx` e timeouts. |
| `JUDGE_ADAPTIVE_BASE_BACKOFF_SECONDS` | `2` | Backoff inicial quando não houver `Retry-After`. |
| `JUDGE_ADAPTIVE_MAX_BACKOFF_SECONDS` | `60` | Backoff máximo por tentativa. |
| `REMOTE_JUDGE_TIMEOUT_SECONDS` | `180` | Para modelos lentos. |
| `REMOTE_JUDGE_TEMPERATURE` | `0.0` | Mantenha assim para avaliação mais determinística. |
| `REMOTE_JUDGE_MAX_TOKENS` | `4000` | Reduza só se o endpoint tiver limite baixo; Gemini truncou JSON com valores menores nos testes. |
| `REMOTE_JUDGE_TOP_P` | `1.0` | Normalmente não precisa mudar. |
| `REMOTE_JUDGE_OPENAI_COMPATIBLE` | `true` | Use `false` só para endpoint JSON não OpenAI. |
| `JUDGE_SAVE_RAW_RESPONSE` | `true` | Use `false` se não quiser manter a resposta bruta no run. |

Precedência:

```text
.env > ambiente do processo > default do código > erro de validação
```

Para parâmetros passados diretamente ao `run-judge`, a CLI continua tendo precedência sobre a configuração resolvida do `.env`.

### Modelos juízes selecionados

Aliases aceitos no `.env` e no CLI:

| Alias | Provider model id | Papel |
|---|---|---|
| `gpt-oss-120b` | `openai/gpt-oss-120b` | primário |
| `llama-3.3-70b-instruct` | `meta-llama/Llama-3.3-70B-Instruct` | primário |
| `m-prometheus-14b` | `Unbabel/M-Prometheus-14B` | árbitro/calibração |

Também é possível passar diretamente um provider model id. Valores sem alias são usados como informados.

### Modos de execução

| Modo | O que faz | Quando usar |
|---|---|---|
| `single` | Roda um juiz. | Smoke test, debug ou endpoint com um modelo só. |
| `primary_only` | Roda o painel primário. | Comparar dois juízes sem árbitro. |
| `2plus1` | Roda dois primários e chama árbitro se houver divergência. | Execução metodológica principal. |
| `2plus1 --always-run-arbiter` | Roda os três juízes sempre. | Amostra de auditoria ou apresentação. |

### Execução sequencial ou paralela

Controle chamadas de API em `.env`:

```env
JUDGE_EXECUTION_STRATEGY=adaptive
```

Use `sequential` para modelo local, pouca VRAM ou endpoint frágil. Use `parallel` para endpoint remoto que aceita concorrência fixa. Use `adaptive` para deixar o executor ajustar concorrência por endpoint/modelo, reduzindo em `429` e refileirando com backoff.

No modo `2plus1`, só os dois primários podem rodar em paralelo. O árbitro sempre roda depois, porque depende da diferença entre as notas.

No modo `adaptive`, a prioridade é `juiz 1 -> juiz 2 -> árbitro`. O árbitro continua sendo agendado apenas depois das notas primárias e da regra de arbitragem. Para ver o plano inicial sem selecionar respostas nem avaliar modelos:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode 2plus1 \
  --dataset J2 \
  --batch-size 10 \
  --judge-execution-strategy adaptive \
  --preflight-report
```

Override por execução:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode primary_only \
  --judge-model gpt-oss-120b \
  --secondary-judge-model llama-3.3-70b-instruct \
  --judge-execution-strategy parallel \
  --dataset J2 \
  --batch-size 10
```

### Execução incremental por batch

Use `--batch-size` para limitar quantas respostas pendentes serão selecionadas no banco:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --dataset J2 \
  --batch-size 10
```

Se `--batch-size` não for informado, a CLI usa `JUDGE_BATCH_SIZE` do `.env`; se a variável também não existir, usa `10`.

A seleção é automática: a pipeline busca respostas ainda sem avaliação bem-sucedida para o modo/modelos resolvidos. Avaliações ausentes e avaliações registradas com status diferente de `success` permanecem elegíveis para uma próxima execução. Assim, repetir o mesmo comando continua a partir do estado persistido em `avaliacoes_juiz`.

### Exemplos

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

Smoke test real com uma peça/discursiva:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --dataset J1 \
  --batch-size 1
```

Rodar exatamente dois juízes:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode primary_only \
  --judge-model gpt-oss-120b \
  --secondary-judge-model llama-3.3-70b-instruct \
  --dataset J2 \
  --batch-size 10
```

Rodar o modo `2plus1` padrão do `.env`:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode 2plus1 \
  --dataset J2 \
  --batch-size 10
```

Rodar os três juízes sempre:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode 2plus1 \
  --always-run-arbiter \
  --dataset J2 \
  --batch-size 20
```

### Audit log

Toda execução de `run-judge` grava um log detalhado em arquivo e também mostra o progresso no terminal.

- Terminal: mostra cada etapa principal, com pontos dinâmicos em operações longas quando o terminal suporta animação.
- Arquivo: grava timestamps UTC, configuração resolvida sem segredos, seleção de respostas, chamadas por resposta/modelo, parsing, persistência, skips e resumo final.
- Resumo de execução: inclui os modelos efetivos e o host de cada endpoint resolvido.
- Padrão: `outputs/audit/judge_run_YYYYmmdd_HHMMSS.log`.
- Logs locais em `outputs/audit/*.log` são ignorados pelo Git.

Escolha um caminho explícito:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --judge-model m-prometheus-14b \
  --dataset J2 \
  --batch-size 1 \
  --audit-log outputs/audit/smoke_j2.log
```

Desative a animação de terminal quando quiser saída estável para captura em arquivo:

```bash
.venv/bin/python -m atividade_2.cli run-judge \
  --panel-mode single \
  --judge-model m-prometheus-14b \
  --dataset J2 \
  --batch-size 1 \
  --no-audit-animation
```

### Colab e endpoints compatíveis

Para usar Colab, exponha uma URL HTTP compatível com OpenAI e configure somente no `.env`:

```env
REMOTE_JUDGE_BASE_URL=https://seu-endpoint-colab.example.com/v1
REMOTE_JUDGE_API_KEY=sua-chave-local
REMOTE_JUDGE_OPENAI_COMPATIBLE=true
```

O mesmo padrão vale para vLLM, llama.cpp, LM Studio ou proxy Ollama. Trocar endpoint e trocar modelos são preocupações separadas: normalmente o endpoint fica fixo em `.env`, e o modelo ou painel é selecionado por CLI.

### Persistência

Cada juiz executado gera uma linha individual em `avaliacoes_juiz`. O campo `chain_of_thought` é usado como justificativa auditável curta, não como raciocínio privado. A pipeline adiciona colunas opcionais de metadados de painel (`papel_juiz`, `rodada_julgamento`, `motivo_acionamento`) se o schema restaurado ainda não as tiver.

O prompt instrui o juiz a retornar apenas um objeto JSON bruto, sem markdown nem texto extra. O parser aceita JSON puro, blocos cercados por crase e respostas com texto antes/depois do primeiro objeto JSON válido, mas rejeita nota fora da escala 1-5 ou justificativa ausente/vazia.

### Backup e restore

Antes de uma execução completa:

```bash
make db-up
make db-migrate-or-create
make db-restore-validate
```

Depois de uma execução relevante:

```bash
make db-backup
```

Para restaurar do zero, use o fluxo recomendado do projeto:

```bash
make db-reset
make db-up
make db-migrate-or-create
make db-restore-validate
```

### Troubleshooting

- `REMOTE_JUDGE_BASE_URL is required`: copie as variáveis novas de `.env.example` para seu `.env`.
- `REMOTE_JUDGE_API_KEY is required`: defina uma chave local para o endpoint. Não use chave real em `.env.example`.
- `Remote judge response did not contain model text`: confirme se o endpoint responde no formato OpenAI `/chat/completions` ou defina `REMOTE_JUDGE_OPENAI_COMPATIBLE=false`.
- `Judge response contains invalid JSON`: o modelo não seguiu o contrato; rode com `--batch-size 1`, ajuste o endpoint/modelo e repita.
- Avaliações duplicadas não são regravadas para o mesmo conjunto resposta/modelo/papel/modo; use outro modelo ou limpe intencionalmente o banco se precisar reconstruir uma execução.
- Use sempre `.venv/bin/python`, nunca `python` ou `python3`, para comandos do projeto.

## Comandos Make

```bash
make venv
make install
make test
make db-up
make db-migrate-or-create
make db-restore-validate
make db-backup
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

`make db-reset` remove o volume local do PostgreSQL. Use apenas quando quiser descartar o banco local e restaurar do zero.
`make db-migrate-or-create FORCE=1` limpa o schema `public` do banco atual e restaura `backup_atividade_2_reset.sql` sem depender de o banco estar vazio.

## Fluxo Recomendado

Para validar o projeto do zero:

```bash
make install
make db-up
make db-migrate-or-create
make db-restore-validate
make db-backup
make test
make db-down
```

## Fora de Escopo Neste Estágio

- ORM;
- Alembic/migrations;
- importadores de datasets;
- cálculo de Spearman;
- automação via notebook.

`atividade2.ipynb` permanece como artefato separado e não é necessário para subir ou validar o ambiente local.
