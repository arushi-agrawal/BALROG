# wiki-BALROG: Enhancing Language Model Agents Through Retrieval-Augmented Game Knowledge

BALROG is a novel benchmark evaluating agentic LLM and VLM capabilities on long-horizon interactive tasks using reinforcement learning environments. We introduce Wiki-BALROG that enhances the LLMs by dynamically integrating game knowledge from the NetHack Wiki into the BALROG agents.
Our agents outperform the benchmarks set by [BALROG](https://arxiv.org/abs/2411.13543).

## Installation

We advise using conda for the installation

```bash
conda create -n balrog python=3.10 -y
conda activate balrog

git clone https://github.com/arushi-agrawal/wiki-BALROG.git
cd BALROG
pip install -e .
balrog-post-install
```

On Mac make sure you have `wget` installed for the `balrog-post-install`

## Docker

We have provided some docker images. Please see the [relevant README](docker/README.md).

## Agents added
1. Naive RAG - Integrates RAG in Naive Agent
2. Robust Naive RAG - Integrates RAG in Robust Naive Agent
3. COT RAG - Integrates RAG in COT Agent
4. Robust COT RAG - Integrates RAG in Robust COT Agent
5. Robust COT Improved - New Agent developed by us, with enhanced prompts.

## ⚡️ Evaluate using vLLM locally

We support running LLMs/VLMs locally using [vLLM](https://github.com/vllm-project/vllm). You can spin up a vLLM client and evaluate your agent on BALROG in the following way:

```bash
pip install vllm numpy==1.23
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8080

python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=32 \
  client.client_name=vllm \
  client.model_id=meta-llama/Llama-3.2-1B-Instruct \
  client.base_url=http://0.0.0.0:8080/v1
```

On Mac you might have to first export the following to suppress some fork() errors:

```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

Check out [vLLM](https://github.com/vllm-project/vllm) for more options on how to serve your models fast and efficiently.

## 🛜 Evaluate using API

We support how of the box clients for OpenAI, Anthropic and Google Gemini APIs. If you want to evaluate an agent using one of these APIs, you first have to set up your API key in one of two ways:

You can either directly export it:

```bash
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
```

Or you can modify the `SECRETS` file, adding your api keys.

You can then run the evaluation with:

```bash
python eval.py \
  agent.type=robust_cot_improved_rag \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=16 \
  client.client_name=gemini \
  client.model_id=gemini-2.0-flash
```
