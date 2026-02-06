# Awesome Graph-based Agent Memory

<div align="center">
     <a href="https://arxiv.org/abs/2602.05665" target="_blank"><img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv:2602.05665"></a>
     <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/DEEP-PolyU/Awesome-GraphMemory?color=blue"/></a>
     <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/DEEP-PolyU/Awesome-GraphMemory"/></a>
     <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/forks/DEEP-PolyU/Awesome-GraphMemory"/></a>
   
</div>

This repository provides a comprehensive collection of research papers, benchmarks, and open-source projects on **Graph-based Agent Memory**. It includes contents from our survey paper 📖<em>"[**Graph-based Agent Memory: Taxonomy, Techniques, and Applications**](https://arxiv.org/pdf/2602.05665)"</em> and will be continuously updated.

🤗 **You are vey welcome to contribute to this repository** by launching an issue or a pull request. If you find any missing resources or come across interesting new research works, please don’t hesitate to open an issue or submit a PR!

📫 **Contact us via emails:** `chang.yang@connect.polyu.hk`, `qinggang.zhang@polyu.edu.hk`

---

<div>
<h3 align="center">
       <p align="center"><img width="100%" src="figures/illustration_memory_comparison.png" /></p>
    <p align="center"><em>Comparison between Traditional Agent Memory and Graph-based Agent Memory.</em></p>
</div>

## 📜 Catalog

> **[Awesome Graph-based Agent Memory](#awesome-graph-based-agent-memory)**
>
> - **[🔥 News](#-news)**
> - **[📖 Overview](#-overview)**
> - **[🪴 Taxonomy](#-taxonomy)**
>   - [Memory Extraction](#memory-extraction)
>   - [Memory Storage](#memory-storage)
>   - [Memory Retrieval](#memory-retrieval)
>   - [Memory Evolution](#memory-evolution)
> - **[📦 Benchmark](#-benchmark)**
> - **[📦 Projects](#-projects)**
> - **[📃 Citation](#-citation)**

---

## 🔥 News

* **[2025-02-03]** 🔥🔥 Repository launched based on our survey paper.

## 🪴 Taxonomy

### Memory Extraction

<p align="center"><img width="100%" src="figures/extraction.png" /></p>

#### Textual Data
#### Sequential Data
#### Multimodal Data

- (arXiv‘25) **Can an LLM Induce a Graph? Investigating Memory Drift and Context Length** [[Paper]](https://arxiv.org/pdf/2510.03611)
- (arXiv‘24) **On the structural memory of llm agents** [[Paper]](https://arxiv.org/pdf/2412.15266?)
- (arXiv‘25) **Personaagent with graphrag: Community-aware knowledge graphs for personalized llm** [[Paper]](https://arxiv.org/abs/2511.17467)
- (arXiv‘25) **Scaling graph chain-of-thought reasoning: A multi-agent framework with efficient llm serving** [[Paper]](https://arxiv.org/pdf/2511.01633)
- (ACL‘25) **HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model** [[Paper]](https://aclanthology.org/2025.acl-long.1575.pdf)
  
- (NeurIPS‘23) **Reflexion: Language agents with verbal reinforcement learning** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)
- (arXiv‘25) **Mem-α: Learning Memory Construction via Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2509.25911)
- (arXiv‘25) **MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation** [[Paper]](https://arxiv.org/abs/2508.19236)
- (arXiv‘23) **Multi-Temporal Lip-Audio Memory for Visual Speech Recognition** [[Paper]](https://arxiv.org/abs/2305.04542)

### Memory Storage
<p align="center"><img width="100%" src="figures/storage.png" /></p>

#### Knowledge Graph Structure

- (TMLR'25) **MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory**  [[Paper]](https://arxiv.org/abs/2404.11672)
- (arXiv'25) **AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents** [[Paper]](https://arxiv.org/abs/2407.04363)
- (arXiv'25) **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** [[Paper]](https://arxiv.org/abs/2504.19413)

#### Hierarchical Memory Structure

- (arXiv'25) **ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents** [[Paper]](https://arxiv.org/abs/2511.12960)
- (arXiv'25) **SGMem: Sentence Graph Memory for Long-Term Conversational Agents** [[Paper]](https://arxiv.org/abs/2509.21212)
- (arXiv'25) **G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems** [[Paper]](https://arxiv.org/abs/2506.07398)
- (arXiv'25) **LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning** [[Paper]](https://arxiv.org/abs/2502.05453)
- (EMNLP'24) **Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs** [[Paper]](https://arxiv.org/abs/2409.19401)

#### Temporal Graph Structure

- (arXiv'25) **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** [[Paper]](https://arxiv.org/abs/2501.13956)
- (arXiv'25) **TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues** [[Paper]](https://arxiv.org/abs/2502.01630)
- (arXiv'25) **MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning** [[Paper]](https://arxiv.org/abs/2510.13614)

#### Hypergraph Structure

- (arXiv'25) **HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation** [[Paper]](https://arxiv.org/abs/2503.21322)
- (arXiv'25) **HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge** [[Paper]](https://arxiv.org/abs/2502.18125)

#### Hybrid Graph Architectures

- (arXiv'24) **Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks** [[Paper]](https://arxiv.org/abs/2408.03615)
- (arXiv'24) **KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph** [[Paper]](https://arxiv.org/abs/2402.11163)

### Memory Retrieval

<p align="center"><img width="100%" src="figures/retrieval.png" /></p>

### Memory Evolution

<p align="center"><img width="100%" src="figures/evolution.png" /></p>


#### Internal Self-Evolving


- (aXiv'25) **Zep: a temporal knowledge graph architecture for agent memory**. [[paper]](https://arxiv.org/abs/2501.13956)
- (aXiv'25) **Nemori: Self-organizing agent memory inspired by cognitive science**. [[paper]](https://arxiv.org/abs/2508.03341)
- (aXiv'25) **Mem-α: Learning Memory Construction via Reinforcement Learning**. [[paper]](https://arxiv.org/abs/2509.25911)
- (aXiv'24) **From local to global: A graph rag approach to query-focused summarization**. [[paper]](https://arxiv.org/abs/2404.16130)
- (aXiv'23) **RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models**. [[paper]](https://arxiv.org/abs/2307.02738)
- (aXiv'25) **Agent kb: Leveraging cross-domain experience for agentic problem solving**. [[paper]](https://arxiv.org/abs/2507.06229)
- (aXiv'25) **Mem0: Building production-ready ai agents with scalable long-term memory**. [[paper]](https://arxiv.org/abs/2504.19413)
- (aXiv'25) **Flex: Continuous agent evolution via forward learning from experience**. [[paper]](https://arxiv.org/abs/2511.06449)
- (NeurIPS'23) **Reflexion: language agents with verbal reinforcement learning**. [[paper]](https://openreview.net/pdf?id=vAElhFcKW6) 
- (ICLR'24) **Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph**. [[paper]](https://openreview.net/forum?id=nnVO1PvbTv)
- (TKDE'26) **Reliable Reasoning Path: Distilling Effective Guidance for LLM Reasoning With Knowledge Graphs**. [[paper]](https://arxiv.org/abs/2506.10508)
- (ICLR'24) **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**. [[paper]](https://openreview.net/forum?id=ZGNWW7xZ6Q)
- (AAAI'24) **Memorybank: Enhancing large language models with long-term memory**. [[paper]](https://arxiv.org/abs/2305.10250)
- (EMNLP'25) **Memory OS of AI Agent**. [[paper]](https://aclanthology.org/2025.emnlp-main.1318.pdf)
- (aXiv'23) **MemGPT: Towards LLMs as Operating Systems**. [[paper]](https://arxiv.org/abs/2310.08560)
- (ICML'25) **From RAG to Memory: Non-Parametric Continual Learning for Large Language Models**. [[paper]](https://openreview.net/forum?id=LWH8yn4HS2)

#### External Self-Exploration

- (AAAI'24) **Expel: Llm agents are experiential learners**. [[paper]](https://arxiv.org/abs/2308.10144)
- (ICRA'24) **MATRIX: multi-agent trajectory generation with diverse contexts**. [[paper]](https://arxiv.org/abs/2403.06041)
- (aXiv'25) **Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning**. [[paper]](https://arxiv.org/abs/2508.19828)
- (aXiv'25) **Inside-out: Hidden factual knowledge in llms**. [[paper]](https://arxiv.org/abs/2503.15299)
- (aXiv'25) **Memevolve: Meta-evolution of agent memory systems**. [[paper]](https://arxiv.org/abs/2512.18746)
- (aXiv'26) **Beyond static summarization: Proactive memory extraction for llm agents**. [[paper]](https://arxiv.org/abs/2601.04463)
- (aXiv'25) **Agentevolver: Towards efficient self-evolving agent system**. [[paper]](https://arxiv.org/abs/2511.10395)
- (TechReport_Moonshot'25) **Kimi K2.5**. [[paper]](https://www.kimi.com/blog/kimi-k2-5.html)

# 📚 Related Survey Papers
- (arXiv 2025) **Retrieval-Augmented Generation with Graphs (GraphRAG)** [[Paper]](https://arxiv.org/abs/2501.00309)
- (arXiv 2024) **Graph Retrieval-Augmented Generation: A Survey** [[Paper]](https://arXiv.org/pdf/2408.08921)
- (AIxSET 2024) **Graph Retrieval-Augmented Generation for Large Language Models: A Survey** [[Paper]](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4895062)


# 🏆 Benchmarks
| Dataset | Task | Paper | Repo |
| --- | --- | --- | --- |
| GraphRAG-Bench | GraphRAG evaluation | [[arXiv 2025]](https://arxiv.org/abs/2506.05690) | [[Github]](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) |
| DIGIMON | Large-scale graphRAG | [[arXiv 2025]](https://www.arxiv.org/pdf/2503.04338) | [[Github]](https://github.com/JayLZhou/GraphRAG) |

# 💻 Open-source Project
- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Hawksight-AI/semantica) Semantica: an open-source, production-ready semantic layer and GraphRAG framework that sits between raw corpora and LLMs.
- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bibinprathap/VeritasGraph) Graph RAG pipeline that runs locally with ollama and has full source attribution 

## 📃 Citation

```
@article{yang2026graph,
  title={Graph-based Agent Memory: Taxonomy, Techniques, and Applications},
  author={Chang Yang and Chuang Zhou and Yilin Xiao and Su Dong and Luyao Zhuang and Yujing Zhang and Zhu Wang and Zijin Hong and Zheng Yuan and Zhishang Xiang and Shengyuan Chen and Huachi Zhou and Qinggang Zhang and Ninghao Liu and Jinsong Su and Xinrun Wang and Yi Chang and Xiao Huang},
  journal={arXiv preprint arXiv:2602.05665},
  year={2025}
}
```
