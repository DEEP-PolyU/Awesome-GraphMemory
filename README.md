# Awesome Graph-based Agent Memory

<div align="center">
     <a href="https://arxiv.org/abs/2602.05665" target="_blank"><img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv:2602.05665"></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/DEEP-PolyU/Awesome-GraphMemory"/></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/forks/DEEP-PolyU/Awesome-GraphMemory"/></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/DEEP-PolyU/Awesome-GraphMemory?color=blue"/></a>
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
<p align="center"><img width="100%" src="figures/extraction.png" /></p>

### Memory Extraction
- (arXiv 2025) **LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora**  [[Paper]](https://arxiv.org/abs/2510.10114)
- (EMNLP 2025) **Don’t Forget the Base Retriever! A Low-Resource Graph-based Retriever for Multi-hop Question Answering** [[Paper]](https://aclanthology.org/2025.emnlp-industry.174/)

<p align="center"><img width="100%" src="figures/storage.png" /></p>

### Memory Storage
#### Knowledge Graph Structure
- (TMLR 2025) **MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory**  [[Paper]](https://arxiv.org/abs/2404.11672)
- (arXiv 2025) **AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents** [[Paper]](https://arxiv.org/abs/2407.04363)
- (arXiv 2025) **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** [[Paper]](https://arxiv.org/abs/2504.19413)

#### Hierarchical Memory Structure
- (arXiv 2025) **ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents** [[Paper]](https://arxiv.org/abs/2511.12960)
- (arXiv 2025) **SGMem: Sentence Graph Memory for Long-Term Conversational Agents** [[Paper]](https://arxiv.org/abs/2509.21212)
- (arXiv 2025) **G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems** [[Paper]](https://arxiv.org/abs/2506.07398)
- (arXiv 2025) **LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning** [[Paper]](https://arxiv.org/abs/2502.05453)
- (EMNLP 2024) **Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs** [[Paper]](https://arxiv.org/abs/2409.19401)

#### Temporal Graph Structure
- (arXiv 2025) **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** [[Paper]](https://arxiv.org/abs/2501.13956)
- (arXiv 2025) **TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues** [[Paper]](https://arxiv.org/abs/2502.01630)
- (arXiv 2025) **MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning** [[Paper]](https://arxiv.org/abs/2510.13614)

#### Hypergraph Structure
- (arXiv 2025) **HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation** [[Paper]](https://arxiv.org/abs/2503.21322)
- (arXiv 2025) **HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge** [[Paper]](https://arxiv.org/abs/2502.18125)

#### Hybrid Graph Architectures
- (arXiv 2024) **Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks** [[Paper]](https://arxiv.org/abs/2408.03615)
- (arXiv 2024) **KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph** [[Paper]](https://arxiv.org/abs/2402.11163)

### Memory Retrieval

<p align="center"><img width="100%" src="figures/retrieval.png" /></p>

### Memory Evolution

<p align="center"><img width="100%" src="figures/evolution.png" /></p>


## 📃 Citation

```
@article{yang2026graph,
  title={Graph-based Agent Memory: Taxonomy, Techniques, and Applications},
  author={Chang Yang and Chuang Zhou and Yilin Xiao and Su Dong and Luyao Zhuang and Yujing Zhang and Zhu Wang and Zijin Hong and Zheng Yuan and Zhishang Xiang and Shengyuan Chen and Huachi Zhou and Qinggang Zhang and Ninghao Liu and Jinsong Su and Xinrun Wang and Yi Chang and Xiao Huang},
  journal={arXiv preprint arXiv:2602.05665},
  year={2025}
}
```
