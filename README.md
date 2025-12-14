# Automating Structural Engineering Workflows with Large Language Model Agents

[![arXiv](https://img.shields.io/badge/arXiv-2510.11004-b31b1b.svg)](https://arxiv.org/abs/2510.11004)

## Analogy of future human–AI collaborations

Traditional practice relies on long apprenticeships, with senior engineers transferring expertise to junior engineers through mentorship and problem solving. In contrast, LLM-based multi-agent systems instantiate scalable junior engineer agents that inherit workflows, perform specialized tasks, and coordinate under senior engineers' oversight. As these systems evolve toward self-planning and adaptive learning, fully agentic hierarchies—with senior engineer agent directing junior engineer agents—could transform engineering into a continuously improving, highly efficient practice.

![Analogy of future human–AI collaborations](assets/analogy.png)

## MASSE Framework

Reflecting this practical organizational model, MASSE introduces three distinct agent teams within a simulated structural engineering consultancy environment: Analyst Team, Engineer Team and Management Team. Each agent is assigned a unique role, goal, and set of constraints, and is further equipped with predefined contexts and specialized tools aligned with these responsibilities. MASSE organizes these agent roles into the following structured teams. More detailed descriptions can be found in the appendices.

![MASSE Framework](assets/pipeline.png)

> *MASSE* Overall Framework. The **Analyst Team** combines four layers of tools (LLMs, FEM solvers, engineering documents, and fundamentals) to enable multi-agent collaboration. The **Engineer Team** performs limit state verification (Response<Capacity), while the **Project Manager** coordinates workflows and the **Structural Memory** stores analysis data. The **Safety Manager** conducts the final adequacy check. Three specialized teams are shown: **Analyst** (data extraction and analysis), **Engineer** (design and verification), and **Management** (coordination and decision-making).

### Analyst Team

The Analyst Team automates the preparation of structural engineering data by coordinating specialized agents that extract loading conditions, retrieve information from engineering documentation, execute load determination using rule-based methods, and generate structural models. At a high level, this team transforms unstructured project information and regulatory data into standardized engineering inputs, ensuring that subsequent model, design and verification tasks can be carried out with consistency, efficiency, and scalability.

![Analyst Team](assets/analyst_team.png)

### Engineer Team

The Engineer Team operationalizes the data prepared by the Analyst Team by conducting structural analysis, structural design, and adequacy verification. In broad terms, this team integrates automated simulations and capacity checks to evaluate structural integrity under prescribed loading conditions, enabling systematic, tool-driven assessment of design safety and structural performance.

![Engineer Team](assets/engineer_team.png)

### Manager Team

The Management Team oversees and coordinates the overall MASSE workflow, transforming analytical outputs from the Analyst Team and Engineer Team into authoritative engineering decisions. Broadly speaking, this team manages task allocation, integrates intermediate results, and issues final structural safety conclusions that guide the entire system. As illustrated in the figure below, the Safety Manager plays a central role within this process by delivering the ultimate adequacy verdict, ensuring that all decisions are consistent with professional safety standards.

![Manager Team](assets/safety_manager.png)

## Installation

## MASSE Package

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better. We welcome collaboration from both industry and academy. If you are interested in this line of research, please contact me at [hliang7@ualberta.ca](mailto:hliang7@ualberta.ca) ([Google Scholar](https://scholar.google.com/citations?user=0z0KKtsAAAAJ&hl=en)) or the head of lab Dr. Qipei (Gavin) Mei at [qipei.mei@ualberta.ca](mailto:qipei.mei@ualberta.ca) ([Google Scholar](https://scholar.google.com/citations?user=06VJSBgAAAAJ&hl=en)).

## Disclaimer

- **Research and Educational Purpose Only.**  
  The multi-agent system, methods, code, and data described in this repository are developed solely for academic research and educational use. They are not intended, nor should they be relied upon, for direct application in real-world engineering design, construction, or deployment. The authors, their institutions, and any collaborators explicitly disclaim all liability for any consequences, including but not limited to structural deficiencies, under-design, damages, or failures, arising from the use, misuse, or adaptation of the methods, parameters, or formulas presented herein. Any attempt to implement or deploy the system in practice is done entirely at the user's own risk and responsibility.

- **Human Evaluation.**  
  All human evaluation conducted in this study was performed exclusively for the purpose of benchmarking and comparative analysis. Data generated by participants is kept strictly confidential, used only within the scope of this research, and will not be applied to any commercial, industrial, or real-world engineering activities.

- **Privacy and Data Protection.**  
  The dataset employed in this work is derived from production records within structural engineering consultancy practice but has been anonymized, cleaned, and used strictly for academic research. No identifying information, proprietary designs, or sensitive client data are disclosed. All outputs generated by the multi-agent system remain confined to research purposes and will not be released, applied, or repurposed for operational engineering, deployment, or commercial use. This research adheres to applicable data privacy principles and safeguards the rights and interests of all parties involved.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{liang2025automating,
  title={Automating Structural Engineering Workflows with Large Language Model Agents},
  author={Liang, Haoran and Zhou, Yufa and Kalaleh, Mohammad Talebi and Mei, Qipei},
  journal={arXiv preprint arXiv:2510.11004},
  year={2025}
}
```
