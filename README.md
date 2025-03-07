<h1 align="center">
	🦉 OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation
</h1>


<div align="center">

[![Documentation][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![Wechat][wechat-image]][wechat-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Star][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]

</div>


<hr>

<div align="center">
<h4 align="center">

[Community](https://github.com/camel-ai/camel#community) |
[Installation](https://github.com/camel-ai/camel#installation) |
[Examples](https://github.com/camel-ai/camel/tree/HEAD/examples) |
[Paper](https://arxiv.org/abs/2303.17760) |
[Citation](https://github.com/camel-ai/camel#citation) |
[Contributing](https://github.com/camel-ai/camel#contributing-to-camel-) |
[CAMEL-AI](https://www.camel-ai.org/)

</h4>

<p style="line-height: 1.5; text-align: center;"> 🦉 OWL is a cutting-edge framework for multi-agent collaboration that pushes the boundaries of task automation, built on top of the <a href="https://github.com/camel-ai/camel">CAMEL-AI Framework</a>. 
OWL achieves 58.18 average score on GAIA benchmark and ranks #1 among open-source frameworks.
Our vision is to revolutionize how AI agents collaborate to solve real-world tasks. By leveraging dynamic agent interactions, OWL enables more natural, efficient, and robust task automation across diverse domains.</p>
<br>


</div>

<!-- # Key Features -->
<!-- # 📋 Table of Contents -->




# 🔥 News

- **[2025.03.07]**: We open-source the codebase of 🦉 OWL project.

# 🛠️ Installation

## **Clone the Github repository**

```bash
git clone https://github.com/camel-ai/owl.git
cd owl
```

## **Set up Environment**

Using Conda (recommended):
```bash
conda create -n owl python=3.11
conda activate owl
```

Using venv (alternative):
```bash
python -m venv owl_env
# On Windows
owl_env\Scripts\activate
# On Unix or MacOS
source owl_env/bin/activate
```

## **Install Dependencies**

```bash
python -m pip install -r requirements.txt
```

## **Setup Environment Variables** 

In the `owl/.env_example` file, you will find all the necessary API keys along with the websites where you can register for each service. To use these API services, follow these steps:

1. *Copy and Rename*: Duplicate the `.env_example` file and rename the copy to `.env`.
2. *Fill in Your Keys*: Open the `.env` file and insert your API keys in the corresponding fields. 

# 🚀 Quick Start
   
Run the following minimal example:

```bash
python owl/run.py
```

# 🧪 Experiments

We provided a script to reproduce the results on GAIA. 
You can check the `run_gaia_roleplaying.py` file and run the following command:

```bash
python run_gaia_roleplaying.py
```

# ⏱️ Future Plans

- [ ] Write a technical blog post detailing our exploration and insights in multi-agent collaboration in real-world tasks.
- [ ] Enhance the toolkit ecosystem with more specialized tools for domain-specific tasks.
- [ ] Develop more sophisticated agent interaction patterns and communication protocols

<!-- # Architecture



# Cite -->

# 📄 License

The source code is licensed under Apache 2.0.


[docs-image]: https://img.shields.io/badge/Documentation-EB3ECC
[docs-url]: https://camel-ai.github.io/camel/index.html
[star-image]: https://img.shields.io/github/stars/camel-ai/owl?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/camel/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/camel/blob/master/licenses/LICENSE

[colab-url]: https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing
[colab-image]: https://colab.research.google.com/assets/colab-badge.svg
[huggingface-url]: https://huggingface.co/camel-ai
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[discord-url]: https://discord.camel-ai.org/
[discord-image]: https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb
[wechat-url]: https://ghli.org/camel/wechat.png
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[x-url]: https://x.com/CamelAIOrg
[x-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social
[twitter-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social&color=brightgreen&logo=twitter
[reddit-url]: https://www.reddit.com/r/CamelAI/
[reddit-image]: https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white
[ambassador-url]: https://www.camel-ai.org/community
