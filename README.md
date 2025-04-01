# LLM-enhanced-competitive-coding-algorithm-retrieval

The repo here only shows the code for fine-tuning the microsoft/unixcoder model for specific programming language algorithm classification for competitive programming solutions.

The dataset is from CodeForces. The dataset is also now available on huggingface
https://huggingface.co/datasets/AlbertHatsuki/codeforces-question-solution-label

For the usage of the codes. Please run pip install in the terminal:

```markdown
pip install -r requirements.txt
```
And for the models, it is estimated that each will cost about 100GB storage. And please change the language in the fine-tuning.py to train different languange-based models.

For cuda based training, please change the device from mps to cuda.
