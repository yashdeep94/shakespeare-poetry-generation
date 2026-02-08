# Shakespeare Poetry Generation: A Comparative Study

A comprehensive neural network project exploring different architectures for generating Shakespearean poetry, progressing from RNNs to modern Transformer and Diffusion models.

## 📋 Project Overview

This project was developed as part of a Neural Networks course and demonstrates the evolution of text generation approaches. The work is divided into two phases:

**Phase 1**: Implemented RNN-based poetry generation using static GloVe embeddings.

**Phase 2**: Advanced to Transformer and Diffusion-based architectures with contextualized BERT embeddings, significantly improving semantic coherence.

### Key Achievement
The transition from static (GloVe) to contextualized (BERT) embeddings resulted in generated poems that prioritize **meaningful content** over simple rhyme patterns, producing more coherent and semantically rich Shakespearean-style verse.

## Project Structure

```
├── shakespeare_rnn_glove.ipynb           # RNN with GloVe embeddings
├── shakespeare_transformer_bert.ipynb    # Transformer with BERT embeddings
├── shakespeare_diffusion_bert.ipynb      # Diffusion model with BERT embeddings
├── data/                                    # Shakespeare text datasets
├── weights/                                 # Trained model weights
└── README.md
```

## Required Files Not Included
Due to file size, the following files are not included in this repository:
- GloVe Embeddings
- Model Weights (~1000MB total)

## Models Implemented

### 1. RNN with GloVe Embeddings
- **Architecture**: Recurrent Neural Network
- **Embeddings**: Pre-trained GloVe (static word embeddings)
- **Focus**: Learning sequential patterns and rhyme structures
- **Outcome**: Generated poetry with good rhythm but limited semantic coherence

### 2. Transformer with BERT Embeddings
- **Architecture**: Transformer-based sequence model
- **Embeddings**: BERT contextualized embeddings
- **Focus**: Capturing long-range dependencies and contextual meaning
- **Outcome**: Significantly improved semantic coherence and meaning

### 3. Diffusion Model with U-Net and BERT Embeddings - Experimental
- **Architecture**: U-Net based diffusion model (adapted from image generation to text)
- **Embeddings**: BERT contextualized embeddings
- **Focus**: Experimental approach - applying U-Net architecture (traditionally used for images) to learn text representations
- **Outcome**: Novel cross-domain approach exploring whether vision-based diffusion architectures can generate coherent poetry

## Model Weights

Pre-trained model weights (~1000MB) are not included in this repository.

**Options to access weights:**
1. **Train yourself**: Run the notebooks (training details in each notebook)
2. **Contact me**: Available upon request

## Technical Details

### Training Process
- Extensive **hyperparameter tuning** for optimal performance
- Custom training pipelines implemented from scratch
- Appropriate **epoch selection** and early stopping to prevent overfitting
- Model convergence monitoring and validation

### Technologies Used
- Python
- PyTorch / TensorFlow
- GloVe embeddings
- BERT (Transformers library)
- NumPy, Pandas
- Jupyter Notebooks

## Getting Started

### Prerequisites
```bash
pip install torch transformers numpy pandas
```

### Usage
1. Clone the repository
2. Ensure the `data/` folder contains Shakespeare text corpus
3. Run notebooks sequentially to see the progression:
   - Start with `shakespeare_rnn_glove.ipynb`
   - Progress to `shakespeare_transformer_bert.ipynb`
   - Explore advanced approach in `shakespeare_diffusion_bert.ipynb`

### Trained Weights
Pre-trained model weights are stored in the `weights/` directory and can be loaded directly for inference without retraining.

## Key Findings

1. **Embedding Quality Matters**: Contextualized BERT embeddings dramatically outperformed static GloVe embeddings in generating semantically coherent poetry.

2. **Architecture Evolution**: Progressing from RNNs to Transformers showed clear improvements in capturing long-range dependencies in poetic structure.

3. **Experimental U-Net Diffusion**: Demonstrated innovative thinking by adapting U-Net architecture (originally designed for image segmentation/generation) to the text domain, exploring cross-modal architecture applications.

## Learning Outcomes

- Deep understanding of RNN, Transformer, and Diffusion architectures
- Practical experience with embedding techniques (static vs. contextualized)
- Hyperparameter optimization and model training best practices
- Comparative analysis of different neural network approaches
- End-to-end implementation of NLP generative models

## Future Improvements

- Fine-tune BERT specifically on Shakespearean corpus
- Implement attention visualization for interpretability
- Add evaluation metrics (BLEU, perplexity, human evaluation)
- Explore few-shot learning approaches
- Develop interactive web interface for poetry generation

## Course Context

This project was completed as part of a comprehensive Neural Networks course, demonstrating proficiency in:
- Building models from scratch
- Training and optimizing deep learning architectures
- Comparative analysis of different approaches
- Practical application of theoretical concepts

## License

This project is available for educational and research purposes.

## Acknowledgments

- Shakespeare corpus for training data
- Pre-trained GloVe and BERT embeddings
- Neural Networks course instructors and materials