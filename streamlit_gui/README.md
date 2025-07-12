# CodonTransformer Streamlit GUI

A user-friendly web interface for the CodonTransformer model that allows researchers to optimize protein sequences for codon usage with real-time analysis and visualization.

## Features

### 🔬 Real-Time Analysis
- **Input Validation**: Automatic protein sequence validation with helpful error messages
- **Baseline Metrics**: Calculate GC content, CAI, and tAI for input sequences
- **Live Feedback**: See metrics update as you type

### 🚀 Advanced Optimization
- **Constrained Beam Search**: Use the improved beam search algorithm with GC content control
- **Multiple Organisms**: Support for E. coli, S. cerevisiae, H. sapiens, and more
- **Customizable Parameters**: Adjust GC content targets, beam size, and other parameters

### 📊 Comprehensive Visualizations
- **GC Content Plots**: Sliding window analysis showing GC content distribution
- **Before/After Comparison**: Side-by-side metrics comparison
- **Interactive Charts**: Plotly-powered visualizations for better insights

### 💾 Export & Download
- **Multiple Formats**: Export optimized sequences in various formats
- **One-Click Download**: Easy download of results
- **Batch Processing**: Support for multiple sequences (coming soon)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, but CPU works too)
- At least 8GB RAM (16GB recommended for longer sequences)

### Quick Start

1. **Clone the repository** (if you haven't already):
```bash
git clone <repository-url>
cd ecoli/streamlit_gui
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Launch the GUI**:
```bash
python run_gui.py
```

The application will automatically open in your web browser at `http://localhost:8501`

## Usage

### Basic Workflow

1. **Enter Protein Sequence**: 
   - Paste your protein sequence in single-letter amino acid format
   - The system validates the sequence and shows basic metrics

2. **Configure Settings**:
   - Select target organism from the dropdown
   - Choose optimization method (standard or constrained beam search)
   - Adjust GC content targets if using constrained search

3. **Run Optimization**:
   - Click "Optimize Sequence" to start the process
   - Watch real-time progress and status updates

4. **Analyze Results**:
   - Compare before/after metrics
   - View GC content distribution plots
   - Download optimized sequences

### Advanced Features

#### Constrained Beam Search
- **GC Content Control**: Set min/max GC content percentages
- **Beam Size**: Adjust search breadth (larger = more thorough but slower)
- **Organism-Specific**: Optimizes for organism-specific codon usage patterns

#### Visualization Options
- **Sliding Window GC Plot**: Shows GC content variation across the sequence
- **Metrics Comparison**: Bar charts comparing input vs optimized metrics
- **Real-time Updates**: All visualizations update automatically

## Input Requirements

### Protein Sequence Format
- **Single-letter amino acids**: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- **Stop codons**: * or _
- **Length**: 3-500 amino acids (demo limitation)
- **Case insensitive**: Automatically converted to uppercase

### Supported Organisms
- Escherichia coli general
- Saccharomyces cerevisiae  
- Homo sapiens
- Bacillus subtilis
- Pichia pastoris

## Example Sequences

### Short Peptide (10 AA)
```
MKTVRQERLK
```

### Medium Protein (60 AA)
```
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA
```

### Long Protein (175 AA)
```
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
```

## Performance Notes

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for sequences >100 AA
- **Memory**: 
  - CPU: 8GB RAM minimum
  - GPU: 6GB VRAM minimum
- **Storage**: 5GB free space for model downloads

### Processing Times (Approximate)
- **Short sequences** (10-50 AA): 5-15 seconds
- **Medium sequences** (50-150 AA): 15-45 seconds  
- **Long sequences** (150-400 AA): 45-120 seconds

## Troubleshooting

### Common Issues

#### Model Loading Errors
```
Error: Model failed to load
```
**Solution**: Ensure you have stable internet connection for initial model download (~2GB)

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce beam size in constrained search
- Use CPU instead of GPU
- Process shorter sequences
- Restart the application

#### Slow Performance
**Solutions**:
- Use GPU if available
- Reduce beam size for faster (but less thorough) optimization
- Close other applications to free up memory

#### Invalid Sequence Error
```
Invalid characters found: X, B, Z
```
**Solution**: Remove invalid characters. Only standard amino acids are supported.

### Getting Help

If you encounter issues:
1. Check the console/terminal for detailed error messages
2. Verify your protein sequence format
3. Ensure all dependencies are installed correctly
4. Try with a shorter test sequence first

## Technical Details

### Model Information
- **Fine-tuned Model**: `saketh11/ColiFormer` - Enhanced with ALM training (6.2% better CAI)
- **Training Dataset**: `saketh11/ColiFormer-Data` - 4,300 high-CAI E. coli sequences
- **Base Model**: BigBird Transformer architecture
- **Training Data**: Large-scale protein-DNA sequence pairs
- **Optimization**: Multi-objective optimization for codon usage, GC content, and expression
- **Automatic Download**: Model is automatically downloaded from Hugging Face Hub on first use
- **Fallback**: If the fine-tuned model fails to load, falls back to base model

### Metrics Calculated
- **GC Content**: Percentage of G and C nucleotides
- **CAI**: Codon Adaptation Index (organism-specific codon usage)
- **tAI**: tRNA Adaptation Index (tRNA availability)

### Files Structure
```
streamlit_gui/
├── app.py              # Main Streamlit application
├── run_gui.py          # Launcher script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Development

### Running in Development Mode
```bash
streamlit run app.py --server.runOnSave true
```

### Adding New Features
The GUI is built with modular components:
- Input validation in `validate_protein_sequence()`
- Metrics calculation in `calculate_input_metrics()`
- Visualization in `create_*_plot()` functions
- Model prediction in `run_optimization()`

## License

This project is licensed under the same terms as the main CodonTransformer repository.

## Citation

If you use this GUI in your research, please cite the original CodonTransformer paper:

```bibtex
@article{codon_transformer,
  title={CodonTransformer: Deep Learning for Codon Usage Optimization},
  author={...},
  journal={...},
  year={2024}
}
```

---

**Happy Optimizing!** 🧬✨