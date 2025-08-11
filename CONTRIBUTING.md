# 🤝 Contributing to RedDust Reclaimer

Thank you for your interest in contributing to Mars terraforming research! This guide will help you get started.

## 🚀 Quick Start for Contributors

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/reddust-reclaimer.git
cd reddust-reclaimer
```

### 2. Set Up Development Environment
```bash
# Option 1: Conda (Recommended)
conda env create -f environment.yml
conda activate reddust-reclaimer

# Option 2: Pip
pip install -r requirements.txt
pip install -e .
```

### 3. Run Tests
```bash
pytest tests/ -v
python scripts/dock_example.py --help
```

## 🛠️ Development Workflow

### Before Making Changes
1. **Check existing issues** for similar work
2. **Create an issue** to discuss major changes
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`

### Code Standards
- **Python 3.9+** with type hints
- **Google-style docstrings**
- **Black formatting**: `black scripts/ tests/`
- **Flake8 linting**: `flake8 scripts/ tests/`
- **Type checking**: `mypy scripts/`

### Testing Requirements
- **Unit tests** for all new functions
- **Integration tests** for workflows
- **Coverage >80%**: `pytest --cov=scripts`
- **Example scripts** must run without errors

## 🧬 Contribution Areas

### 🔬 Molecular Docking
- New docking algorithms for Mars conditions
- Additional protein targets
- Improved scoring functions
- 3D visualization enhancements

### 🧪 Metabolic Modeling
- Extended SBML models
- New metabolic pathways
- Flux variability analysis
- Multi-objective optimization

### 🧬 Genetic Engineering
- Advanced codon optimization
- Synthetic biology tools
- Gene circuit design
- Regulatory network analysis

### 📊 Data and Visualization
- New Mars datasets
- Interactive visualizations
- Dashboard development
- Performance optimization

## 📝 Code Review Process

### Pull Request Guidelines
1. **Descriptive title** and detailed description
2. **Link related issues**: `Closes #123`
3. **Small, focused changes** (prefer multiple small PRs)
4. **Tests included** for new functionality
5. **Documentation updated** as needed

### Review Criteria
- ✅ Code quality and style
- ✅ Test coverage and functionality
- ✅ Documentation completeness
- ✅ Mars terraforming relevance
- ✅ Performance impact
- ✅ Breaking change assessment

## 🐛 Bug Reports

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) with:
- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Environment details** (OS, Python version, etc.)
- **Error logs** and stack traces

## 💡 Feature Requests

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml) with:
- **Use case description** for Mars research
- **Proposed implementation** approach
- **Priority level** and justification

## 🏷️ Issue Labels

### Type Labels
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `question`: General questions and support

### Priority Labels
- `critical`: Blocks research progress
- `high`: Significantly improves workflow
- `medium`: Nice to have improvement
- `low`: Minor enhancement

### Area Labels
- `docking`: Molecular docking components
- `metabolic-modeling`: Flux analysis and SBML
- `genetic-engineering`: Codon optimization and synthetic biology
- `infrastructure`: CI/CD, Docker, deployment

## 🧪 Testing Guidelines

### Unit Tests
```python
def test_mars_docking_basic():
    """Test basic docking functionality"""
    docker = MarsEnzymeDocking()
    ligands = docker.generate_mars_ligands()
    assert len(ligands) > 0
```

### Integration Tests
```python
def test_full_workflow():
    """Test complete Mars analysis workflow"""
    # Test docking -> optimization -> flux analysis pipeline
    pass
```

### Performance Tests
```python
def test_performance_benchmarks():
    """Ensure algorithms meet performance requirements"""
    # Docking: < 5 minutes for example
    # Optimization: < 1 minute for 1kb sequence
    # Flux analysis: < 30 seconds
    pass
```

## 📚 Documentation

### Docstring Format (Google Style)
```python
def mars_function(param1: str, param2: int = 42) -> bool:
    """Brief description of function.
    
    Longer description explaining Mars terraforming relevance
    and algorithm details.
    
    Args:
        param1: Description of first parameter
        param2: Description with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
        
    Example:
        >>> result = mars_function("CO2", 273)
        >>> print(result)
        True
    """
    pass
```

### README Updates
- Keep examples **current and working**
- Update **installation instructions** as needed
- Add **new features** to feature list
- Update **roadmap** with completed items

## 🌟 Recognition

Contributors are recognized in:
- **README.md** contributors section
- **Release notes** for major contributions
- **Academic citations** for research contributions
- **Special thanks** in documentation

## 📞 Getting Help

- **GitHub Discussions**: General questions and ideas
- **Issues**: Bug reports and feature requests
- **Email**: maintainer@reddust-reclaimer.org
- **Discord**: [Mars Terraforming Research Community]

## 🎯 Contribution Goals

We especially welcome contributions that:
- **Improve Mars relevance** of existing algorithms
- **Add new biological datasets** from Mars missions
- **Enhance performance** for large-scale simulations
- **Improve user experience** and documentation
- **Add educational content** for astrobiology students

---

**🚀 Ready to help terraform Mars? Start by picking an issue labeled `good-first-issue`!**
