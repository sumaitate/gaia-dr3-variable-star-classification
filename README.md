# Machine Learning Classification of Variable Stars Using Gaia DR3 Light Curves

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Research Question**: Can classical machine-learning models accurately classify common variable-star types using features extracted from Gaia DR3 photometric light curves, and which features contribute most strongly to the classification?

##  Data Sources
**Gaia DR3 Archive**
	https://gea.esac.esa.int/archive/
	Gaia DR3 provides variable-star classifications and photometric time-series data that can be used for this project.

**Secondary Dataset**:
**ZTF_40k Variable-Star Dataset**
	https://huggingface.co/datasets/StarEmbed/ZTF_40k
	This dataset contains labeled ZTF variable-star light curves and can be used later for comparison, additional testing, or extension of the Gaia project.

## Reference Code
Gaia DR3 Time-Domain Example
	https://github.com/anilipour/Gaia-DR3-Time-Domain-SETI
	Useful for understanding how to access and analyze Gaia DR3 time-domain/light-curve data.

StarEmbed
	https://github.com/skai-institute/StarEmbed
	Useful reference for machine-learning analysis of astronomical light curves.
	
Variable-Star Classifier
	https://github.com/Vector-Pi/varstar-classifier
	A useful beginner-friendly example of feature extraction and variable-star classification.
	
SCoPe
	https://github.com/scope-ml/scope-ml
	A more advanced reference for machine-learning classification of ZTF astronomical sources.


## Working Repository Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         gaia_variable_stars and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── gaia_variable_stars   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes gaia_variable_stars a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

