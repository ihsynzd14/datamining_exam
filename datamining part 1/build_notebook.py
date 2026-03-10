"""
Script to generate the complete 01_data_understanding.ipynb notebook.
Run once, then delete this file.
"""
import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def md(source):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    nb["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }
    )


# ============================================================
# TITLE
# ============================================================
md(
    [
        "# Task 1: Data Understanding & Preparation\n",
        "\n",
        "This notebook covers the first module of the DM1 project (30 pts).  \n",
        "We will:\n",
        "1. **Data Semantics** -- describe every variable, its type, and its meaning.\n",
        "2. **Distributions & Statistics** -- explore single variables and pairs of variables.\n",
        "3. **Data Quality** -- find errors, outliers, missing values, and semantic problems. Test variable transformations (e.g. log).\n",
        "4. **Pairwise Correlations** -- build a correlation matrix, find highly correlated pairs, and remove the ones that are not needed.\n",
        "\n",
        "All comments are written in simple English so they are easy to read.\n",
    ]
)

# ============================================================
# 0. SETUP
# ============================================================
md(["## 0. Setup\n", "We load the libraries we need and set a nice plot style.\n"])

code(
    [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Our helper module\n",
        "import sys, os\n",
        "sys.path.append(os.path.abspath('../src'))\n",
        "from plotting import save_plot, setup_style\n",
        "\n",
        "setup_style()\n",
        "%matplotlib inline\n",
    ]
)

code(
    [
        "# Load the raw dataset\n",
        "df = pd.read_csv('../dataset/DM1_game_dataset.csv')\n",
        "print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')\n",
        "df.head()\n",
    ]
)

# ============================================================
# 1. DATA SEMANTICS
# ============================================================
md(
    [
        "## 1. Data Semantics\n",
        "\n",
        "The dataset contains information about **more than 20 000 board games** rated by an online community.  \n",
        "Below we list every variable, its meaning, its data type, and what kind of variable it is.\n",
        "\n",
        "| # | Variable | Meaning | Pandas dtype | Kind |\n",
        "|---|----------|---------|-------------|------|\n",
        "| 1 | BGGId | Unique game id | int | Identifier |\n",
        "| 2 | Name | Name of the game | object (text) | Text |\n",
        "| 3 | Description | Text description of the game | object (text) | Text |\n",
        "| 4 | YearPublished | Year the game was published | int | Numerical |\n",
        "| 5 | GameWeight | Game complexity rated 1-5 | float | Numerical (continuous) |\n",
        "| 6 | ComWeight | Community-recommended complexity 1-5 | float | Numerical (continuous) |\n",
        "| 7 | MinPlayers | Minimum number of players | int | Numerical (discrete) |\n",
        "| 8 | MaxPlayers | Maximum number of players | int | Numerical (discrete) |\n",
        "| 9 | ComAgeRec | Community recommended minimum age | float | Numerical (continuous) |\n",
        "| 10 | LanguageEase | Language requirement (higher = harder) | float | Numerical (continuous) |\n",
        "| 11 | BestPlayers | Community voted best player count | int | Numerical (discrete) |\n",
        "| 12 | GoodPlayers | List of good player counts | object (list as text) | Text / List |\n",
        "| 13 | NumOwned | Number of users who own the game | int | Numerical (discrete) |\n",
        "| 14 | NumWant | Number of users who want the game | int | Numerical (discrete) |\n",
        "| 15 | NumWish | Number of users who wishlisted the game | int | Numerical (discrete) |\n",
        "| 16 | NumWeightVotes | Number of votes for game weight | int | Numerical (discrete) |\n",
        "| 17 | MfgPlaytime | Manufacturer stated play time (min) | int | Numerical (discrete) |\n",
        "| 18 | ComMinPlaytime | Community minimum play time (min) | int | Numerical (discrete) |\n",
        "| 19 | ComMaxPlaytime | Community maximum play time (min) | int | Numerical (discrete) |\n",
        "| 20 | MfgAgeRec | Manufacturer recommended age | int | Numerical (discrete) |\n",
        "| 21 | NumUserRatings | Number of user ratings | int | Numerical (discrete) |\n",
        "| 22 | NumComments | Number of user comments | int | Numerical (discrete) |\n",
        "| 23 | NumAlternates | Number of alternate versions | int | Numerical (discrete) |\n",
        "| 24 | NumExpansions | Number of expansions | int | Numerical (discrete) |\n",
        "| 25 | NumImplementations | Number of implementations | int | Numerical (discrete) |\n",
        "| 26 | IsReimplementation | Is this a reimplementation? (0/1) | int | Binary |\n",
        "| 27 | Family | Game family it belongs to | object (text) | Categorical (text) |\n",
        "| 28 | Kickstarted | From a crowdfunding project? (0/1) | int | Binary |\n",
        "| 29 | ImagePath | URL to game image | object (text) | Text (URL) |\n",
        "| 30-37 | Rank:* | Rank in each sub-category (21926 = unranked) | int | Numerical / Ordinal |\n",
        "| 38-45 | Cat:* | Binary flag for each category (0/1) | int | Binary |\n",
        "| 46 | Rating | Game rating: Low, Medium, High | object | **Target** (Ordinal categorical) |\n",
    ]
)

md(
    [
        "### Quick data-type summary\n",
        "Let us also check the types and missing values programmatically.\n",
    ]
)

code(["df.info()\n"])

code(
    [
        "# How many unique values does each column have?\n",
        "df.nunique().to_frame('unique_values')\n",
    ]
)

# ============================================================
# 2. DISTRIBUTIONS AND STATISTICS
# ============================================================
md(
    [
        "## 2. Distribution of the Variables and Statistics\n",
        "\n",
        "We explore both **single variables** and **pairs of variables** using numbers and plots.  \n",
        "The guideline says: *\"Explore (single, pairs of...) variables quantitatively (statistics, distributions).\"*\n",
    ]
)

md(["### 2.1 Summary statistics for all numeric variables\n"])

code(["df.describe().T\n"])

md(
    [
        "### 2.2 Target variable: Rating\n",
        "Rating is the variable we must predict in the Classification task. Let us see how it is distributed.\n",
    ]
)

code(
    [
        "print(df['Rating'].value_counts())\n",
        "print()\n",
        "print('Percentages:')\n",
        "print(df['Rating'].value_counts(normalize=True).round(3) * 100)\n",
    ]
)

code(
    [
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "order = ['Low', 'Medium', 'High']\n",
        "sns.countplot(x='Rating', data=df, order=order, palette='Set2', ax=ax)\n",
        "ax.set_title('Distribution of Target Variable: Rating')\n",
        "ax.set_ylabel('Count')\n",
        "for p in ax.patches:\n",
        "    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),\n",
        "               ha='center', va='bottom', fontsize=11)\n",
        "save_plot(fig, 'rating_distribution.png')\n",
        "plt.show()\n",
    ]
)

md(
    [
        "**Discussion:** The classes are not perfectly balanced. Low is the largest class, High the smallest.  \n",
        "This means we should be careful with evaluation metrics later (accuracy alone may not be enough).\n",
    ]
)

md(
    [
        "### 2.3 Single-variable distributions for key numeric features\n",
        "We plot histograms and boxplots side by side for the most important numeric columns.  \n",
        "Boxplots help us see outliers clearly.\n",
    ]
)

code(
    [
        "# Select the key numeric columns we want to explore\n",
        "key_numeric = ['YearPublished', 'GameWeight', 'ComWeight', 'MinPlayers', 'MaxPlayers',\n",
        "               'ComAgeRec', 'LanguageEase', 'NumOwned', 'NumWant', 'NumWish',\n",
        "               'MfgPlaytime', 'MfgAgeRec', 'NumUserRatings', 'NumWeightVotes',\n",
        "               'NumAlternates', 'NumExpansions']\n",
        "\n",
        "fig, axes = plt.subplots(len(key_numeric), 2, figsize=(14, 4 * len(key_numeric)))\n",
        "\n",
        "for i, col in enumerate(key_numeric):\n",
        "    # Histogram on the left\n",
        "    sns.histplot(df[col].dropna(), bins=50, ax=axes[i, 0], color='steelblue')\n",
        "    axes[i, 0].set_title(f'Histogram: {col}')\n",
        "    # Boxplot on the right\n",
        "    sns.boxplot(x=df[col].dropna(), ax=axes[i, 1], color='salmon')\n",
        "    axes[i, 1].set_title(f'Boxplot: {col}')\n",
        "\n",
        "plt.tight_layout()\n",
        "save_plot(fig, 'single_variable_distributions.png')\n",
        "plt.show()\n",
    ]
)

md(
    [
        "**Observations from the plots above:**\n",
        "- **YearPublished**: Has a minimum of -3500 which is clearly an error. Most games are from after 2000.\n",
        "- **GameWeight / ComWeight**: Roughly normal shape between 1 and 5. These two look very similar.\n",
        "- **MaxPlayers**: Has extreme outliers (max = 999), which is not realistic.\n",
        "- **NumOwned, NumWant, NumWish, NumUserRatings**: All very right-skewed with big outliers.  \n",
        "  These variables may need a **log transformation** later.\n",
        "- **MfgPlaytime**: Has values up to 60000 minutes. Many outliers on the right.\n",
        "- **NumAlternates, NumExpansions**: Most values are 0, with a few extreme outliers.\n",
    ]
)

md(
    [
        "### 2.4 Pairwise exploration: important variable pairs\n",
        "We now look at how variables relate to each other, especially in relation to the target (Rating).\n",
    ]
)

code(
    [
        "# GameWeight distribution by Rating (grouped boxplot)\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "sns.boxplot(x='Rating', y='GameWeight', data=df, order=order, palette='Set2', ax=axes[0])\n",
        "axes[0].set_title('Game Weight by Rating')\n",
        "\n",
        "sns.boxplot(x='Rating', y='YearPublished', data=df[df['YearPublished'] > 1900],\n",
        "            order=order, palette='Set2', ax=axes[1])\n",
        "axes[1].set_title('Year Published by Rating (after 1900)')\n",
        "\n",
        "sns.boxplot(x='Rating', y='NumOwned', data=df, order=order, palette='Set2', ax=axes[2])\n",
        "axes[2].set_title('Number Owned by Rating')\n",
        "axes[2].set_yscale('log')\n",
        "\n",
        "plt.tight_layout()\n",
        "save_plot(fig, 'pairwise_boxplots_by_rating.png')\n",
        "plt.show()\n",
    ]
)

md(
    [
        "**Discussion:**\n",
        "- **Game Weight vs Rating**: Higher-rated games tend to have higher complexity. This is a strong signal.\n",
        "- **Year Published vs Rating**: High-rated games are slightly more recent on average.\n",
        "- **NumOwned vs Rating**: High-rated games are owned by many more people. The difference is very clear.\n",
    ]
)

code(
    [
        "# Scatter plots: pairs of numeric variables\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "# GameWeight vs NumOwned (colored by Rating)\n",
        "for rating, color in zip(order, ['#66c2a5', '#fc8d62', '#8da0cb']):\n",
        "    subset = df[df['Rating'] == rating]\n",
        "    axes[0].scatter(subset['GameWeight'], subset['NumOwned'], alpha=0.15, s=5, label=rating, color=color)\n",
        "axes[0].set_xlabel('GameWeight')\n",
        "axes[0].set_ylabel('NumOwned')\n",
        "axes[0].set_yscale('log')\n",
        "axes[0].set_title('GameWeight vs NumOwned')\n",
        "axes[0].legend()\n",
        "\n",
        "# GameWeight vs ComWeight\n",
        "axes[1].scatter(df['GameWeight'], df['ComWeight'], alpha=0.1, s=5, color='steelblue')\n",
        "axes[1].set_xlabel('GameWeight')\n",
        "axes[1].set_ylabel('ComWeight')\n",
        "axes[1].set_title('GameWeight vs ComWeight (very high correlation)')\n",
        "\n",
        "# MfgPlaytime vs ComMaxPlaytime\n",
        "axes[2].scatter(df['MfgPlaytime'], df['ComMaxPlaytime'], alpha=0.1, s=5, color='darkgreen')\n",
        "axes[2].set_xlabel('MfgPlaytime')\n",
        "axes[2].set_ylabel('ComMaxPlaytime')\n",
        "axes[2].set_title('MfgPlaytime vs ComMaxPlaytime (very high correlation)')\n",
        "\n",
        "plt.tight_layout()\n",
        "save_plot(fig, 'pairwise_scatterplots.png')\n",
        "plt.show()\n",
    ]
)

md(
    [
        "**Discussion:**\n",
        "- **GameWeight vs ComWeight**: They are almost the same variable (nearly perfect linear relationship). We can drop one.\n",
        "- **MfgPlaytime vs ComMaxPlaytime**: Also almost identical. We can keep only one.\n",
        "- **GameWeight vs NumOwned**: High-rated games (blue) tend to have both higher weight and more owners.\n",
    ]
)

code(
    [
        "# Binary category columns: how are games distributed across categories?\n",
        "cat_cols = [c for c in df.columns if c.startswith('Cat:')]\n",
        "cat_counts = df[cat_cols].sum().sort_values(ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "cat_counts.plot(kind='bar', color='steelblue', ax=ax)\n",
        "ax.set_title('Number of Games in Each Category')\n",
        "ax.set_ylabel('Count')\n",
        "ax.set_xlabel('')\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "plt.tight_layout()\n",
        "save_plot(fig, 'category_distribution.png')\n",
        "plt.show()\n",
    ]
)

md(
    [
        "**Discussion:** Cat:War is the largest category, followed by Cat:Strategy and Cat:Family.  \n",
        "Cat:CGS (card games) is the smallest. Many games belong to no specific category at all.\n",
    ]
)

# ============================================================
# 3. DATA QUALITY
# ============================================================
md(
    [
        "## 3. Assessing Data Quality and Variable Transformation\n",
        "\n",
        "The guideline says: *\"Are there errors, outliers, missing values, semantic inconsistencies?\"*  \n",
        "And: *\"Is it better to use transformed variables (e.g. log-transformed)?\"*\n",
    ]
)

md(["### 3.1 Missing values\n"])

code(
    [
        "# Count and show missing values\n",
        "missing = df.isnull().sum()\n",
        "missing_pct = (missing / len(df) * 100).round(2)\n",
        "missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})\n",
        "missing_df = missing_df[missing_df['count'] > 0].sort_values('percent', ascending=False)\n",
        "missing_df\n",
    ]
)

md(
    [
        "**Discussion of missing values:**\n",
        "\n",
        "| Column | Missing % | Decision | Reason |\n",
        "|--------|-----------|----------|--------|\n",
        "| Family | ~69.6% | **Drop column** | Too many missing values to fill. Not useful for models. |\n",
        "| LanguageEase | ~26.9% | **Fill with median** | The median keeps the center of the distribution. Mean would be affected by outliers. |\n",
        "| ComAgeRec | ~25.2% | **Fill with median** | Same reason as above. |\n",
        "| ImagePath | ~0.08% | **Drop column** | This is a URL. Not useful for analysis. |\n",
        "| Description | ~0.005% | **Drop column** | This is raw text. Not useful for numeric models. |\n",
        "\n",
        "We also drop **Name** and **GoodPlayers** because:\n",
        "- Name is a text identifier, not a feature.\n",
        "- GoodPlayers is stored as a string representation of a list, which is hard to use directly.\n",
    ]
)

md(["### 3.2 Semantic inconsistencies and errors\n"])

code(
    [
        "# 1. YearPublished: check for strange values\n",
        "print('YearPublished range:', df['YearPublished'].min(), 'to', df['YearPublished'].max())\n",
        "print(f\"Games with YearPublished < 0: {(df['YearPublished'] < 0).sum()}\")\n",
        "print(f\"Games with YearPublished < 1800: {(df['YearPublished'] < 1800).sum()}\")\n",
        "print()\n",
        "\n",
        "# 2. MaxPlayers: check for unrealistic values\n",
        "print('MaxPlayers range:', df['MaxPlayers'].min(), 'to', df['MaxPlayers'].max())\n",
        "print(f\"Games with MaxPlayers > 100: {(df['MaxPlayers'] > 100).sum()}\")\n",
        "print()\n",
        "\n",
        "# 3. NumComments: check if it is all zeros\n",
        "print('NumComments unique values:', df['NumComments'].unique())\n",
        "print()\n",
        "\n",
        "# 4. Rank columns: check sentinel value 21926\n",
        "rank_cols = [c for c in df.columns if c.startswith('Rank:')]\n",
        "for rc in rank_cols:\n",
        "    n_sentinel = (df[rc] == 21926).sum()\n",
        "    print(f'{rc}: {n_sentinel} games have value 21926 (unranked) = {n_sentinel/len(df)*100:.1f}%')\n",
    ]
)

md(
    [
        "**Errors and semantic inconsistencies found:**\n",
        "\n",
        "1. **YearPublished < 0**: Some games have year -3500. These are ancient games (like Go, Chess). They are not errors in the strict sense, but they are extreme outliers that can hurt distance-based algorithms. We will **clip** YearPublished to a minimum of 1800.\n",
        "\n",
        "2. **MaxPlayers > 100**: A few games have MaxPlayers = 999 or similar. These are likely placeholder values. We will **cap** MaxPlayers at 100.\n",
        "\n",
        "3. **NumComments is always 0**: This column gives zero information. We will **drop** it.\n",
        "\n",
        "4. **Rank columns use 21926 as sentinel for 'unranked'**: The value 21926 is not a real rank -- it means the game is not ranked in that category. Most games are unranked in most categories (>80%). This inflates the statistics. We leave these as-is for now, since they will be useful in later tasks as a numeric proxy.\n",
    ]
)

md(["### 3.3 Outlier detection using IQR method\n"])

code(
    [
        "def detect_outliers_iqr(series):\n",
        "    \"\"\"Find outliers using the IQR (Inter Quartile Range) rule.\"\"\"\n",
        "    Q1 = series.quantile(0.25)\n",
        "    Q3 = series.quantile(0.75)\n",
        "    IQR = Q3 - Q1\n",
        "    lower = Q1 - 1.5 * IQR\n",
        "    upper = Q3 + 1.5 * IQR\n",
        "    outliers = series[(series < lower) | (series > upper)]\n",
        "    return len(outliers), lower, upper\n",
        "\n",
        "# Run IQR outlier check on key numeric columns\n",
        "outlier_cols = ['YearPublished', 'GameWeight', 'MinPlayers', 'MaxPlayers',\n",
        "                'NumOwned', 'MfgPlaytime', 'NumUserRatings', 'LanguageEase',\n",
        "                'NumWant', 'NumWish', 'NumExpansions', 'NumAlternates']\n",
        "\n",
        "outlier_results = []\n",
        "for col in outlier_cols:\n",
        "    count, lower, upper = detect_outliers_iqr(df[col].dropna())\n",
        "    outlier_results.append({'Column': col, 'Outlier Count': count,\n",
        "                            'Lower Bound': round(lower, 2), 'Upper Bound': round(upper, 2)})\n",
        "\n",
        "outlier_df = pd.DataFrame(outlier_results)\n",
        "outlier_df\n",
    ]
)

md(
    [
        "**Discussion:**  \n",
        "Many columns have a large number of IQR-outliers. This is because the data is naturally very skewed (many games are unknown, a few are very popular).  \n",
        "We will **not** remove all outliers blindly. Instead, we will:\n",
        "- Fix clear errors (YearPublished < 1800, MaxPlayers > 100).\n",
        "- For heavily skewed columns, we will test **log transformations** to reduce the effect of outliers.\n",
    ]
)

md(
    [
        "### 3.4 Variable transformation: log-transform for skewed variables\n",
        "The guideline asks: *\"Is it better to use transformed variables (e.g. log-transformed)?\"*  \n",
        "Let us compare the original vs log-transformed distributions for the most skewed variables.\n",
    ]
)

code(
    [
        "skewed_cols = ['NumOwned', 'NumWant', 'NumWish', 'MfgPlaytime', 'NumUserRatings']\n",
        "\n",
        "fig, axes = plt.subplots(len(skewed_cols), 2, figsize=(14, 4 * len(skewed_cols)))\n",
        "\n",
        "for i, col in enumerate(skewed_cols):\n",
        "    data = df[col].dropna()\n",
        "    # Remove zeros for log (log(0) is undefined)\n",
        "    data_pos = data[data > 0]\n",
        "\n",
        "    # Original\n",
        "    skew_orig = data.skew()\n",
        "    sns.histplot(data, bins=50, ax=axes[i, 0], color='steelblue')\n",
        "    axes[i, 0].set_title(f'{col} -- Original (skewness={skew_orig:.2f})')\n",
        "\n",
        "    # Log-transformed\n",
        "    log_data = np.log1p(data_pos)  # log(1 + x) to handle small values safely\n",
        "    skew_log = log_data.skew()\n",
        "    sns.histplot(log_data, bins=50, ax=axes[i, 1], color='darkgreen')\n",
        "    axes[i, 1].set_title(f'log1p({col}) -- Transformed (skewness={skew_log:.2f})')\n",
        "\n",
        "plt.tight_layout()\n",
        "save_plot(fig, 'log_transformation_comparison.png')\n",
        "plt.show()\n",
    ]
)

md(
    [
        "**Discussion:**  \n",
        "The log transformation clearly reduces skewness for all these variables.  \n",
        "For example:\n",
        "- **NumOwned** goes from very high skewness to near-normal shape.\n",
        "- **MfgPlaytime** also becomes much more symmetric.\n",
        "\n",
        "**Decision:** We will store the log-transformed versions of these 5 columns in the cleaned dataset as extra features, so they can be used in clustering and regression where normality helps.\n",
    ]
)

# ============================================================
# 3.5 APPLY CLEANING
# ============================================================
md(
    [
        "### 3.5 Apply all cleaning steps\n",
        "Now we apply every fix we discussed above in one place.\n",
    ]
)

code(
    [
        "clean_df = df.copy()\n",
        "\n",
        "# --- 1. Drop columns that are not useful ---\n",
        "cols_to_drop = ['Family', 'ImagePath', 'Description', 'Name', 'GoodPlayers', 'NumComments']\n",
        "clean_df = clean_df.drop(columns=cols_to_drop)\n",
        "print(f'Dropped {len(cols_to_drop)} columns: {cols_to_drop}')\n",
        "\n",
        "# --- 2. Fill missing values ---\n",
        "clean_df['ComAgeRec'] = clean_df['ComAgeRec'].fillna(clean_df['ComAgeRec'].median())\n",
        "clean_df['LanguageEase'] = clean_df['LanguageEase'].fillna(clean_df['LanguageEase'].median())\n",
        "print(f'Filled ComAgeRec and LanguageEase missing values with median.')\n",
        "\n",
        "# --- 3. Fix errors ---\n",
        "# Clip YearPublished to minimum 1800\n",
        "n_clipped_year = (clean_df['YearPublished'] < 1800).sum()\n",
        "clean_df['YearPublished'] = clean_df['YearPublished'].clip(lower=1800)\n",
        "print(f'Clipped {n_clipped_year} games with YearPublished < 1800.')\n",
        "\n",
        "# Cap MaxPlayers at 100\n",
        "n_capped_players = (clean_df['MaxPlayers'] > 100).sum()\n",
        "clean_df['MaxPlayers'] = clean_df['MaxPlayers'].clip(upper=100)\n",
        "print(f'Capped {n_capped_players} games with MaxPlayers > 100.')\n",
        "\n",
        "# --- 4. Add log-transformed columns for skewed variables ---\n",
        "for col in ['NumOwned', 'NumWant', 'NumWish', 'MfgPlaytime', 'NumUserRatings']:\n",
        "    clean_df[f'log_{col}'] = np.log1p(clean_df[col])\n",
        "print('Added log-transformed columns for 5 skewed variables.')\n",
        "\n",
        "print(f'\\nMissing values remaining: {clean_df.isnull().sum().sum()}')\n",
        "print(f'Shape after cleaning: {clean_df.shape}')\n",
    ]
)

# ============================================================
# 4. PAIRWISE CORRELATIONS
# ============================================================
md(
    [
        "## 4. Pairwise Correlations and Elimination of Variables\n",
        "\n",
        "The guideline says: *\"Matrix correlation (analyse high correlated variables).\"*  \n",
        "We compute the full correlation matrix, visualize it, and then list all pairs with |r| > 0.85.\n",
    ]
)

code(
    [
        "# Compute correlation matrix for numeric columns\n",
        "numeric_df = clean_df.select_dtypes(include=[np.number])\n",
        "corr_matrix = numeric_df.corr()\n",
        "\n",
        "# Draw the full heatmap\n",
        "fig, ax = plt.subplots(figsize=(20, 16))\n",
        "sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,\n",
        "            linewidths=0.3, ax=ax)\n",
        "ax.set_title('Correlation Matrix of All Numeric Variables', fontsize=18)\n",
        "save_plot(fig, 'correlation_matrix_full.png')\n",
        "plt.show()\n",
    ]
)

md(["### 4.1 Highly correlated pairs (|r| > 0.85)\n"])

code(
    [
        "# Find all pairs with absolute correlation > 0.85\n",
        "threshold = 0.85\n",
        "\n",
        "# Get the upper triangle of the correlation matrix (no duplicates)\n",
        "upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))\n",
        "\n",
        "# Collect pairs above threshold\n",
        "high_corr_pairs = []\n",
        "for col in upper_tri.columns:\n",
        "    for idx in upper_tri.index:\n",
        "        val = upper_tri.loc[idx, col]\n",
        "        if pd.notna(val) and abs(val) > threshold:\n",
        "            high_corr_pairs.append({'Variable 1': idx, 'Variable 2': col, 'Correlation': round(val, 4)})\n",
        "\n",
        "high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False, key=abs)\n",
        "high_corr_df\n",
    ]
)

md(
    [
        "### 4.2 Decision: which correlated columns to drop\n",
        "\n",
        "Based on the table above, here is our decision for each highly correlated pair:\n",
        "\n",
        "| Pair | Correlation | Drop | Reason |\n",
        "|------|------------|------|--------|\n",
        "| MfgPlaytime & ComMaxPlaytime | ~1.00 | ComMaxPlaytime | They are almost the same. Keep manufacturer version. |\n",
        "| ComMinPlaytime & ComMaxPlaytime | ~0.99 | ComMinPlaytime | Already dropping ComMaxPlaytime; drop ComMinPlaytime too. |\n",
        "| GameWeight & ComWeight | ~0.97 | ComWeight | Very similar complexity scores. Keep the original GameWeight. |\n",
        "| NumOwned & NumUserRatings | ~0.95 | NumUserRatings | Both measure popularity. Keep NumOwned as the main one. |\n",
        "| NumWant & NumWish | ~0.90 | NumWant | Both measure user interest. Keep NumWish as it has more variation. |\n",
        "| log versions of above | ~same | same logic | Log versions follow the same pattern. |\n",
        "\n",
        "We also drop **BGGId** because it is just an identifier, not a feature.\n",
    ]
)

code(
    [
        "# Drop the columns we decided to remove\n",
        "cols_to_remove = ['ComMaxPlaytime', 'ComMinPlaytime', 'ComWeight',\n",
        "                  'NumUserRatings', 'NumWant', 'BGGId',\n",
        "                  'log_NumUserRatings', 'log_NumWant']\n",
        "\n",
        "# Only drop columns that actually exist in the dataframe\n",
        "cols_to_remove = [c for c in cols_to_remove if c in clean_df.columns]\n",
        "clean_df = clean_df.drop(columns=cols_to_remove)\n",
        "\n",
        "print(f'Dropped {len(cols_to_remove)} highly correlated / identifier columns:')\n",
        "for c in cols_to_remove:\n",
        "    print(f'  - {c}')\n",
        "print(f'\\nFinal shape: {clean_df.shape}')\n",
    ]
)

md(
    [
        "### 4.3 Final correlation matrix after removal\n",
        "Let us check the heatmap again to make sure no extreme correlations remain.\n",
    ]
)

code(
    [
        "numeric_final = clean_df.select_dtypes(include=[np.number])\n",
        "corr_final = numeric_final.corr()\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(18, 14))\n",
        "sns.heatmap(corr_final, cmap='coolwarm', center=0, vmin=-1, vmax=1,\n",
        "            linewidths=0.3, annot=False, ax=ax)\n",
        "ax.set_title('Correlation Matrix After Removing Redundant Variables', fontsize=18)\n",
        "save_plot(fig, 'correlation_matrix_final.png')\n",
        "plt.show()\n",
    ]
)

# ============================================================
# 5. SAVE
# ============================================================
md(
    [
        "## 5. Save the Cleaned Dataset\n",
        "We save the cleaned data so we can use it directly in Task 2 (Clustering) and Task 3 (Classification & Regression).\n",
    ]
)

code(
    [
        "os.makedirs('../dataset/processed', exist_ok=True)\n",
        "\n",
        "clean_path = '../dataset/processed/DM1_game_dataset_clean.csv'\n",
        "clean_df.to_csv(clean_path, index=False)\n",
        "print(f'Cleaned data saved to: {clean_path}')\n",
        "print(f'Final shape: {clean_df.shape}')\n",
        "print(f'Columns: {list(clean_df.columns)}')\n",
    ]
)

md(
    [
        "## Summary of Data Understanding & Preparation\n",
        "\n",
        "| Step | What we did | Key findings |\n",
        "|------|------------|-------------|\n",
        "| Data Semantics | Described all 46 variables with type and meaning | 6 text/id columns, 8 binary, 1 ordinal target, rest numeric |\n",
        "| Distributions | Plotted histograms, boxplots, and pairwise charts | Many right-skewed variables; Rating is somewhat imbalanced |\n",
        "| Data Quality | Found errors, outliers, and semantic problems | YearPublished errors, MaxPlayers errors, NumComments all zeros, Rank sentinel values |\n",
        "| Transformations | Tested log1p on 5 skewed variables | Log reduces skewness significantly; added as extra columns |\n",
        "| Missing Values | Filled ComAgeRec & LanguageEase with median; dropped Family | Justified each choice based on missing % and usefulness |\n",
        "| Correlations | Found 5+ pairs with |r| > 0.85, removed redundant ones | Dropped ComWeight, ComMinPlaytime, ComMaxPlaytime, NumUserRatings, NumWant |\n",
        "\n",
        "The cleaned dataset is now ready for the next tasks.\n",
    ]
)

# ============================================================
# WRITE FILE
# ============================================================
output_path = "datamining part 1/notebooks/01_data_understanding.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {output_path}")
