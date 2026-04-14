"""
Generate synthetic fake news dataset for local development and testing.

Creates realistic-looking propagation features without requiring
BigQuery or the Twitter API. Useful for:
- Testing the full pipeline locally
- Demo / portfolio showcase
- CI/CD test fixtures

Usage:
    python scripts/generate_sample_data.py --n 5000 --output data/raw/fake_news.csv
"""

import click
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def generate_cascade_features(n: int, label: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate propagation features that mimic real fake/real news patterns.

    Based on research insights:
    - Fake news tends to spread faster, wider, but shallower
    - Real news has more consistent, slower spread with deeper cascades
    - Fake news has higher burstiness in early propagation
    """
    if label == 1:  # Fake news
        retweet_count = rng.lognormal(mean=5.5, sigma=1.8, size=n).astype(int)
        favorite_count = rng.lognormal(mean=3.5, sigma=1.5, size=n).astype(int)
        followers_count = rng.lognormal(mean=6.0, sigma=2.0, size=n).astype(int)
        friends_count = (followers_count * rng.uniform(0.5, 2.0, n)).astype(int)
        cascade_size = rng.lognormal(mean=4.0, sigma=1.5, size=n).astype(int)
        cascade_depth = rng.integers(1, 8, size=n)
        cascade_max_breadth = (cascade_size * rng.uniform(0.3, 0.7, n)).astype(int)
        spread_speed_mean = rng.exponential(scale=30, size=n)  # Faster spread
        spread_speed_std = spread_speed_mean * rng.uniform(1.0, 3.0, n)  # High variance
        hour_bias = rng.choice([0, 1, 2, 3, 22, 23], size=n)  # Late night posting
    else:  # Real news
        retweet_count = rng.lognormal(mean=4.5, sigma=1.5, size=n).astype(int)
        favorite_count = rng.lognormal(mean=4.0, sigma=1.2, size=n).astype(int)
        followers_count = rng.lognormal(mean=7.0, sigma=1.5, size=n).astype(int)
        friends_count = (followers_count * rng.uniform(0.1, 0.8, n)).astype(int)
        cascade_size = rng.lognormal(mean=3.5, sigma=1.2, size=n).astype(int)
        cascade_depth = rng.integers(2, 15, size=n)
        cascade_max_breadth = (cascade_size * rng.uniform(0.1, 0.4, n)).astype(int)
        spread_speed_mean = rng.exponential(scale=120, size=n)  # Slower spread
        spread_speed_std = spread_speed_mean * rng.uniform(0.3, 1.0, n)  # Lower variance
        hour_bias = rng.choice(range(8, 20), size=n)  # Business hours

    # Ensure minimums
    cascade_size = np.maximum(cascade_size, 1)
    cascade_depth = np.maximum(cascade_depth, 1)
    cascade_max_breadth = np.maximum(cascade_max_breadth, 1)
    followers_count = np.maximum(followers_count, 1)
    friends_count = np.maximum(friends_count, 1)

    # Generate timestamps across 2024
    base_date = pd.Timestamp("2024-01-01")
    random_days = rng.integers(0, 365, size=n)
    random_seconds = rng.integers(0, 86400, size=n)
    created_at = [
        base_date + pd.Timedelta(days=int(d), seconds=int(s))
        for d, s in zip(random_days, random_seconds)
    ]

    return pd.DataFrame({
        "tweet_id": [f"tw_{label}_{i:06d}" for i in range(n)],
        "text": [f"Sample {'fake' if label == 1 else 'real'} news text #{i}" for i in range(n)],
        "user_id": [f"user_{rng.integers(1000, 99999)}" for _ in range(n)],
        "retweet_count": retweet_count,
        "favorite_count": favorite_count,
        "followers_count": followers_count,
        "friends_count": friends_count,
        "created_at": created_at,
        "cascade_size": cascade_size,
        "cascade_depth": cascade_depth,
        "cascade_max_breadth": cascade_max_breadth,
        "spread_speed_mean": np.round(spread_speed_mean, 2),
        "spread_speed_std": np.round(spread_speed_std, 2),
        "label": label,
        "ingested_at": pd.Timestamp.now(),
    })


@click.command()
@click.option("--n", default=5000, help="Total number of samples.")
@click.option("--fake-ratio", default=0.45, help="Proportion of fake news (0-1).")
@click.option("--output", default="data/raw/fake_news.csv", help="Output CSV path.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def main(n: int, fake_ratio: float, output: str, seed: int):
    """Generate synthetic fake news propagation dataset."""
    rng = np.random.default_rng(seed)

    n_fake = int(n * fake_ratio)
    n_real = n - n_fake

    logger.info(f"Generating {n} samples: {n_fake} fake + {n_real} real")

    fake_df = generate_cascade_features(n_fake, label=1, rng=rng)
    real_df = generate_cascade_features(n_real, label=0, rng=rng)

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

    # Save
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    logger.info(f"Saved {len(df)} rows to {output}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"\nSample:\n{df.head(3).to_string()}")


if __name__ == "__main__":
    main()
