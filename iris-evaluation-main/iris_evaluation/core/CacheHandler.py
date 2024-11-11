"""
Written by Rob Hart of Walsh Lab @ IU Indianapolis.
"""

import os
import numpy as np
import polars as pl
import seaborn as sns
import cv2
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats


class CacheHandler:
    def __init__(self, target, verbose=False, self_comparison=False):
        self.target = target
        self.path = (f'{target}/.cache.tsv', f'{target}/hamming.parquet', f'{target}/hamming.csv')
        self.self_comparison = self_comparison #boolean indicating whether or not comparisons are intradataset
        self.verbose = verbose
        self.clear()
        
    def new_line(self, comparison, result, SSIM = False):
        with open(self.path[0], 'a') as file:
            if self.verbose and not self.self_comparison:
                file.write(f'{comparison}\t{result}\t{SSIM}\n')
            else:
                file.write(f'{comparison}\t{result}\n')

    def clear(self):
        if os.path.exists(self.path[0]):
            os.remove(self.path[0])

class PairwiseCache(CacheHandler):
    def save(self):
        data = self.__get_all()
        data[0].write_parquet(self.path[1])
        data[0].write_csv(self.path[2])
        if self.verbose:
            np_path = self.path[1].replace(".parquet", ".npy")
            np.save(np_path, data[1])
            with open(self.path[1].replace("hamming.parquet", "summary_stats.txt"), 'w') as file:
                file.write(f"Mean: {data[2]['mean']}\nStdev: {data[2]['stdev']}\n")

            mean = np.mean(data[1])
            stdev = np.std(data[1])
            n = len(data[1])
            stderr = stdev / np.sqrt(n)
            ci_low, ci_high = stats.norm.interval(0.95, loc=mean, scale=stderr)

            colors = sns.color_palette("colorblind")
            plt.figure(figsize=(7.16, 5.37))
   
            sns.kdeplot(data[1], color=colors[0], linewidth=2.5)
            plt.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
            plt.axvline(mean - stdev, color='g', linestyle='--', linewidth=2, label=f'-1 SD: {mean - stdev:.4f}')
            plt.axvline(mean + stdev, color='g', linestyle='--', linewidth=2, label=f'+1 SD: {mean + stdev:.4f}')
            plt.axvline(ci_low, color='b', linestyle='-.', linewidth=2, label=f'95% CI Low: {ci_low:.4f}')
            plt.axvline(ci_high, color='b', linestyle='-.', linewidth=2, label=f'95% CI High: {ci_high:.4f}')

            plt.title('Hamming Distance Histogram with Confidence Intervals', fontsize=16)
            plt.xlabel('Hamming Distance', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.xlim(0.3, 0.6)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.legend()

            plt.savefig(self.path[1].replace(".parquet", ".png"), dpi=300)
            plt.clf()

    def __import(self):
        df = pl.read_csv(self.path[0], has_header=False, separator='\t', new_columns=['comparison', 'HD'], glob=False)
        #df = df.unique(subset=["comparison"])
        df = df.with_columns(pl.col("comparison").str.split_exact("|", 1).struct.rename_fields(["img1", "img2"]).alias("fields")).unnest("fields")
        #if self.self_comparison:
        #    df = df.filter(pl.col("img1") != pl.col("img2"))

        return df

    cache = property(fget = __import)

    def __get_hamming_distances(self):
        df = self.__import()
        hamming_distances = df["HD"].to_numpy()

        return hamming_distances
    
    hamming_distances = property(fget = __get_hamming_distances)

    def __get_stats(self):
        hamming_distances = self.__get_hamming_distances()
        hamming_stats = {
            "mean": np.mean(hamming_distances),
            "stdev": np.std(hamming_distances),
        }
        
        return hamming_stats
    
    hamming_stats = property(fget = __get_stats)
    
    def __get_all(self): #probably not the most efficient way to do this, but it is what it is.
        df = self.__import()
        hamming_distances = df["HD"].to_numpy()
        hamming_stats = {
            "mean": np.mean(hamming_distances),
            "stdev": np.std(hamming_distances),
        }

        all_data = tuple([df, hamming_distances, hamming_stats])

        return all_data

    all_data = property(fget = __get_all)

class LinearCache(CacheHandler):
    def save(self):
        df = self.__import()
        path = self.path[1]
        df.write_parquet(path)

    def __import(self):
        df = pl.read_csv(self.path, has_header=False, separator='\t', new_columns=['comparison', 'HD'])
        df = df.with_columns(pl.col("comparison").str.split_exact("|", 1).struct.rename_fields(["image", "condition"]).alias("fields")).unnest("fields")
        df = df.with_columns(pl.col("condition").str.split_exact("_", 1).struct.rename_fields(["rotation", "mask"]).alias("fields")).unnest("fields")
        df = df.select(["image", "condition", "rotation", "mask", "HD", "comparison"])
        df = df.drop("comparison")

        return df

    cache = property(fget = __import)