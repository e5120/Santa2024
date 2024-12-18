import random

import numpy as np


def get_population_scores(population, scorer, precomputed):
    scores = []
    for text in population:
        if text in precomputed:
            score = precomputed[text]
        else:
            score = scorer.get_perplexity(text)
            precomputed[text] = score
        scores.append(score)
    scores = np.array(scores)
    return scores, precomputed


def genetic_algorithm(population, scorer, n_gens, crossover_sampler, mutate_sampler, mutate_rate, elite_rate, precomputed={}, logging_step=10):
    assert 0 <= mutate_rate <= 1
    assert 0 <= elite_rate <= 1
    population = np.array(population)
    pop_size = len(population)
    for gen in range(n_gens):
        scores, precomputed = get_population_scores(population, scorer, precomputed)
        sorted_indices = np.argsort(scores)
        # スコア同じで異なる解が出てくる可能性あるので，この多様性の計算方法は微妙．修正した方が良い
        pop_diversity = np.std(scores[sorted_indices[: pop_size//2]])
        if pop_diversity < 1e-3:
            break
        # エリート選択
        n_elites = max(1, int(pop_size*elite_rate))
        next_population = population[sorted_indices[: n_elites]]
        # 解の生成
        while len(next_population) < pop_size:
            p1, p2 = np.random.choice(population, size=2, replace=False)
            p1 = p1.split(" ")
            p2 = p2.split(" ")
            crossover_op = crossover_sampler.sample()
            child = crossover_op(p1, p2)
            if random.random() < mutate_rate:
                mutate_op = mutate_sampler.sample()
                child = mutate_op(child)
            child = " ".join(child)
            next_population = np.concatenate([next_population, [child]])
        population = next_population
        # ログ出力
        if gen % logging_step == 0:
            print(f"[{gen:0>4}] best score: {scores[sorted_indices[0]]:.3f}, population diversity: {pop_diversity:.3f}")
    # 最後の集団のスコアを計算
    scores, precomputed = get_population_scores(population, scorer, precomputed)
    sorted_indices = np.argsort(scores)
    next_population = next_population[sorted_indices]
    scores = scores[sorted_indices]
    return next_population, scores, precomputed
