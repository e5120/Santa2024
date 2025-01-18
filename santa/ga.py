import random

import numpy as np


def get_token2id(text):
    token2id = {}
    tokens = text.split()
    for i in range(len(tokens)):
        token = tokens[i]
        if token not in token2id:
            token2id[token] = len(token2id)
        else:
            j = 2
            new_token = f"{token}_{j}"
            while new_token in token2id:
                j += 1
                new_token = f"{token}_{j}"
            token2id[new_token] = len(token2id)
    return token2id


def tokens2order(tokens, token2id):
    token2id = token2id.copy()
    order = []
    for token in tokens:
        if token in token2id:
            order.append(token2id.pop(token))
        else:
            j = 2
            new_token = f"{token}_{j}"
            while new_token not in token2id:
                j += 1
                new_token = f"{token}_{j}"
            order.append(token2id.pop(new_token))
    return order


def order2token(order, id2token):
    tokens = []
    for i in order:
        tokens.append(id2token[i].split("_")[0])
    return tokens


def get_population_scores(population, scorer, precomputed):
    scores = []
    for text in population:
        if text not in precomputed:
            precomputed[text] = scorer.get_perplexity(text)
        scores.append(precomputed[text])
    scores = np.array(scores)
    return scores, precomputed


def genetic_algorithm(best_text, pop_size, scorer, n_gens, crossover_sampler, mutate_sampler, mutate_rate, elite_rate, precomputed={}, logging_step=10):
    assert 0 <= mutate_rate <= 1
    assert 0 <= elite_rate <= 1
    # 初期集団の生成
    best_tokens = best_text.split()
    token2id = get_token2id(best_text)
    id2token = {v: k for k, v in token2id.items()}
    best_order = tokens2order(best_tokens, token2id)
    population = [best_text]
    for _ in range(pop_size - 1):
        # op = mutate_sampler.sample()
        # new_order = op(best_order)
        new_order = best_order.copy()
        np.random.shuffle(new_order)
        new_tokens = order2token(new_order, id2token)
        new_text = " ".join(new_tokens)
        population.append(new_text)
    population = np.array(population)
    # 世代の更新
    for gen in range(n_gens):
        scores, precomputed = get_population_scores(population, scorer, precomputed)
        sorted_indices = np.argsort(scores)
        top50_text = population[sorted_indices[: pop_size//2]]
        pop_diversity = (len(set(top50_text)) - 1) / len(top50_text)
        if pop_diversity < 1e-3:
            break
        # エリート選択
        n_elites = max(1, int(pop_size*elite_rate))
        next_population = population[sorted_indices[: n_elites]]
        np.random.shuffle(next_population)
        # 解の生成
        while len(next_population) < pop_size:
            p1, p2 = np.random.choice(population, size=2, replace=False)
            p1 = tokens2order(p1.split(), token2id)
            p2 = tokens2order(p2.split(), token2id)
            crossover_op = crossover_sampler.sample()
            childs = crossover_op(p1, p2)
            for child in childs:
                if random.random() < mutate_rate:
                    mutate_op = mutate_sampler.sample()
                    child = mutate_op(child)
                child = order2token(child, id2token)
                child = " ".join(child)
                next_population = np.concatenate([next_population, [child]])
        population = next_population
        # ログ出力
        if gen % logging_step == 0:
            print(f"[{gen:0>4}] best score: {scores[sorted_indices[0]]:.3f}, population diversity: {pop_diversity:.3f}")
    # 最後の集団のスコアを計算
    scores, precomputed = get_population_scores(population, scorer, precomputed)
    sorted_indices = np.argsort(scores)
    population = population[sorted_indices]
    scores = scores[sorted_indices]
    return population, scores, precomputed
