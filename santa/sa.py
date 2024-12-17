import math
import random


# ToDo: どの操作がスコアを上げたのかをログする機能の追加
def simulated_annealing(text, sampler, scorer, temp_start=10, temp_end=0.5, cooling_rate=0.95, steps_per_temp=5, alpha=1.0, precomputed={}, verbose=False, logging_step=1, batch_size=1):
    # initial setting
    text = text.strip()
    current_tokens = text.split(" ")
    current_score = scorer.get_perplexity(text, batch_size=batch_size)
    best_tokens = current_tokens.copy()
    best_score = current_score
    text_history, score_history = [text], [best_score]
    # optimization
    temp = temp_start
    print(f"start temp: {temp:.2f}, init score: {best_score:.5f}")
    num_steps = 0
    while temp > temp_end:
        num_steps += 1
        for _ in range(steps_per_temp):
            op = sampler.sample()
            new_tokens = op(current_tokens.copy())
            new_text = " ".join(new_tokens)
            if new_text in precomputed:
                new_score = precomputed[new_text]
            else:
                new_score = scorer.get_perplexity(new_text, batch_size=batch_size)
                precomputed[new_text] = new_score
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-alpha*delta / temp):
                current_tokens = new_tokens.copy()
                current_score = new_score
                text_history.append(new_text)
                score_history.append(new_score)
                if new_score < best_score:
                    best_tokens = new_tokens.copy()
                    best_score = new_score
                    print(">", end="")
                else:
                    print("<", end="")
            else:
                print("-", end="")
        temp *= cooling_rate
        if verbose and num_steps % logging_step == 0:
            print(f"\ncurrent temp: {temp:.2f}, current score: {current_score:.5f}, best score: {best_score:.5f}")
    return " ".join(best_tokens), best_score, precomputed, text_history, score_history
