import math
import random
import collections


# ToDo: どの操作がスコアを上げたのかをログする機能の追加
def simulated_annealing(text, sampler, scorer, temp_start=10, temp_end=0.5, cooling_rate=0.95, steps_per_temp=5, alpha=1.0, precomputed={}, verbose=False, logging_step=1, batch_size=1, taboo_size=0):
    # initial setting
    text = text.strip()
    current_tokens = text.split()
    current_text = " ".join(current_tokens)
    current_score = scorer.get_perplexity(text, batch_size=batch_size)
    best_tokens = current_tokens.copy()
    best_score = current_score
    text_history, score_history = [text], [best_score]
    taboo_list = collections.deque([text], maxlen=taboo_size)
    # optimization
    temp = temp_start
    print(f"start temp: {temp:.2f}, init score: {best_score:.5f}")
    num_steps = 0
    while temp > temp_end:
        num_steps += 1
        for _ in range(steps_per_temp):
            # 近傍解の生成
            op = sampler.sample()
            new_tokens = op(current_tokens.copy())
            new_text = " ".join(new_tokens)
            # 近傍解の評価
            if new_text in precomputed:
                new_score = precomputed[new_text]
            else:
                new_score = scorer.get_perplexity(new_text, batch_size=batch_size)
                precomputed[new_text] = new_score
            # 近傍解の採否判定
            # - : 解を更新しない
            # = : スコアの悪化した解を受け入れる
            # < : 現在のスコアを更新
            # > : 最良のスコアを更新
            if new_text not in taboo_list and new_text != current_text:
                delta = new_score - current_score
                if new_score < best_score:
                    taboo_list.append(new_text)
                    best_tokens = new_tokens.copy()
                    best_text = new_text
                    best_score = new_score
                    current_tokens = new_tokens.copy()
                    current_text = new_text
                    current_score = new_score
                    text_history.append(new_text)
                    score_history.append(new_score)
                    print(">", end="")
                elif delta < 0 or random.random() < math.exp(-alpha*delta / temp):
                    taboo_list.append(new_text)
                    current_tokens = new_tokens.copy()
                    current_text = new_text
                    current_score = new_score
                    text_history.append(new_text)
                    score_history.append(new_score)
                    if delta < 0:
                        print("<", end="")
                    else:
                        print("=", end="")
                else:
                    print("-", end="")
            else:
                print("-", end="")
        temp *= cooling_rate
        if verbose and num_steps % logging_step == 0:
            print(f"\ntemp: {temp:.2f}, current score: {current_score:.5f}, best score: {best_score:.5f}")
    best_text = " ".join(best_tokens)
    current_text = " ".join(current_tokens)
    return best_text, best_score, current_text, current_score, precomputed, text_history, score_history
