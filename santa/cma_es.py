from math import log

import torch


class FastCMA(object):
    def __init__(self, N, samples):
        self.samples = samples
        mu = samples // 2
        self.weights = torch.tensor([log(mu + 0.5)]).cuda()
        self.weights = self.weights - torch.linspace(
            start=1, end=mu, steps=mu).cuda().log()
        self.weights /= self.weights.sum()
        self.mueff = (self.weights.sum() ** 2 / (self.weights ** 2).sum()).item()
        # settings
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff)
        self.cmu /= ((N + 2) ** 2 + 2 * self.mueff / 2)
        # variables
        self.mean = torch.zeros(N).cuda()
        self.b = torch.eye(N).cuda()
        self.d = self.b.clone()
        bd = self.b * self.d
        self.c = bd * bd.T
        self.pc = self.mean.clone()

    def step(self, objective_f, tokens, step_size = 0.5):
        z = torch.randn(self.mean.size(0), self.samples).cuda()
        s = self.mean.view(-1, 1) + step_size * self.b.matmul(self.d.matmul(z))
        s_i = s.argsort(dim=0).cpu().numpy()
        texts = []
        for i in range(s_i.shape[1]):
            texts.append(" ".join(tokens[s_i[:, i]]))
        fitness = objective_f(texts, batch_size=16)
        results = []
        for i in range(len(fitness)):
            results.append({
                "parameters": s.T[i],
                "z": z.T[i],
                "fitness": fitness[i],
                "tokens": texts[i],
            })

        ranked_results = sorted(results, key=lambda x: x['fitness'])
        selected_results = ranked_results[0:self.samples//2]
        z = torch.stack([g['z'] for g in selected_results])
        g = torch.stack([g['parameters'] for g in selected_results])

        self.mean = (g * self.weights.unsqueeze(1)).sum(0)
        zmean = (z * self.weights.unsqueeze(1)).sum(0)
        self.pc *= (1 - self.cc)
        pc_cov = self.pc.unsqueeze(1) * self.pc.unsqueeze(1).T
        pc_cov = pc_cov + self.cc * (2 - self.cc) * self.c

        bdz = self.b.matmul(self.d).matmul(z.T)
        cmu_cov = bdz.matmul(self.weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.T)

        self.c *= (1 - self.c1 - self.cmu)
        self.c += (self.c1 * pc_cov) + (self.cmu * cmu_cov)
        self.d, self.b = torch.linalg.eigh(self.c, UPLO='U')
        self.d = self.d.sqrt().diag_embed()
        return ranked_results
