
# Online Python - IDE, Editor, Compiler, Interpreter
import math
import pangolin as pg
from matplotlib import pyplot as plt
print("import complete")
# weights = pg.categorical([.2, .2, .2, .2, .2]) + 1
prior_prob = .2
# model_weight = weights
measurements = [2.9, 4.2, 3.5, 2.5]
print("measurements created")
models = [pg.normal(w, 1) for w in range(1, 6)]
print("models created")
# counts = [[0 for measurement in measurements] for model in models]
# print("counts created")
probs = []
print("started sampling")
#calculate each probability
for i, model in enumerate(models):
    sample = pg.sample(model)
    print(sample.shape)
    print("sample created for model : ", i)
    measurement_total_samples = 0
    current_model_count = [0 for _ in range(len(measurements))]
    print("beginning sampling")
    for s in sample:
        measurement_total_samples += 1
        for j, measurement in enumerate(measurements):
            count = 0
            if s <= measurement + .05 and s >= measurement - .05:
                    current_model_count[j] += 1
    current_model_probs = [count / measurement_total_samples for count in current_model_count]
    probs.append(current_model_probs)
    print("sampling complete for model : ", i)
print("sampling complete")
print(probs)
posteriors = [math.prod(p) * prior_prob for p in probs]
print(posteriors)
s = sum(posteriors)
normal_posteriors = [p / s for p in posteriors]
print(normal_posteriors)
sum(normal_posteriors)