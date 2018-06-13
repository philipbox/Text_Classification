categories = ["child",
              "culture",
              "economy",
              "education",
              "health",
              "life",
              "person",
              "policy",
              "society"]

predict_file = open("output.txt", "r", encoding="utf-8")
answer_file = open("answer.txt", "r", encoding="utf-8")

predicts = []
answers = []

lines = predict_file.readlines()
print(len(lines))
for line in lines:
    percents = []
    percents_strs = line.split("\t")
    for percents_str in percents_strs:
        percents.append(float(percents_str))

    maxi = 0
    for i in range(len(percents)):
        if percents[i] > percents[maxi]:
            maxi = i
    predicts.append(maxi)

lines = answer_file.readlines()
for line in lines:
    answers.append(int(line))

correct = 0
for i in range(len(predicts)):
    if predicts[i] == answers[i]:
        correct += 1

print("Accuracy: " + str(correct / len(predicts)))

true_positive = [0 for _ in range(len(categories))]
# true_negative = [0 for _ in range(len(categories))]

false_positive = [0 for _ in range(len(categories))]
false_negative = [0 for _ in range(len(categories))]

for i in range(len(predicts)):
    if predicts[i] == answers[i]:
        true_positive[predicts[i]] += 1
    else:
        false_positive[predicts[i]] += 1
        false_negative[answers[i]] += 1

total_TP = 0
total_FP = 0
total_FN = 0

for i in range(len(categories)):
    total_TP += true_positive[i]
    total_FP += false_positive[i]
    total_FN += false_negative[i]

total_precision = total_TP / (total_TP + total_FP)
total_recall = total_TP / (total_TP + total_FN)

print("Micro-average F1: " + str((2 * total_precision * total_recall) / (total_precision + total_recall)))

precisions = [0 for _ in range(len(categories))]
recalls = [0 for _ in range(len(categories))]

for i in range(len(categories)):
    if true_positive[i] == 0:
        precisions[i] = 0
        recalls[i] = 0
    else:
        precisions[i] = true_positive[i] / (true_positive[i] + false_positive[i])
        recalls[i] = true_positive[i] / (true_positive[i] + false_negative[i])

avg_precisions = 0
avg_recalls = 0

for i in range(len(categories)):
    avg_precisions += precisions[i]
    avg_recalls += recalls[i]

avg_precisions /= len(categories)
avg_recalls /= len(categories)

print("Macro-average F1: " + str((2 * avg_precisions * avg_recalls) / (avg_precisions + avg_recalls)))