import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

breastCancer = pd.read_csv('data/breastCancer.csv')
print(breastCancer.head(10))

# diagnosis count
sns.set(style="whitegrid")
plt.figure(figsize=(5, 4))
sns.countplot(x='diagnosis', data=breastCancer, palette='Pastel1')
plt.title('Diagnosis Count')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()

# 10 means, 10 se, 10 worst
print(breastCancer.columns)

x = breastCancer.drop(['id','diagnosis', 'Unnamed: 32'],axis = 1)
print("x")
print(x.head())
y_label = breastCancer['diagnosis']
y = [1 if each == "M" else 0 for each in breastCancer.diagnosis]
print("y")
print(y)

# mean columns vs label
sns.set(style="whitegrid")
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
for i in range(10):
    row, col = divmod(i, 2)
    sns.histplot(x=x.iloc[:, i], hue=y_label, data=x, ax=axes[row, col], palette='Pastel1', element="step", stat="density", common_norm=False, alpha=0.7)
    axes[row, col].set_title(f'{x.columns[i]} vs Diagnosis')
plt.tight_layout()
plt.show()

# correlation matrix
correlation_matrix = x.corr()
plt.figure(figsize=(24, 20))
cmap = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_scaled, y_train)

y_pred = svm_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(3, 2.3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], annot_kws={'size': 20})
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()