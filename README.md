# VGG16
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classes in y_true:", np.unique(y_true))
print("Classes in y_pred:", np.unique(y_pred))
print("Test set distribution:")
for i, emotion in enumerate(valid_emotions):
    count_true = np.sum(y_true == i)
    count_pred = np.sum(y_pred == i)
    print(f"{emotion}: {count_true} true, {count_pred} predicted")

all_labels = list(range(len(valid_emotions)))
cm = confusion_matrix(y_true, y_pred, labels=all_labels)

print(f"\nConfusion Matrix Shape: {cm.shape}")
print("Confusion Matrix:\n", cm)


plt.figure(figsize=(12, 10))
annot_matrix = cm.astype(str)


ax = sns.heatmap(cm, 
                 annot=annot_matrix,
                 fmt='',
                 cmap='Blues',
                 cbar=True,
                 xticklabels=valid_emotions,
                 yticklabels=valid_emotions,
                 linewidths=0.5,
                 linecolor='gray',
                 square=True,
                 annot_kws={"size": 12, "weight": "bold"})

plt.title('Confusion Matrix - Thermal Emotion Recognition\n(VGG16 Transfer Learning)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
cbar = ax.collections[0].colorbar
cbar.set_label('Number of Samples', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=valid_emotions, zero_division=0))


print(f"\nFinal Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Total test samples: {len(y_test)}")
print(f"Classes with samples in test set: {len(np.unique(y_true))}/{len(valid_emotions)}")
