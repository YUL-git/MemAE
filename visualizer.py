import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def visual_defective(threshold, pred_rec_error, pred_label_ano, anomaly_threshold):
    plt.style.use('tableau-colorblind10')
    # 데이터 분리
    non_defective_train = threshold
    non_defective_test = [pred_rec_error[i] for i in range(len(pred_rec_error)) if pred_label_ano[i] == 0]
    defective_test = [pred_rec_error[i] for i in range(len(pred_rec_error)) if pred_label_ano[i] == 1]

    # 히스토그램 그리기
    plt.figure(figsize=(9, 5)) 
    plt.hist(non_defective_test, color='darkorange', label='Non-defective test', alpha=0.4, bins=30)
    plt.hist(non_defective_train, color='blue', label='Non-defective train', alpha=0.5, bins=30)
    plt.hist(defective_test, color='red', label='Defective test', alpha=0.5, bins=30)

    plt.axvline(x=anomaly_threshold, color='green', linestyle='-', label='Threshold')
    plt.gca().xaxis.grid(False)
    # plt.gca().yaxis.grid(False)
    plt.gca().set_facecolor('whitesmoke')

    plt.ylim(0, 60)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Samples")
    plt.legend()
    plt.title("Histogram of Reconstruction Errors")
    plt.show()

def visual_roc_auc(pred_label_ano, pred_rec_error):
    plt.style.use('tableau-colorblind10')

    fpr, tpr, _ = roc_curve(pred_label_ano, pred_rec_error)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def visual_rec_error(pred, pred_rec_error, anomaly_threshold):
    plt.style.use('tableau-colorblind10')
    colors = ['blue' if p == 0 else 'red' for p in pred]
    plt.figure(figsize=(8, 6)) 
    plt.scatter(range(len(pred_rec_error)), pred_rec_error, color=colors, alpha=0.5, label='pred_rec_error')

    plt.axhline(y=anomaly_threshold, color='green', linestyle='--', label='Anomaly Threshold')
    plt.ylim(0, 1.3)  # y축의 범위를 0과 1.5 사이로 설정
    plt.legend()
    plt.title("Normalized Histogram of a and Scatter plot of pred_rec_error")
    plt.show()

def visual_normal_anomal_image(pred_label_ano, pred_image):
    # 정상 이미지와 비정상 이미지를 분리
    normal_indices = [i for i, label in enumerate(pred_label_ano) if label == 0]
    anomaly_indices = [i for i, label in enumerate(pred_label_ano) if label == 1]

    def plot_images(indices, title):
        for idx in indices:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # x 이미지
            axes[0].imshow(pred_image[idx][0].transpose(1, 2, 0), cmap='gray')
            axes[0].set_title(f"{title} - Original")
            axes[0].axis('off')
            
            # y_hat 이미지
            axes[1].imshow(pred_image[idx][1].transpose(1, 2, 0), cmap='gray')
            axes[1].set_title(f"{title} - Reconstructed")
            axes[1].axis('off')
            
            plt.show()
    # # 정상 이미지 출력
    # plot_images(normal_indices, "Normal")

    # 비정상 이미지 출력
    plot_images(anomaly_indices, "Anomaly")

def visual_memory_items(model):
    # Assuming you have an instance of MemAE called 'model'
    weights = model.backbone.mem_rep.memory.weight.detach().cpu().numpy()

    for i in range(4):
        # Extract the 1xN vector
        vector_1xN = weights[i, :]

        # Resize to 1x28x28
        image = torch.from_numpy(np.expand_dims(vector_1xN.reshape(-1, 1, 1), axis=0))
        memory_item = model.backbone.decoder(image)
        memory_item = memory_item.squeeze().detach().cpu().numpy()

        # Visualize
        plt.imshow(memory_item, cmap='gray')
        plt.title("Decoded Image from Memory Weight")
        plt.axis('off')
        plt.show()