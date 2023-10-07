import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from deepface import DeepFace
import PIL.Image as Image


def get_dataset():
    """
    t[0]: name
    t[1]: database image
    t[2]: same person image | different person name
    if t[2] == different person name that t[3] = different person image
    :return:
    """
    # 按行读取 pairsDevTrain.txt
    test_dataset = []
    with open("./pairsDevTest.txt", "r") as f:
        lines = f.readlines()
        lines.pop(0)
        for line in lines:
            t = line.split("\t")
            t[-1] = t[-1].removesuffix("\n")
            test_dataset.append(t)
    return test_dataset


def verify_test_dataset(dataset, model_name="VGG-Face", distance_metric="cosine",
                        detector_backend="opencv"):
    actuals = []
    predictions = []
    predictions_halftone = []
    if not os.path.exists(f"./{model_name}_{distance_metric}_{detector_backend}.csv"):
        with open(f"./{model_name}_{distance_metric}_{detector_backend}.csv", "w") as f:
            f.write("1st person,2nd person,dist,verified,dist_halftone,verified_halftone\n")

    for img in dataset:
        same = True if len(img) == 3 else False
        first_img = f"./lfw_funneled/{img[0]}/{img[0]}_{int(img[1]):04d}.jpg"
        second_img = f"./lfw_funneled/{img[0]}/{img[0]}_{int(img[2]):04d}.jpg" \
            if same else f"./lfw_funneled/{img[2]}/{img[2]}_{int(img[3]):04d}.jpg"

        second_img_halftone = np.array(Image.open(second_img).convert("1").convert("RGB"))
        first_img = np.array(Image.open(first_img))
        second_img = np.array(Image.open(second_img))
        # opencv expects bgr instead of rgb
        first_img = first_img[:, :, ::-1]
        second_img = second_img[:, :, ::-1]
        second_img_halftone = second_img_halftone[:, :, ::-1]
        result = DeepFace.verify(first_img, second_img, model_name=model_name, distance_metric=distance_metric,
                                 detector_backend=detector_backend, enforce_detection=False)
        result_halftone = DeepFace.verify(first_img, second_img_halftone, model_name=model_name,
                                          distance_metric=distance_metric,
                                          detector_backend=detector_backend, enforce_detection=False)

        data = pd.DataFrame({"1st person": [img[0]],
                             "2nd person": [img[0] if same else img[2]],
                             "dist": [result["distance"]],
                             "verified": [result["verified"]],
                             "dist_halftone": [result_halftone["distance"]],
                             "verified_halftone": [result_halftone["verified"]]
                             })

        data.to_csv(f"./{model_name}_{distance_metric}_{detector_backend}.csv", mode="a", header=False, index=False)

        print(
            f"{img} dist: {result['distance']} verified: {result['verified']} dist_halftone: {result_halftone['distance']} verified_halftone: {result_halftone['verified']}")
        actuals.append(True if same else False)
        predictions.append(result["verified"])
        predictions_halftone.append(result_halftone["verified"])

    accuracy = 100 * accuracy_score(actuals, predictions)
    precision = 100 * precision_score(actuals, predictions)
    recall = 100 * recall_score(actuals, predictions)
    f1 = 100 * f1_score(actuals, predictions)
    tn, fp, fn, tp = confusion_matrix(actuals, predictions).ravel()

    metric = pd.DataFrame({"accuracy": [accuracy],
                           "precision": [precision],
                           "recall": [recall],
                           "f1": [f1],
                           "tn": [tn],
                           "fp": [fp],
                           "fn": [fn],
                           "tp": [tp]
                           })
    metric.to_csv(f"./{model_name}_{distance_metric}_{detector_backend}.csv", mode="a", header=True, index=False)

    accuracy_halftone = 100 * accuracy_score(actuals, predictions_halftone)
    precision_halftone = 100 * precision_score(actuals, predictions_halftone)
    recall_halftone = 100 * recall_score(actuals, predictions_halftone)
    f1_halftone = 100 * f1_score(actuals, predictions_halftone)
    tn_halftone, fp_halftone, fn_halftone, tp_halftone = confusion_matrix(actuals, predictions_halftone).ravel()
    metric_halftone = pd.DataFrame({"h_accuracy": [accuracy_halftone],
                                    "h_precision": [precision_halftone],
                                    "h_recall": [recall_halftone],
                                    "h_f1": [f1_halftone],
                                    "h_tn": [tn_halftone],
                                    "h_fp": [fp_halftone],
                                    "h_fn": [fn_halftone],
                                    "h_tp": [tp_halftone]
                                    })
    metric_halftone.to_csv(f"./{model_name}_{distance_metric}_{detector_backend}.csv", mode="a", header=True,
                           index=False)


# Euclidean L2 form seems to be more stable than cosine and regular Euclidean distance based on experiments.

# Face recognition models are actually CNN models and they expect standard sized inputs. So, resizing is required before
# representation. To avoid deformation, deepface adds black padding pixels according to the target size argument
# after detection and alignment.

# RetinaFace and MTCNN seem to over-perform in detection and alignment stages, but they
# are much slower. If the speed of your pipeline is more important, then you should use opencv or ssd. On the other
# hand, if you consider the accuracy, then you should use retinaface or mtcnn.
if __name__ == '__main__':
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
    ]
    backends = [
        'opencv',
        'ssd',
        'dlib',
        'mtcnn',
        'retinaface',
        'mediapipe',
        'yolov8',
        'yunet',
    ]
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    dataset = get_dataset()
    # t = dataset[0]
    # res = df.verify(f"./lfw_funneled/{t[0]}/{t[0]}_{int(t[1]):04d}.jpg",
    #                 f"./lfw_funneled/{t[0]}/{t[0]}_{int(t[2]):04d}.jpg",
    #                 model_name="VGG-Face", distance_metric="cosine", detector_backend="opencv")
    # print(res)
    # facenet proved to not be a good model for halftone face recognition
    verify_test_dataset(dataset, model_name=models[0], distance_metric=metrics[0], detector_backend=backends[0])
