#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script minimal: ouvre la webcam et affiche la tranche d'âge estimée avec OpenCV DNN.
Prérequis modèles (par défaut dans ./models) :
- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel
- age_deploy.prototxt
- age_net.caffemodel

Quitter: appuyer sur 'q'
"""
import os
import argparse
import cv2
import numpy as np

# Réduire le verbiage TensorFlow si DeepFace est utilisé plus tard
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def _import_deepface():
    """Import paresseux de DeepFace, retourne le module ou lève l'exception.
    Permet d'afficher une erreur claire si l'import échoue après installation.
    """
    try:
        from deepface import DeepFace  # type: ignore
        return DeepFace
    except Exception as e:  # montrer l'erreur réelle (pas seulement ImportError)
        raise RuntimeError(f"Import DeepFace échoué: {e}")

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]


def find_file(models_dir, names):
    for n in names:
        p = os.path.join(models_dir, n)
        if os.path.isfile(p):
            return p
    return None


def load_models(models_dir):
    face_proto = find_file(models_dir, ["deploy.prototxt", "deploy.prototxt.txt"]) 
    face_model = find_file(models_dir, ["res10_300x300_ssd_iter_140000.caffemodel"]) 
    age_proto = find_file(models_dir, ["age_deploy.prototxt", "age_deploy.prototxt.txt"]) 
    age_model = find_file(models_dir, ["age_net.caffemodel"]) 

    missing = []
    if not face_proto:
        missing.append("deploy.prototxt")
    if not face_model:
        missing.append("res10_300x300_ssd_iter_140000.caffemodel")
    if not age_proto:
        missing.append("age_deploy.prototxt")
    if not age_model:
        missing.append("age_net.caffemodel")
    else:
        # Vérifier taille minimale pour éviter fichiers corrompus
        try:
            if os.path.getsize(age_model) < 5_000_000:  # 5 Mo
                missing.append("age_net.caffemodel (fichier trop petit/corrompu)")
        except OSError:
            missing.append("age_net.caffemodel (inaccessible)")

    if missing:
        return None, None, missing

    face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    return face_net, age_net, []

def run(models_dir="./models", cam_index=0, conf_threshold=0.6, engine="auto"):
    # Choix du moteur: 'auto' (par défaut), 'deepface', 'caffe'
    use_deepface = False
    face_net = age_net = None
    if engine == "deepface":
        try:
            _ = _import_deepface()
            use_deepface = True
        except Exception as e:
            print("[ERREUR] DeepFace indisponible:", e)
            print("Astuce: assurez-vous d'exécuter avec le même Python que celui ayant installé deepface.")
            return
    elif engine == "caffe":
        face_net, age_net, missing = load_models(models_dir)
        if face_net is None:
            print("[ERREUR] Modèles manquants dans:", models_dir)
            print("- Manquants:", ", ".join(missing))
            print("Solution (Windows): powershell -ExecutionPolicy Bypass -File .\\scripts\\download_models.ps1 -OutputDir '" + models_dir + "'")
            return
    else:  # auto
        # Priorité à DeepFace si présent, sinon Caffe
        try:
            _ = _import_deepface()
            use_deepface = True
        except Exception:
            use_deepface = False
            face_net, age_net, missing = load_models(models_dir)
            if face_net is None:
                print("[ERREUR] Modèles manquants et DeepFace indisponible.")
                print("- Manquants:", ", ".join(missing))
                print("Solution: téléchargez les modèles OU installez DeepFace: python -m pip install deepface")
                return

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam")
        return

    print("Appuyez sur 'q' pour quitter")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if use_deepface:
            # DeepFace: estimation directe (plus lent), sans détection locale
            try:
                DeepFace = _import_deepface()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis = DeepFace.analyze(rgb, actions=["age"], detector_backend="opencv", enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                reg = analysis.get("region") or analysis.get("facial_area") or {}
                x, y, w, h = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
                age = analysis.get("age")
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                    cv2.putText(frame, f"age: {int(age)}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                else:
                    cv2.putText(frame, f"age: {int(age)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
            except Exception as e:
                cv2.putText(frame, f"DeepFace erreur: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            h, w = frame.shape[:2]
            # Détection visage (SSD Caffe)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < conf_threshold:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                # Age
                face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                age_net.setInput(face_blob)
                preds = age_net.forward()
                idx = int(np.argmax(preds[0]))
                label = f"{AGE_BUCKETS[idx]} {preds[0][idx]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imshow("Webcam - Age", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="./models", help="Dossier contenant les modèles Caffe")
    parser.add_argument("--cam-index", type=int, default=0, help="Index de la webcam (0 par défaut)")
    parser.add_argument("--conf", type=float, default=0.6, help="Seuil de confiance détection visage")
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "deepface", "caffe"], help="Moteur d'estimation d'âge")
    args = parser.parse_args()
    run(models_dir=args.models_dir, cam_index=args.cam_index, conf_threshold=args.conf, engine=args.engine)
