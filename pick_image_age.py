#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sélection d'images via une boîte de dialogue (sans serveur) et estimation d'âge.
- Choix d'une ou plusieurs images avec tkinter.filedialog
- Moteurs: OpenCV DNN (Caffe) ou DeepFace (fallback si DNN absent)
- Affiche chaque image annotée; touche pour passer, 'q' pour quitter
"""
import os
import argparse
import cv2
import numpy as np

# réduire le verbiage TF si DeepFace est utilisé
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]


def _import_deepface():
    try:
        from deepface import DeepFace  # type: ignore
        return DeepFace
    except Exception as e:
        raise RuntimeError(f"Import DeepFace échoué: {e}")


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
        try:
            if os.path.getsize(age_model) < 5_000_000:
                missing.append("age_net.caffemodel (fichier trop petit/corrompu)")
        except OSError:
            missing.append("age_net.caffemodel (inaccessible)")

    if missing:
        return None, None, missing

    face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    return face_net, age_net, []


def analyze_caffe(img_bgr, face_net, age_net, conf_threshold=0.6):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        face = img_bgr[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(face_blob)
        preds = age_net.forward()
        idx = int(np.argmax(preds[0]))
        results.append({
            "region": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "label": f"{AGE_BUCKETS[idx]} {preds[0][idx]:.2f}",
        })
    return results


def analyze_deepface(img_bgr):
    DeepFace = _import_deepface()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    analysis = DeepFace.analyze(rgb, actions=["age"], detector_backend="opencv", enforce_detection=False)
    if isinstance(analysis, list):
        analysis = analysis[0]
    reg = analysis.get("region") or analysis.get("facial_area") or {}
    x, y, w, h = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
    age = analysis.get("age")
    label = f"age: {int(age) if age is not None else '?'}"
    return [{"region": {"x": x, "y": y, "w": w, "h": h}, "label": label}]


def draw_results(img, results):
    out = img.copy()
    if not results:
        cv2.putText(out, "Aucun visage / estimation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    for r in results:
        reg = r.get("region", {})
        x, y, w, h = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
        label = r.get("label", "")
        if w > 0 and h > 0:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(out, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        else:
            cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    return out


def select_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(f"Tkinter indisponible: {e}")
    root = tk.Tk()
    root.withdraw()
    filetypes = [
        ("Images", ".jpg .jpeg .png .bmp .tif .tiff"),
        ("Tous les fichiers", "*.*"),
    ]
    paths = filedialog.askopenfilenames(title="Choisir des images", filetypes=filetypes)
    root.destroy()
    return list(paths)


def main():
    parser = argparse.ArgumentParser(description="Sélection d'images et estimation d'âge")
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "deepface", "caffe"], help="Moteur d'estimation d'âge")
    parser.add_argument("--models-dir", type=str, default="./models", help="Dossier des modèles Caffe pour 'caffe' ou 'auto'")
    parser.add_argument("--conf", type=float, default=0.6, help="Seuil confiance détection visage (caffe)")
    parser.add_argument("--display-width", type=int, default=1280, help="Largeur d'affichage max")
    args = parser.parse_args()

    # Sélection des fichiers
    try:
        images = select_files()
    except Exception as e:
        print("[ERREUR]", e)
        return
    if not images:
        print("Aucune image sélectionnée.")
        return

    # Choix moteur
    use_deepface = False
    face_net = age_net = None

    if args.engine == "deepface":
        _ = _import_deepface()
        use_deepface = True
    elif args.engine == "caffe":
        face_net, age_net, missing = load_models(args.models_dir)
        if face_net is None:
            print("[ERREUR] Modèles manquants:", ", ".join(missing))
            return
    else:  # auto
        try:
            _ = _import_deepface()
            use_deepface = True
        except Exception:
            face_net, age_net, missing = load_models(args.models_dir)
            if face_net is None:
                print("[ERREUR] Modèles manquants et DeepFace indisponible:", ", ".join(missing))
                return

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Impossible de lire: {path}")
            continue
        # display resize
        if args.display_width and img.shape[1] > args.display_width:
            scale = args.display_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        try:
            if use_deepface:
                results = analyze_deepface(img)
            else:
                results = analyze_caffe(img, face_net, age_net, conf_threshold=args.conf)
        except Exception as e:
            print(f"[ERREUR] analyse: {path}: {e}")
            continue

        out = draw_results(img, results)
        cv2.imshow("Age Estimation - image", out)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
