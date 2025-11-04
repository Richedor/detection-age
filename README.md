# detection-age

Outil minimal de détection/estimation d'âge sur image ou webcam. Par défaut, il s'appuie sur le moteur OpenCV DNN et sur les modèles Caffe classiques (`age_net.caffemodel` + détecteur SSD). Un mode de secours `deepface` est disponible si vous avez installé le package.

## Prérequis
- Python 3.8+
- Environnement `D:\vision\env` (fourni) ou tout environnement contenant `opencv-python` et `numpy`
- Connexion internet lors du téléchargement initial des modèles Caffe (environ 100 Mo)

## Installation / activation de l'environnement existant

## Détection d'âge — minimal, local, sans serveur

	- `webcam_age.py` ouvre la webcam et affiche l'âge estimé.
	- `pick_image_age.py` ouvre une boîte de dialogue pour choisir des images et les annote.

Deux moteurs au choix:
- caffe (OpenCV DNN + modèles Caffe) → rapide, renvoie des tranches d'âge
- deepface (fallback) → aucun modèle Caffe requis, renvoie un âge estimé

L'ancien `detect_age.py` est déprécié et n'est plus utilisé.

## Prérequis rapides

- Python 3.x, OpenCV (`opencv-python`), NumPy
- Optionnel si vous utilisez le moteur deepface: `deepface` (+ TensorFlow)

Installez les dépendances (inclut désormais DeepFace + tf-keras compatibles):

```
pip install -r requirements.txt
```

Si vous n'utilisez jamais le moteur deepface, vous pouvez retirer `deepface` et `tf-keras==2.19.*` du `requirements.txt`.

# detection-age

Outil minimal et local d’estimation d’âge sur webcam ou images. Deux moteurs disponibles:
- caffe (OpenCV DNN + modèles Caffe) → rapide, renvoie une tranche d’âge
- deepface (fallback) → aucun modèle Caffe requis, renvoie un âge estimé

Pas de serveur, tout s’exécute en local. `detect_age.py` est déprécié; utilisez les deux scripts ci-dessous.

## Scripts

- `webcam_age.py` — ouvre la webcam et affiche l’âge estimé en temps réel
- `pick_image_age.py` — ouvre une boîte de dialogue pour choisir des images et les annote

## Prérequis

- Python 3.8+
- Dépendances Python (installées via `requirements.txt`)

Installation:

```
pip install -r requirements.txt
```

Le requirements inclut `opencv-python`, `numpy`, ainsi que `deepface` + `tf-keras==2.19.*` pour permettre le moteur DeepFace. Si vous n’utilisez jamais DeepFace, vous pouvez retirer ces deux dernières dépendances.

## Modèles Caffe (moteur « caffe »)

Créez un dossier local `models/` à la racine du projet et placez-y les fichiers suivants:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`
- `age_deploy.prototxt`
- `age_net.caffemodel`

Téléchargement (Windows PowerShell):

```
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1 -OutputDir ".\models"
```

Conseils:
- Vérifiez que `age_net.caffemodel` fait plusieurs Mo (évite un fichier corrompu)
- Si les miroirs échouent, utilisez le moteur `deepface` en attendant

## Utilisation

Webcam (Windows PowerShell):

```
python .\webcam_age.py --engine auto --models-dir ".\models"
```

Images (boîte de dialogue):

```
python .\pick_image_age.py --engine auto --models-dir ".\models"
```

Options:
- `--engine` : `auto` (par défaut), `caffe`, `deepface`
- `--cam-index` : index webcam (0 par défaut)
- `--display-width` : redimension de l’affichage (ex: 960)
- `--analyze-every` : DeepFace uniquement; analyse 1 frame sur N (ex: 15)

## Dépannage

- DeepFace peut télécharger ~500 Mo de poids au premier lancement.
- Si la webcam ne s’ouvre pas, essayez `--cam-index 1` (ou plus).
- Si `age_net.caffemodel` est manquant/corrompu, repassez en `--engine deepface` le temps de re-télécharger les modèles.
