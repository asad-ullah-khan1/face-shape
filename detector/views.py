from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
import cv2
import dlib
import numpy as np
import os
from deepface import DeepFace
from django.http import JsonResponse

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_face_shape(landmarks):
    # Extract the key facial landmarks for analysis
    jaw = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]  # Jawline points
    left_cheek = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 4)]  # Left cheekbone points
    right_cheek = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(13, 16)]  # Right cheekbone points
    forehead = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(19, 22)]  # Forehead points

    # Calculate the jaw width, height, cheek and forehead distances
    jaw_width = np.linalg.norm(np.array(jaw[0]) - np.array(jaw[16]))  # Jaw width distance
    jaw_height = np.linalg.norm(np.array(jaw[8]) - np.array(jaw[0]))  # Jaw height distance

    left_cheek_distance = np.linalg.norm(np.array(left_cheek[0]) - np.array(left_cheek[2]))
    right_cheek_distance = np.linalg.norm(np.array(right_cheek[0]) - np.array(right_cheek[2]))
    forehead_width = np.linalg.norm(np.array(forehead[0]) - np.array(forehead[2]))

    # Use the ratios between jaw, cheek, and forehead dimensions for better classification
    if jaw_width > jaw_height * 1.5 and left_cheek_distance < right_cheek_distance:
        return "Square"
    elif jaw_width < jaw_height * 1.2 and forehead_width > jaw_width * 1.2:
        return "Oval"
    elif jaw_width < jaw_height * 1.2 and left_cheek_distance > right_cheek_distance:
        return "Heart"
    else:
        return "Round"


def upload_image(request):
    face_shape = None
    image_url = None
    measurements = {}
    gender = None  # Initialize gender variable
    age = None
    emotion = None
    race = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_model = form.save()
            image_path = img_model.image.path
            image_url = img_model.image.url

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                landmarks = predictor(gray, faces[0])
                face_shape = get_face_shape(landmarks)

                # Get facial measurements
                jaw_width = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][16]))
                jaw_height = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][8]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][0]))
                left_cheek_distance = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 4)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 4)][2]))
                right_cheek_distance = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(13, 16)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(13, 16)][2]))
                forehead_width = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(19, 22)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(19, 22)][2]))

                measurements = {
                    'jaw_width': jaw_width,
                    'jaw_height': jaw_height,
                    'left_cheek_distance': left_cheek_distance,
                    'right_cheek_distance': right_cheek_distance,
                    'forehead_width': forehead_width
                }

                # Gender detection using DeepFace
                result = DeepFace.analyze(image_path, actions=['gender', 'age', 'emotion', 'race'], enforce_detection=False)
                gender_result = result[0]['gender']
                age = result[0]['age']
                emotion = result[0]['dominant_emotion']
                race = result[0]['dominant_race']
                gender = 'Man' if gender_result['Man'] > gender_result['Woman'] else 'Woman'

    else:
        form = ImageUploadForm()

    # Suggested Glasses based on Face Shape (this can be extended based on measurements)
    glasses_suggestions = {
        'Oval': "Round or Rectangular frames",
        'Square': "Round or Oval glasses to soften sharp features",
        'Round': "Angular frames to contrast round features",
        'Heart': "Cat-eye glasses or frames with wider bottoms",
        'Diamond': "Oval or geometric frames to emphasize cheekbones"
    }

    return render(request, 'upload.html', {
        'form': form,
        'face_shape': face_shape,
        'image_url': image_url,
        'measurements': measurements,
        'gender': gender,  # gender is passed correctly now
        'age': age,
        'emotion': emotion,
        'race': race,
        'glasses_suggestions': glasses_suggestions.get(face_shape, "Try different styles!")
    })

def api_view(request):
    face_shape = None
    image_url = None
    measurements = {}
    gender = None
    age = None
    emotion = None
    race = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_model = form.save()
            image_path = img_model.image.path
            image_url = img_model.image.url

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                landmarks = predictor(gray, faces[0])
                face_shape = get_face_shape(landmarks)

                # Get facial measurements
                jaw_width = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][16]))
                jaw_height = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][8]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)][0]))
                left_cheek_distance = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 4)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 4)][2]))
                right_cheek_distance = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(13, 16)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(13, 16)][2]))
                forehead_width = np.linalg.norm(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(19, 22)][0]) - np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(19, 22)][2]))

                measurements = {
                    'jaw_width': jaw_width,
                    'jaw_height': jaw_height,
                    'left_cheek_distance': left_cheek_distance,
                    'right_cheek_distance': right_cheek_distance,
                    'forehead_width': forehead_width
                }

                # Gender detection using DeepFace
                result = DeepFace.analyze(image_path, actions=['gender', 'age', 'emotion', 'race'], enforce_detection=False)
                gender_result = result[0]['gender']
                age = result[0]['age']
                emotion = result[0]['dominant_emotion']
                race = result[0]['dominant_race']
                gender = 'Man' if gender_result['Man'] > gender_result['Woman'] else 'Woman'

    # Suggested Glasses based on Face Shape (this can be extended based on measurements)
    glasses_suggestions = {
        'Oval': "Round or Rectangular frames",
        'Square': "Round or Oval glasses to soften sharp features",
        'Round': "Angular frames to contrast round features",
        'Heart': "Cat-eye glasses or frames with wider bottoms",
        'Diamond': "Oval or geometric frames to emphasize cheekbones"
    }

    response_data = {
        'face_shape': face_shape,
        'image_url': image_url,
        'measurements': measurements,
        'gender': gender,
        'age': age,
        'emotion': emotion,
        'race': race,
        'glasses_suggestions': glasses_suggestions.get(face_shape, "Try different styles!")
    }

    return JsonResponse(response_data)