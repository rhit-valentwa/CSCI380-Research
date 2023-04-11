import face_recognition as fr
import cv2
import numpy

def load_face_dataset(face_image_file_path):
    face_image = fr.load_image_file(face_image_file_path)
    face_location = fr.face_locations(face_image)
    face_image_encoding = fr.face_encodings(face_image)[0]
    return face_image_encoding

images = ['test.jpg', 'test2.jpg', 'test3.jpg']
known_face_encodings = [load_face_dataset("./img/Joe_Biden.jpg")]
known_face_names = ["Joe Biden"]

while True:
    # Performs facial recognition on each of the images in the images list
    for image in images:
        img = cv2.imread(image, 1)
        face_location = fr.face_locations(img)
        face_encoding = fr.face_encodings(img, face_location)

        face_names = []
        for fe in face_encoding:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(known_face_encodings, fe)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, fe)
            best_match_index = numpy.argmin(face_distances)
            if matches[best_match_index]: name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_location, face_names):
                # Draw a box around the face
                cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (left - 20, bottom + 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        
        # Shows a unique window for each image
        cv2.imshow('Image Recognition: ' + image, img)

    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()