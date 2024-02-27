import os
import streamlit as st
from deepface import DeepFace
from PIL import Image
import pickle

path = "datasets/"
img_list = []
for image in os.listdir(path):
    img = os.path.join(path, image)
    img_list.append(img)
    print(image)
pickle.dump(img_list, open('images.pkl', "wb"))

imgs = pickle.load(open(r'images.pkl', 'rb'))

# Main Function
def predict_actor(input_image_path, dataset_folder):
    predicted_actors = []
    threshold = 0.35
    
    # Load input image only once
    # try:
    #     input_results = DeepFace.represent(input_image_path)
    #     input_embedding = input_results[0]["embedding"]
    # except ValueError as e:
    #     print("Error processing input image:", e)
    #     return predicted_actors
    
    # Process dataset images
    progress_bar = st.progress(0)
    with st.spinner("Verifying actors..."):
        for i, actor_image_name in enumerate(imgs):
            actor_image_path = actor_image_name
            
            # Update progress bar
            progress_percent = (i + 1) / len(imgs) * 100
            progress_bar.progress(int(progress_percent))
            
            # Use DeepFace to verify similarity between input image and actor image
            try:
                # result = DeepFace.verify(input_embedding, actor_image_path)
                result = DeepFace.verify(input_image_path, actor_image_path)
                dis_similarity_score = result["distance"]
                if dis_similarity_score < threshold:
                    predicted_actors.append(actor_image_name)
            except ValueError as e:
                print("Error processing", actor_image_path, ":", e)

    return predicted_actors


def main():
    st.title("Actor Recognition App")

    # Image Upload
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        sav_img = Image.open(uploaded_image).convert('RGB')
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image",
                 use_column_width=True)
        sav_img.save("uploads/temp.jpg")
        # Predict Actors
        dataset_folder = 'datasets/'
        predicted_actors = predict_actor("uploads/test.jfif", dataset_folder)

        # Commandline Testing
        # for i in predicted_actors:
        #   print(str(i).split('.')[0].split('/')[1])

        os.remove('uploads/temp.jpg')
        # Display Predictions
        if predicted_actors:
            st.write("Predicted Actor(s):")
            for actor in predicted_actors:
                st.write(actor)
        else:
            st.write("No actor found in the dataset.")


# Run the Streamlit app
if __name__ == "__main__":
    # Intel Support (OMP)
    # os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
    main()
