# Actor Recognition

Predicts the image of the few actors from the custom dataset

## Dataset Desc

The dataset contains images of famous Indian actors. Each actor is represented by a single image, and all images are in color format (RGB).
[actor_name].[jfif/jpg] - The images are collected by scraping websites like IMDb, Wikipedia and other social media platforms.

## Run Command

On Terminal

```
cd Actor-Recognition
streamlit run name_the_face.py
```

This will open a web page on your browser, where you can upload an image and it will predict which actor is in that image.
On Colab

```
streamlit run name_the_actor.py &>/content/actors_log.txt & npx localtunnel --port 8501
```

Open in Browser `Your_Local_Tunnel_URL`
