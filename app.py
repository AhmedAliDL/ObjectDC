import os
import cv2
import streamlit as st
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from PIL import Image
import subprocess
import shutil

st.set_page_config(
    page_title="Object Detection",
    page_icon="🤖",
    layout="wide"

)
st.markdown("""
<style>
.css-1rs6os.edgvbvh3
{
  visibility:hidden;
}
.css-10pw50.egzxvld1
{
 visibility:hidden;
}
<style/>
""", unsafe_allow_html=True)


def convert_avi_to_mp4(avi_file, mp4_file):
    command = ["ffmpeg", "-i", avi_file, mp4_file]
    subprocess.run(command)


def convert_mp4_to_avi(avi_file, mp4_file):
    command = ["ffmpeg", "-i", mp4_file, avi_file]
    subprocess.run(command)


def clear_session():
    st.session_state.clear()
    if os.path.exists("runs"):
        shutil.rmtree("runs", ignore_errors=True)
    if os.path.exists("user"):
        shutil.rmtree("user", ignore_errors=True)


def get_keys(names, values):
    res = []
    for key, value in names.items():
        for target in values:
            if value == target:
                res.append(key)
    return res


def count_objects(model, video_path, classes, output_dir):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    line_points = [(100, 400), (1900, 400)]  # line or region points
    classes_to_count = classes  # person and car classes for count

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=line_points,
                     classes_names=model.names,
                     draw_tracks=True)
    images = []
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False,
                             classes=classes_to_count)

        im0 = counter.start_counting(im0, tracks)
        images.append(im0)

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_dir, fps=fps)


def main():
    st.title("Object Detection")
    task = st.selectbox("Select type of file", [None, "Image", "Video"], on_change=clear_session)
    # Load a model
    model = YOLO("yolov8x.pt")
    classes = model.names
    if task == "Image":
        files = st.file_uploader("Upload your image:", accept_multiple_files=True,
                                 type=["jpg",
                                       "jpeg",
                                       "png"])

        if files:
            for file in files:
                if model:
                    image = Image.open(file)
                    if not os.path.exists("user"):
                        os.makedirs("user")
                    if not os.path.exists(f"user/{file.name.split('.')[0]}.png"):
                        image.save(f"user/{file.name.split('.')[0]}.png")

                    st.subheader("Original Image")
                    st.image(file)
                    st.subheader("Detected Image")
                    if st.session_state.counter != 1:
                        if not os.path.exists(
                                fr"runs\detect\predict{st.session_state.counter}\{file.name.split('.')[0]}.png"):
                            model.predict(f"user/{file.name.split('.')[0]}.png",
                                          save=True)
                        if os.path.exists(
                                fr"runs\detect\predict{st.session_state.counter}\{file.name.split('.')[0]}.png"):
                            st.image(
                                fr"runs\detect\predict{st.session_state.counter}\{file.name.split('.')[0]}.png")
                    else:
                        if not os.path.exists(fr"runs\detect\predict\{file.name.split('.')[0]}.png"):
                            model.predict(f"user/{file.name.split('.')[0]}.png",
                                          save=True)

                        if os.path.exists(fr"runs\detect\predict\{file.name.split('.')[0]}.png"):
                            st.image(fr"runs\detect\predict\{file.name.split('.')[0]}.png")

            st.session_state.counter += 1

    elif task == "Video":
        files = st.file_uploader("Upload your video:",
                                 accept_multiple_files=True,
                                 type=["mp4"])
        c_d_task = st.selectbox("Select from count object or detect object:",
                                [None, "count", "detect"], on_change=clear_session)
        if files:
            for file in files:
                if not os.path.exists("user"):
                    os.makedirs("user")
                if not os.path.exists(f"user/{file.name.split('.')[0]}.mp4"):
                    with open(f"user/{file.name.split('.')[0]}.mp4", "wb") as f:
                        f.write(file.read())
                st.subheader("Original Video")
                st.video(file)
                if c_d_task == "count":
                    objects = st.multiselect("Select objects:",
                                             list(classes.values()), on_change=clear_session)
                    classes_n = get_keys(classes, objects)
                    proc = st.button("Process")
                    if classes_n and proc:
                        if not os.path.exists(f"user/{file.name.split('.')[0]}c.mp4"):
                            count_objects(model=model,
                                          video_path=f"user/{file.name.split('.')[0]}.mp4",
                                          classes=classes_n, output_dir=f"user/{file.name.split('.')[0]}c.mp4")

                        if os.path.exists(f"user/{file.name.split('.')[0]}c.mp4"):
                            st.subheader("object counted")
                            st.video(f"user/{file.name.split('.')[0]}c.mp4")
                elif c_d_task == "detect":
                    st.subheader("Detected Video")
                    counter = len(os.listdir(r"runs\detect"))
                    if counter > 1:
                        if not os.path.exists(
                                fr"runs\detect\predict\{file.name.split('.')[0]}.mp4") \
                                and not os.path.exists(fr"runs\detect\predict\{file.name.split('.')[0]}.mp4"):
                            model.predict(f"user/{file.name.split('.')[0]}.mp4",
                                          save=True)
                            convert_avi_to_mp4(
                                fr"runs\detect\predict{counter}\{file.name.split('.')[0]}.avi",
                                fr"runs\detect\predict{counter}\{file.name.split('.')[0]}.mp4")
                        if os.path.exists(
                                fr"runs\detect\predict{counter}\{file.name.split('.')[0]}.mp4"):
                            st.video(
                                fr"runs\detect\predict{counter}\{file.name.split('.')[0]}.mp4",
                            )
                            convert_avi_to_mp4(
                                fr"runs\detect\predict{counter}\{file.name.split('.')[0]}.avi",
                                fr"runs\detect\predict{counter}\{file.name.split('.')[0]}.mp4")
                        elif os.path.exists(fr"runs\detect\predict\{file.name.split('.')[0]}.mp4"):
                            st.video(fr"runs\detect\predict\{file.name.split('.')[0]}.mp4")

                    else:
                        if not os.path.exists(fr"runs\detect\predict\{file.name.split('.')[0]}.mp4"):
                            model.predict(f"user/{file.name.split('.')[0]}.mp4",
                                          save=True)
                            convert_avi_to_mp4(
                                fr"runs\detect\predict\{file.name.split('.')[0]}.avi",
                                fr"runs\detect\predict\{file.name.split('.')[0]}.mp4")

                        if os.path.exists(fr"runs\detect\predict\{file.name.split('.')[0]}.mp4"):
                            st.video(fr"runs\detect\predict\{file.name.split('.')[0]}.mp4")


if __name__ == "__main__":
    main()
