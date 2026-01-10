#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Duuba-AI Cocoa detection Streamlit app
# This script loads a trained TensorFlow Object Detection model, accepts an uploaded
# image via Streamlit, runs inference to detect cocoa pods (healthy vs infected),
# and visualizes results.

# Standard library imports
from distutils.sysconfig import get_python_inc
import os

# Third-party imports
import streamlit as st

# -----------------------------
# Configuration / constants
# -----------------------------
# Model and file names used by the app
CUSTOM_MODEL_NAME = 'my_ssd_mobnetpod'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# File system layout used throughout the project. Change these if your workspace moves.
paths = {
    'WORKSPACE_PATH': os.path.join('Cocoa','Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Cocoa','Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Cocoa','Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join(CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join(CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join(CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('Cocoa','Tensorflow','protoc')
}

# Helpful file paths derived from `paths`
files = {
    'PIPELINE_CONFIG': os.path.join(CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# -----------------------------
# Optional profile links (set these to your URLs to show links in the sidebar)
# -----------------------------
# Example: PROFILE_LINKEDIN = 'https://www.linkedin.com/in/your-profile'
#          PROFILE_TWITTER = 'https://twitter.com/your-handle'
PROFILE_LINKEDIN = 'https://www.linkedin.com/in/mubarak-kwanda/'
PROFILE_TWITTER = 'https://twitter.com/mbrkkwanda'
# Default model URL (Google Drive share link provided by user)
DEFAULT_MODEL_URL = 'https://drive.google.com/file/d/1Biw-K0DlbOAxGVEm4wy2LLGdDSuZl1Wg/view?usp=sharing'
# Ensure workspace folders exist (this block can be skipped if folders already present)
for path in paths.values():
    if not os.path.exists(path):
        # These lines were originally written for notebook environments; keep as-is
        if os.name == 'posix':
            os.system(f'mkdir -p {path}')
        if os.name == 'nt':
            os.system(f'mkdir "{path}"')


# Optional setup steps commented out (git clone of TF models, protobuf install, etc.)
# They are useful when preparing a new environment from scratch.
# if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
#     get_ipython().system("git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")
# get_ipython().system('pip install protobuf==3.19.6')


# -----------------------------
# Object detection imports and label map creation
# -----------------------------
import object_detection

# Labels used by this model; update if label ids or names change.
labels = [{'name':'Healthy Pod', 'id':1}, {'name':'Infected Pod', 'id':2}]

# Write the label map file used by the TF OD API
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# TFRecord generation steps are left commented (this project includes training utilities)
# if not os.path.exists(files['TF_RECORD_SCRIPT']):
#     get_ipython().system("git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")


# -----------------------------
# Update pipeline config for transfer learning
# -----------------------------
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import subprocess
from pathlib import Path

# Read the pipeline config and merge into protobuf object
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Update a few fields for this fine-tuning setup (num classes, checkpoint, input paths)
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-3')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

# Persist the edited pipeline config back to disk
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)


# -----------------------------
# Load trained detection model from checkpoint
# -----------------------------
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util as od_config_util

# Build the model and restore weights from checkpoint (cached for fast reruns)
@st.cache_resource
def load_model():
    """Load and restore the detection model from checkpoint (cached)."""
    configs = od_config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()
    return detection_model

detection_model = load_model()


def model_exists():
    """Return True if the checkpoint or exported SavedModel exists locally."""
    ckpt_dir = Path(paths['CHECKPOINT_PATH'])
    # Check for specific checkpoint files or an exported saved_model
    if ckpt_dir.exists():
        # checkpoint files (ckpt-*.index) or saved_model
        patterns = ['ckpt-3.index', 'saved_model.pb', 'pipeline.config']
        for p in patterns:
            if (ckpt_dir / p).exists():
                return True
        # also check inside export/saved_model
        if (ckpt_dir / 'export' / 'saved_model' / 'saved_model.pb').exists():
            return True
    return False


def ensure_model_available():
    """Ensure model files are present; if not, attempt to download using MODEL_URL env var.

    Raises RuntimeError if model is still missing after attempted download.
    """
    if model_exists():
        return

    model_url = os.environ.get('MODEL_URL') or globals().get('DEFAULT_MODEL_URL')
    if not model_url:
        # give a clear error to the user (Streamlit will show this when run)
        raise RuntimeError('Model not found at "{}" and MODEL_URL not provided.'.format(paths['CHECKPOINT_PATH']))

    # Call the downloader script in the repo root
    downloader = Path(__file__).parent / 'download_model.py'
    if not downloader.exists():
        raise RuntimeError('Model missing and downloader not found at {}'.format(downloader))

    # Run the downloader; it will download/extract into the expected folder
    try:
        subprocess.check_call(["python", str(downloader), "--url", model_url])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Downloader failed: {e}')

    if not model_exists():
        raise RuntimeError('Model still not found after downloader ran.')


# Ensure model is available before attempting to load it
try:
    ensure_model_available()
except Exception as e:
    # When running as a module this will propagate; for Streamlit show a user-friendly message
    try:
        st.error(f'Model unavailable: {e}')
    except Exception:
        pass
    raise


@st.cache_resource
def get_detect_fn():
    """Return a cached detect_fn that uses the loaded model."""
    @tf.function
    def detect_fn(image):
        """Run detection model on a single input tensor and return processed detections."""
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    return detect_fn

detect_fn = get_detect_fn()


# -----------------------------
# Image I/O and helper functions
# -----------------------------
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import io
import warnings
warnings.filterwarnings("ignore")

# Create a mapping from label ids to display names for visualization
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


def format_score(score, digits=2):
    """Safely convert various numeric types to float and round to `digits`.

    Returns a float (rounded) or 0.0 on failure.
    """
    try:
        return round(float(score), digits)
    except Exception:
        try:
            return float(score)
        except Exception:
            return 0.0


def healthy_msg(score1, score2):
    """Display messages/sidebars when a pod is classified as healthy.

    score1, score2: confidence scores (percent) for the top detections.
    """
    s1 = format_score(score1, 2)
    s2 = format_score(score2, 2)
    if s1 > 50.00:
        st.write(f' {s1}% HEALTHY')
        st.sidebar.write("Accuracy Metric: Above 50%")
        st.sidebar.success(f'Accuracy: {s1}% HEALTHY')
    else:
        # No action for low-confidence healthy detection
        pass


def infected_msg(score1, score2):
    """Display messages/sidebars when a pod is classified as infected."""
    s1 = format_score(score1, 2)
    s2 = format_score(score2, 2)
    if s1 > 50.00:
        st.write(f'Accuracy:  {s1}% INFECTED')
        st.sidebar.write("Accuracy Metric: Above 50%")
        st.sidebar.error(f'Accuracy:  {s1}% INFECTED')
    else:
        pass


# -----------------------------
# Streamlit app layout
# -----------------------------
with st.sidebar:
    st.title("Duuba-AI - Cocoa 🍍 Disease Detector")
    st.subheader("Accurate detection of Infected or Healthy Cocoa pods.")
    st.write("Classification: 1 = HEALTHY / 2 = INFECTED")
    st.warning("⚠️ **Under Development** — This model is in pilot phase and may have errors. Use results for reference only.")
    # Show profile links if configured (set PROFILE_GITHUB and PROFILE_LINKEDIN near top of file)
        # (Profile links moved to footer)

# Sidebar status placeholder for processing alerts
status_placeholder = st.sidebar.empty()

st.title("Duuba-AI - Cocoa 🍍 Monitoring made Easy")
st.info("📌 **Pilot Model** — This detection model is under active development. Predictions may not be 100% accurate. Please validate results independently.")


# live camera view removed (was here). To re-enable, reintroduce the function
# and the UI control. Keeping this comment to make it easy to restore later.



# -----------------------------
# Image upload flow
# -----------------------------
IMAGE_PATH = st.file_uploader("Choose an Image", type=["jpg", "png"])
if IMAGE_PATH is not None:
    # Read and display the uploaded image
    img_raw = Image.open(IMAGE_PATH).resize((400, 400))
    st.image(img_raw, use_column_width=False)

    # Read uploaded file bytes directly and decode with OpenCV (avoids filesystem writes)
    file_bytes = np.asarray(bytearray(IMAGE_PATH.getbuffer()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error('Failed to decode uploaded image')
        raise RuntimeError('cv2.imdecode returned None')
    image_np = img

    # Notify user and run inference with spinner
    status_placeholder.info('Starting detection...')
    try:
        with st.spinner('Running detection — this may take a few seconds...'):
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
        status_placeholder.success('Detection finished')
    except Exception as e:
        status_placeholder.error(f'Detection failed: {e}')
        raise
    finally:
        # keep status for a short while then clear (non-blocking)
        pass 

    # Post-process detection  utputs
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Extract top detection classes and scores (convert to percent)
    det_num = detections['detection_classes'] + label_id_offset
    det_score = detections['detection_scores']
    det_score1 = det_score[0] * 100
    det_score2 = det_score[1] * 100

    # Decide classification based on detections that meet a confidence threshold.
    # Provide a sidebar slider so the user can tune the threshold to match the visualization.
    conf_thresh = st.sidebar.slider('Confidence threshold', 0.3, 0.95, 0.8, step=0.05)

    # Gather classes and scores (apply label id offset)
    classes = detections['detection_classes'] + label_id_offset
    scores = detections['detection_scores']

    # Find indices of detections above threshold
    valid_idxs = [i for i, s in enumerate(scores) if s >= conf_thresh]

    if not valid_idxs:
        st.write('No detection seen above the confidence threshold. Try lowering the threshold.')
    else:
        # Aggregate scores per class (sum of confidences) to pick the dominant class
        agg = {}
        for i in valid_idxs:
            cls = int(classes[i])
            agg[cls] = agg.get(cls, 0.0) + float(scores[i])

        # Choose class with highest aggregated score
        chosen_cls = max(agg, key=agg.get)
        # Compute a representative score for display (max score among that class)
        chosen_score = max(float(scores[i]) for i in valid_idxs if int(classes[i]) == chosen_cls) * 100

        if chosen_cls == 1:
            st.balloons()
            st.sidebar.success("Classification:  HEALTHY POD")
            healthy_msg(chosen_score, 0)
        elif chosen_cls == 2:
            st.sidebar.warning("Classification: INFECTED POD")
            infected_msg(chosen_score, 0)
        else:
            st.write('No detection seen Please upload new image')

    # Visualize bounding boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    # Enhance visualization: draw thicker boxes and label backgrounds for better visibility
    min_score = 0.8  # same threshold used in viz_utils call above
    h, w, _ = image_np_with_detections.shape
    for i in range(min(5, int(detections.get('num_detections', 0)))):
        score = float(detections['detection_scores'][i])
        if score < min_score:
            continue
        box = detections['detection_boxes'][i]
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
        cls = int(detections['detection_classes'][i]) + label_id_offset
        class_name = category_index.get(cls, {'name': 'N/A'})['name']
        label = f"{class_name}: {int(score * 100)}%"
        # Pick color based on class (green for healthy, red for infected)
        color = (0, 200, 0) if class_name.lower().startswith('healthy') or cls == 1 else (0, 0, 200)
        # Draw thick rectangle
        cv2.rectangle(image_np_with_detections, (left, top), (right, bottom), color, thickness=4)
        # Draw filled rectangle for label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image_np_with_detections, (left, max(0, top - text_h - 10)), (left + text_w, top), color, -1)
        cv2.putText(image_np_with_detections, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Convert to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

    # Prepare a presentable 800x800 final image using PIL
    try:
        pil_img = Image.fromarray(annotated_rgb)

        # Resize while preserving aspect ratio to fit within 800x800
        target_size = (800, 800)
        pil_copy = pil_img.copy()
        pil_copy.thumbnail(target_size, Image.LANCZOS)

        # Create white background and paste centered
        final_img = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - pil_copy.width) // 2
        paste_y = (target_size[1] - pil_copy.height) // 2
        final_img.paste(pil_copy, (paste_x, paste_y))

        # Draw a small legend and title on the final image
        draw = ImageDraw.Draw(final_img)
        try:
            font = ImageFont.truetype('arial.ttf', 18)
            font_small = ImageFont.truetype('arial.ttf', 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Title
        title = 'Detections'
        tw, th = draw.textsize(title, font=font)
        draw.text(((target_size[0] - tw) / 2, 8), title, fill=(0, 0, 0), font=font)

        # Legend (top-left corner)
        legend_x = 12
        legend_y = 40
        legend_gap = 6
        legend_items = [(1, 'Healthy Pod', (0, 200, 0)), (2, 'Infected Pod', (200, 0, 0))]
        box_size = 16
        for idx, name, color in legend_items:
            draw.rectangle([legend_x, legend_y, legend_x + box_size, legend_y + box_size], fill=color)
            draw.text((legend_x + box_size + 6, legend_y - 2), name, fill=(0, 0, 0), font=font_small)
            legend_y += box_size + legend_gap

        # Convert final image to bytes for Streamlit display and download
        buf = io.BytesIO()
        final_img.save(buf, format='PNG')
        buf.seek(0)
        img_bytes = buf.getvalue()

        # Show the 800x800 presentable image
        st.image(final_img, caption='Annotated (800x800)', use_column_width=False, width=800)

        # Provide downloads: full-size PNG and open in new tab link
        st.download_button('Download annotated (800x800)', data=img_bytes, file_name='annotated_800.png', mime='image/png')
        data_url = "data:image/png;base64," + __import__('base64').b64encode(img_bytes).decode('utf-8')
        st.markdown(f"[Open annotated image in new tab]({data_url})", unsafe_allow_html=True)

    except Exception as e:
        # Fallback to original display if anything goes wrong
        st.image(annotated_rgb, caption='Detections', use_column_width=True)
        st.write('Could not create 800x800 presentable image:', e)

# -----------------------------
# Footer: show profile links if configured
# -----------------------------
try:
    if PROFILE_LINKEDIN or PROFILE_TWITTER:
        # small inline SVG icons (kept simple & lightweight)
        linkedin_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
                        'style="vertical-align:middle;margin-right:6px;fill:#0A66C2"><title>LinkedIn</title>'
                        '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.038-1.852-3.038-1.853 0-2.135 1.446-2.135 2.94v5.667H9.351V9h3.414v1.561h.049c.476-.9 1.637-1.852 3.369-1.852 3.603 0 4.268 2.372 4.268 5.459v6.284zM5.337 7.433a2.062 2.062 0 1 1 0-4.124 2.062 2.062 0 0 1 0 4.124zM6.962 20.452H3.712V9h3.25v11.452z"/></svg>')

        x_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
                 'style="vertical-align:middle;margin-right:6px;fill:#000000"><title>X</title>'
                 '<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24h-6.6l-5.17-6.76-5.91 6.76h-3.308l7.73-8.835L2.42 2.25h6.76l4.69 6.231 5.386-6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>')

        parts = []
        if PROFILE_LINKEDIN:
            parts.append(f'<a href="{PROFILE_LINKEDIN}" target="_blank" rel="noopener" '
                         f'style="text-decoration:none;color:inherit;margin:0 8px">{linkedin_svg}<span style="vertical-align:middle">LinkedIn</span></a>')
        if PROFILE_TWITTER:
            parts.append(f'<a href="{PROFILE_TWITTER}" target="_blank" rel="noopener" '
                         f'style="text-decoration:none;color:inherit;margin:0 8px">{x_svg}<span style="vertical-align:middle">X</span></a>')

        footer_html = ('<div style="text-align:center; margin-top:20px;">' + ' | '.join(parts) +
                       '<br><small style="color:#666">Connect with the project owner</small></div>')
        st.markdown('---')
        st.markdown(footer_html, unsafe_allow_html=True)
except Exception:
    pass
